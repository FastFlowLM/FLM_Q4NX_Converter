from __future__ import annotations
from gguf.constants import GGMLQuantizationType
from gguf import dequantize, quantize
from gguf.constants import GGML_QUANT_SIZES
import gc
from typing import Tuple
import numpy as np
import torch

class GGUFTensor:
    name: str
    shape: Tuple[int, ...]
    data: np.ndarray
    tensor_type: GGMLQuantizationType

    def __init__(self, name: str, shape: Tuple[int, ...], data: np.ndarray, tensor_type: GGMLQuantizationType):
        self.name = name
        self.shape = shape
        self.data = data
        self.tensor_type = tensor_type

    @staticmethod
    def _safe_from_numpy(array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor.
        
        Preserves read-only semantics — no copy unless mutation is required downstream.
        PyTorch handles read-only tensors safely for all arithmetic/view ops.
        """
        return torch.from_numpy(array)

    @staticmethod
    def unpack_q4_0(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
        data = tensor.view(np.uint8)
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d_raw, qs_raw = np.hsplit(blocks, [2])
        # Zero-copy reinterpretation to float16 — scales stay read-only (safe for reading!)
        d_f16 = d_raw.view(np.float16)

        # Dequantize nibbles into int8: minimal RAM footprint
        qs = qs_raw.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        # Convert to Torch — no forced copy; preserves read-only semantics
        d_torch = GGUFTensor._safe_from_numpy(d_f16)
        qs_torch = GGUFTensor._safe_from_numpy(qs)
        m_torch = torch.zeros_like(d_torch)

        return (
            d_torch.view(-1, columns // block_size),
            m_torch.view(-1, columns // block_size),
            qs_torch.view(-1, columns)
        )

    @staticmethod
    def unpack_q4_1(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_1]
        data = tensor.view(np.uint8)
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))

        d_raw, rest = np.hsplit(blocks, [2])
        m_raw, qs_raw = np.hsplit(rest, [2])

        # Zero-copy reinterpretation — no .astype(np.float32) here!
        d_f16 = d_raw.view(np.float16)
        m_f16 = m_raw.view(np.float16)

        qs = qs_raw.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1))

        # Convert to Torch — _safe_from_numpy avoids copy if read-only
        d_torch = GGUFTensor._safe_from_numpy(d_f16)
        m_torch = GGUFTensor._safe_from_numpy(m_f16)
        qs_torch = GGUFTensor._safe_from_numpy(qs)

        assert columns % block_size == 0, "Columns must be divisible by block size"

        return (
            d_torch.view(-1, columns // block_size),
            m_torch.view(-1, columns // block_size),
            qs_torch.view(-1, columns)
        )

    @staticmethod
    def unpack_q8_0(tensors: np.ndarray, columns: int):
        byte_per_blocks = 34
        assert tensors.dtype == np.uint8 or tensors.dtype == np.int8, "Input must be np.uint8 or np.int8"
        original_shape = tensors.shape
        assert original_shape[-1] % byte_per_blocks == 0, "The last dimension must be a multiple of 34"

        blocks = tensors.reshape(*original_shape[:-1], -1, byte_per_blocks)

        # Zero-copy reinterpretation — scales stay read-only (safe for reading!)
        scales = blocks[..., 0:2].view(np.float16)
        data = blocks[..., 2:].view(np.int8)

        Q8_block_size = 32
        # Reshape directly — PyTorch handles non-contiguous NumPy arrays fine (no ascontiguousarray!)
        return (
            GGUFTensor._safe_from_numpy(scales).reshape(-1, columns // Q8_block_size),
            GGUFTensor._safe_from_numpy(scales).reshape(-1, columns // Q8_block_size),  # d,m compatibility
            GGUFTensor._safe_from_numpy(data).reshape(-1, columns)
        )

    @staticmethod
    def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
        bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)

    @staticmethod
    def reverse_transform_nibble_layout(tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the custom nibble layout transformation.
        
        Memory note: This uses torch.cat and intermediate tensors, but MXFP4 blocks are small (17B),
        so total overhead is ~2GB even for 30B-parameter models — acceptable on 64GB RAM.
        """
        assert tensor.dtype == torch.uint8
        assert tensor.shape[-1] == 16

        # 1. Reverse the final nibble swap
        t_lo = tensor & 0x0F
        t_hi = tensor & 0xF0
        interleaved = (t_lo << 4) | (t_hi >> 4)

        # 2. De-interleave the nibbles from abababab... back to aaaa...bbbb...
        nibbles_a_parts = interleaved & 0xF0
        nibbles_b_parts = interleaved & 0x0F

        # Reconstruct blk_a by packing high nibbles
        blk_a = nibbles_a_parts[..., 0::2] | (nibbles_a_parts[..., 1::2] >> 4)
        
        # Reconstruct blk_b by packing low nibbles
        blk_b = (nibbles_b_parts[..., 0::2] << 4) | nibbles_b_parts[..., 1::2]

        deinterleaved = torch.cat((blk_a, blk_b), dim=-1)

        # 3. Reverse the initial nibble swap
        t_lo = deinterleaved & 0x0F
        t_hi = deinterleaved & 0xF0
        original_tensor = (t_lo << 4) | (t_hi >> 4)

        return original_tensor

    @staticmethod
    def split_ggml_mxfpx_to_scale_blocks(structured_data: np.ndarray):
        """Split GGML MXFP4 data into scales and data blocks

        Format per block (17 bytes):
            - 1 byte: scale (uint8)
            - 16 bytes: 32 x 4-bit float values (2 exponent bits + 1 mantissa bit each)
        """
        
        assert structured_data.dtype == np.uint8 or structured_data.dtype == np.int8, "Input must be np.uint8 or np.int8"

        original_shape = structured_data.shape
        assert original_shape[-1] % 17 == 0, "The last dimension must be a multiple of 17"
        
        # Reshape the last dimension into blocks of 17 bytes — ZERO COPY
        blocks = structured_data.reshape(-1, 17)
        
        scales = blocks[:, 0]
        data_bytes = blocks[:, 1:]
        
        # Reverse nibble layout for each block
        reversed_data = GGUFTensor.reverse_transform_nibble_layout(torch.from_numpy(data_bytes))
        
        return scales, reversed_data.numpy()

    @staticmethod
    def unpack_mxfp4(tensor: np.ndarray, columns: int) -> Tuple[np.ndarray, torch.Tensor]:
        """MXFP4 unpack — returns numpy scales and torch data."""
        scales_np, data_torch = GGUFTensor.split_ggml_mxfpx_to_scale_blocks(tensor)
        return scales_np, data_torch

    def dequantize(self) -> torch.Tensor:
        """
        Dequantizes the tensor. 
        For F32/F16/BF16, this is a zero-copy view.
        For quantized types, it promotes to bfloat16.
        """
        # Fast path for non-quantized types
        if self.tensor_type == GGMLQuantizationType.F32:
            return self._safe_from_numpy(self.data.view(np.float32)).contiguous()
        
        if self.tensor_type == GGMLQuantizationType.F16:
            return self._safe_from_numpy(self.data.view(np.float16)).view(torch.float16)
        
        if self.tensor_type == GGMLQuantizationType.BF16:
            return self._safe_from_numpy(self.data.view(np.uint16)).view(torch.bfloat16)

        # Quantized path
        w_np = dequantize(self.data, self.tensor_type)
        # GGUF dequantize returns float32; cast to BF16 immediately to save 2x RAM
        w_torch = torch.from_numpy(w_np).to(torch.bfloat16)
        
        del w_np
        return w_torch

    def unpack(self, default_tensor_type: GGMLQuantizationType) -> Tuple[torch.Tensor, ...]:
        """
        Unpacks GGUF data into raw Q4NX-ready components.
        """
        # 1. Non-quantized fast paths
        if self.tensor_type == GGMLQuantizationType.F32:
            return (self._safe_from_numpy(self.data.view(np.float32)),)
            
        if self.tensor_type == GGMLQuantizationType.F16:
            return (self._safe_from_numpy(self.data.view(np.float16)).view(torch.float16),)
            
        if self.tensor_type == GGMLQuantizationType.BF16:
            return (self._safe_from_numpy(self.data.view(np.uint16)).view(torch.bfloat16),)

        # 2. Native quantized paths
        if self.tensor_type == GGMLQuantizationType.Q4_0:
            return self.unpack_q4_0(self.data, self.shape[0])
            
        if self.tensor_type == GGMLQuantizationType.Q4_1:
            return self.unpack_q4_1(self.data, self.shape[0])
            
        if self.tensor_type == GGMLQuantizationType.Q8_0:
            return self.unpack_q8_0(self.data, self.shape[0])
            
        if self.tensor_type == GGMLQuantizationType.MXFP4:
            # unpack_mxfp4 returns (scales_np, data_torch)
            scales_np, data_torch = self.unpack_mxfp4(self.data, self.shape[0])
            return (self._safe_from_numpy(scales_np), data_torch)

        # 3. Fallback for mixed-quantization (e.g., Q4_K_M -> Q4_0)
        # We dequantize and then re-quantize using the default type.
        # Note: This is a heavy path; force_pack_q8_to_q4nx_size is preferred for large tensors.
        w_bf16 = self.dequantize()
        data_q = quantize(w_bf16.cpu().numpy(), default_tensor_type)
        
        del w_bf16
        gc.collect()

        if default_tensor_type == GGMLQuantizationType.Q4_1:
            return self.unpack_q4_1(data_q, self.shape[0])
        elif default_tensor_type == GGMLQuantizationType.Q4_0:
            return self.unpack_q4_0(data_q, self.shape[0])
        elif default_tensor_type == GGMLQuantizationType.Q8_0:
            return self.unpack_q8_0(data_q, self.shape[0])
        
        raise ValueError(f"Unsupported fallback target: {default_tensor_type.name}")