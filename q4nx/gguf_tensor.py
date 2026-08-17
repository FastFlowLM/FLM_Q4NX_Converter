from __future__ import annotations
from gguf.constants import GGMLQuantizationType
from gguf import dequantize, quantize
from gguf.constants import GGML_QUANT_SIZES

from typing import Tuple

from dataclasses import dataclass
from mpmath.libmp import int_types
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
    def unpack_q4_0(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
        data = tensor.view(np.uint8)
        shape = data.shape
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        d = torch.from_numpy(d)
        m = torch.zeros_like(d)
        qs = torch.from_numpy(qs)
        d = d.view(-1, columns // block_size)
        m = m.view(-1, columns // block_size)
        qs = qs.view(-1, columns)

        return d, m, qs

    @staticmethod
    def unpack_q4_1(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_1]
        data = tensor.view(np.uint8)
        shape = data.shape
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)

        d = torch.from_numpy(d).contiguous()
        m = torch.from_numpy(m).contiguous()
        qs = torch.from_numpy(qs).contiguous()
        assert columns % block_size == 0, "Columns must be divisible by block size"

        d = d.view(-1, int(columns // block_size))
        m = m.view(-1, int(columns // block_size))
        qs = qs.view(-1, int(columns))

        return d, m, qs
    
    @staticmethod
    def unpack_q8_0(tensors:np.ndarray, columns:int):
        """Split GGML Q8_0 data into scales and quantized values

        Parameters
        ----------
        tensors : np.ndarray
            Q8_0 tensor data
            Format per block (34 bytes):
                - 2 bytes: scale (float16)
                - 32 bytes: 32 x 8-bit quantized values

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            scales, data

        Raises
        ------
        ValueError
            _description_
        """
        byte_per_blocks = 34
        assert tensors.dtype == np.uint8 or tensors.dtype == np.int8, "Input must be np.uint8 or np.int8"
        original_shape = tensors.shape
        assert original_shape[-1] % byte_per_blocks == 0, "The last dimension must be a multiple of 34"
        
        
        blocks = tensors.reshape(*original_shape[:-1], -1, byte_per_blocks)
        
        # Use view() to reinterpret the bytes as float16 bits, not convert the values
        scales = blocks[..., 0:2].view(np.float16) 
        
        # Use view() to reinterpret bytes as signed int8
        data = blocks[..., 2:].view(np.int8)
        
        Q8_block_size = 32
        # Reshape to (rows, cols_per_type) using columns, matching unpack_q4_1 output shape
        scales = np.ascontiguousarray(scales).reshape(-1, columns // Q8_block_size)
        data = np.ascontiguousarray(data).reshape(-1, columns)
        
        return torch.from_numpy(scales.copy()), torch.from_numpy(scales.copy()), torch.from_numpy(data.copy())
        
        
    
    
    @staticmethod
    def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
        bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)

    @staticmethod
    def reverse_transform_nibble_layout( tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the custom nibble layout transformation."""
        assert tensor.dtype == torch.uint8
        assert tensor.shape[-1] == 16

        # 1. Reverse the final nibble swap
        t_lo = tensor & 0x0F
        t_hi = tensor & 0xF0
        interleaved = (t_lo << 4) | (t_hi >> 4)

        # 2. De-interleave the nibbles from abababab... back to aaaa...bbbb...
        # The high nibbles of 'interleaved' contain the nibbles for the first half (blk_a)
        nibbles_a_parts = interleaved & 0xF0
        # The low nibbles of 'interleaved' contain the nibbles for the second half (blk_b)
        nibbles_b_parts = interleaved & 0x0F

        # Reconstruct blk_a by packing the high nibbles back together
        # Pair up nibbles: (1st high nibble) | (2nd high nibble >> 4)
        blk_a = nibbles_a_parts[..., 0::2] | (nibbles_a_parts[..., 1::2] >> 4)

        # Reconstruct blk_b by packing the low nibbles back together
        # Pair up nibbles: (1st low nibble << 4) | (2nd low nibble)
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
        
        assert (structured_data.dtype == np.uint8 or structured_data.dtype == np.int8), "Input must be np.uint8 or np.int8"

        original_shape = structured_data.shape
        assert original_shape[-1] % 17 == 0, "The last dimension must be a multiple of 17"
        
        # Reshape the last dimension into blocks of 17 bytes
        blocks = structured_data.reshape(*original_shape[:-1], -1, 17)
        
        # Extract scales (first byte of each block)
        scales = blocks[..., 0].astype(np.uint8)
        
        # Extract data (remaining 16 bytes, keep as uint8 for 4-bit unpacking)
        data = GGUFTensor.reverse_transform_nibble_layout( torch.from_numpy( blocks[..., 1:].astype(np.uint8))).numpy()     
        return scales, data
    
    @staticmethod
    def unpack_mxfp4(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, data = GGUFTensor.split_ggml_mxfpx_to_scale_blocks(tensor)
        return torch.from_numpy(scale), torch.from_numpy(data)

    
    def dequantize(self) -> torch.Tensor:
        w = dequantize(self.data, self.tensor_type)
        w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
        return w

    def get_used_quantization_type(self, default_tensor_type: GGMLQuantizationType) -> GGMLQuantizationType:
        if self.tensor_type in [GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.MXFP4]:
            return self.tensor_type
        else:
            # For unsupported types (including Q8_0, which is not a native
            # Q4NX packing), we will dequantize and then quantize to
            # default_tensor_type. This makes Q8_0-source GGUFs honor the
            # config target (e.g. Q4_1 main matmuls) instead of keeping every
            # weight 8-bit.
            return default_tensor_type

    def unpack(self, default_tensor_type: GGMLQuantizationType) -> np.ndarray:
        # Native floating point tensor types (F32/F16/BF16) can only be
        # returned as-is when the caller does NOT require a quantized
        # target (e.g. norm weights, whose config entry sets
        # default_tensor_type to a float type). If the caller requests a
        # quantized target (Q4_0/Q4_1/Q8_0) -- which happens for ordinary
        # matmul weights when the source GGUF stores every tensor in full
        # precision (no per-tensor quantization at all) -- we must still
        # quantize+pack the tensor into that format below. Otherwise the
        # weight is left oversized and in the wrong (unpacked) layout,
        # which the NPU runtime cannot read and crashes on.
        is_native_float = self.tensor_type in (
            GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        )
        wants_quantized_target = default_tensor_type in (
            GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q8_0
        )
        # The Q4_0/Q4_1/Q8_0 block-quantized formats only make sense for 2D
        # matmul weight matrices; 1D tensors (e.g. rope_freqs, biases) are
        # never packed this way even when a config entry is missing and
        # falls back to the global default_tensor_type. Treat those as
        # native float passthrough regardless of the requested target.
        if len(self.shape) < 2:
            wants_quantized_target = False

        if is_native_float and not wants_quantized_target:
            if self.tensor_type == GGMLQuantizationType.F32:
                return [torch.Tensor(np.array(self.data.view(np.float32)))]
            elif self.tensor_type == GGMLQuantizationType.F16:
                return [torch.Tensor(np.array(self.data.view(np.float16).astype(np.float32)))]
            else:  # BF16
                return [torch.from_numpy(self.data.copy()).view(torch.bfloat16)]

        elif self.tensor_type == GGMLQuantizationType.Q4_0:
            return self.unpack_q4_0(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q4_1:
            return self.unpack_q4_1(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q8_0:
            if default_tensor_type == GGMLQuantizationType.Q8_0:
                return self.unpack_q8_0(self.data, self.shape[0])
            # Q8_0 is not the runtime's native packing when the config asks
            # for a different target (e.g. Q4_1 main matmuls): dequantize and
            # re-quantize into the requested format before packing.
            return self._requantize_to(default_tensor_type)
        elif self.tensor_type == GGMLQuantizationType.MXFP4:
            return self.unpack_mxfp4(self.data, self.shape[0])
        else:
            """
                If the tensor type is not natively packable as-is (either a
                truly unsupported GGUF quant type such as Q4_K/Q5_K/Q6_K, or
                a native F32/F16/BF16 tensor that the caller wants quantized
                to a Q4_0/Q4_1/Q8_0 target), dequantize it (gguf.dequantize
                already handles F32/F16/BF16 as a passthrough/cast) and then
                quantize it to default_tensor_type before packing.
            """
            # Block-quantized targets (Q4_0/Q4_1/Q8_0) require the last
            # dimension to be a multiple of 32. Tensors that don't fit -- e.g.
            # small F32 auxiliary tensors with a 3-wide axis, common in LFM2
            # (shortconv / ssm-style weights) -- cannot be packed that way, so
            # keep them as a float passthrough rather than crashing in
            # gguf.quantize.
            if wants_quantized_target and self.shape and self.shape[-1] % 32 != 0:
                w = dequantize(self.data, self.tensor_type)
                w = torch.from_numpy(w).contiguous()
                if self.tensor_type == GGMLQuantizationType.BF16:
                    w = w.view(torch.bfloat16)
                return [w]
            return self._requantize_to(default_tensor_type)

    def _requantize_to(self, default_tensor_type: GGMLQuantizationType) -> np.ndarray:
        """Dequantize the source tensor and re-quantize it into the requested
        target format, returning a (d, m, qw) tuple ready for packing."""
        try:
            w = dequantize(self.data, self.tensor_type)
            w = torch.from_numpy(w).contiguous().to(torch.bfloat16)

            if default_tensor_type == GGMLQuantizationType.BF16:
                # BF16 is not a quantized format, so no requantization is
                # needed: just return the dequantized tensor directly,
                # consistent with the native BF16 branch above.
                return [w]

            w = w.to(torch.float32).numpy()
            data_quantized = quantize(w, default_tensor_type).copy()
            if default_tensor_type == GGMLQuantizationType.Q4_1:
                d, m, qw = self.unpack_q4_1(data_quantized, self.shape[0])
            elif default_tensor_type == GGMLQuantizationType.Q4_0:
                d, m, qw = self.unpack_q4_0(data_quantized, self.shape[0])
            elif default_tensor_type == GGMLQuantizationType.Q8_0:
                d, m, qw = self.unpack_q8_0(data_quantized, self.shape[0])
            else:
                raise ValueError(f"Unsupported tensor type: {default_tensor_type.name}")
            return d, m, qw
        except Exception as e:
            print(
                f"[WARN] Could not quantize {self.tensor_type.name} tensor "
                f"(shape {list(self.shape)}) to {default_tensor_type.name}: {e}; "
                "keeping it as a float passthrough instead of crashing."
            )
            try:
                w = dequantize(self.data, self.tensor_type)
                w = torch.from_numpy(np.array(w)).contiguous()
                if self.tensor_type == GGMLQuantizationType.BF16:
                    w = w.view(torch.bfloat16)
                return [w]
            except Exception:
                return None, None, None