from abc import ABC, abstractmethod

# Core GGUF reader + quantization utilities
from gguf import GGUFReader
from q4nx.gguf_tensor import (
    GGUFTensor,
    GGMLQuantizationType,
    dequantize,
    quantize,
)

# Quantization block sizes (required for force_pack_q8_to_q4nx_size)
try:
    from .constants import GGML_QUANT_BLOCK_SIZES as GGML_QUANT_SIZES
except ImportError:
    # Fallback: basic Q4/Q8 block sizes only
    GGML_QUANT_SIZES = {
        GGMLQuantizationType.Q4_0: 32,
        GGMLQuantizationType.Q4_1: 32,
        GGMLQuantizationType.Q8_0: 32,
    }

from .constants import ModelArch, ModelArchNames
from .constants import ModelArchConfigs
from .gguf_tensor import GGUFTensor, GGMLQuantizationType
from typing import List, Dict, Type
import os
import json
import torch.nn.functional as F
from einops import rearrange
import torch
import numpy as np
import tempfile
import shutil
import gc
from .utils import round_up_to_multiple, get_relativeL2, get_relativeL1, get_rmse, get_cosine_similarity, create_dir_if_not_exists
from safetensors.torch import save_file
from q4nx.gguf_tensor import GGUFTensor
from gguf import Q8_0, dequantize, quantize

# Registry to store model classes by architecture
_MODEL_REGISTRY: Dict[ModelArch, Type['__Q4NX_Converter']] = {}

class __Q4NX_Converter(ABC):
    model_arch: ModelArch
    gguf_reader: GGUFReader
    gguf_tensors: Dict[str, GGUFTensor]
    q4nx_tensors: Dict[str, torch.Tensor]
    hidden_size: int
    num_layers: int
    embed_length:int
    q4nx_config: Dict

    row_block_size: int
    col_block_size: int
    parallel_size: int  
    keep_block_in_2D: bool

    default_tensor_type: GGMLQuantizationType
    
    vision_MM_K:int|None
    vision_MM_N:int|None
    
    forward_name_map: Dict[str, str]
    backward_name_map: Dict[str, str]
    tensor_q4nx_type_map: Dict[str, GGMLQuantizationType]

    def __init__(self):
        raise TypeError("This class is virtual, do not instantiate it directly")

    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.model_arch = model_arch
        _MODEL_REGISTRY[model_arch] = cls

    def initialize(self):
        self._read_gguf_tensors()
        self._read_gguf_metadata()
        self._load_config()

    @abstractmethod
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        pass

    def _offload_to_disk(self, tensor: torch.Tensor) -> torch.Tensor:
        # 1. Setup temp dir
        if not hasattr(self, "_mmap_temp_dir"):
            mmap_base = os.environ.get("TMPDIR", tempfile.gettempdir())
            self._mmap_temp_dir = tempfile.mkdtemp(prefix="q4nx_mmap_", dir=mmap_base)
            self._mmap_counter = 0

        # 2. Prepare tensor: CPU + contiguous
        t_cpu = tensor.detach().cpu().contiguous()
        dtype = t_cpu.dtype

        # 3.
        # View as uint8 in Torch (zero-copy) -> Convert to NumPy uint8 (zero-copy) -> Dump to disk
        self._mmap_counter += 1
        filename = os.path.join(self._mmap_temp_dir, f"tensor_{self._mmap_counter}.bin")
        
        t_cpu.view(torch.uint8).numpy().tofile(filename)

        # 4. Map back read-only
        torch_to_np = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.uint16,   
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        
        np_dtype = torch_to_np.get(dtype, None)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        mapped_arr = np.memmap(filename, dtype=np_dtype, mode='r', shape=t_cpu.shape)

        if dtype == torch.bfloat16:
            mapped_tensor = torch.from_numpy(mapped_arr.view(np.uint16)).view(torch.bfloat16)
        elif dtype == torch.float16:
            mapped_tensor = torch.from_numpy(mapped_arr.view(np.uint16)).view(dtype)
        else:
            mapped_tensor = torch.from_numpy(mapped_arr)

        mapped_tensor._mmap_filename = filename

        # 5. Cleanup
        del t_cpu, mapped_arr
        gc.collect()

        return mapped_tensor

    def _read_gguf_tensors(self):
        self.gguf_tensors = {}
        for tensor in self.gguf_reader.tensors:
            self.gguf_tensors[tensor.name] = GGUFTensor(
                name=tensor.name,
                shape=tuple(tensor.shape.tolist()),
                data=tensor.data,
                tensor_type=tensor.tensor_type
            )

    def _read_gguf_metadata(self):
        for field in self.gguf_reader.fields.values():
            if field.name.endswith("embedding_length"):
                self.hidden_size = field.contents()
            elif field.name.endswith("feed_forward_length"):
                self.intermediate_size = field.contents()
            elif field.name.endswith("block_count"):
                self.num_layers = field.contents()

    def _load_config(self, config_file_path: str = "configs"):
        config_path = os.path.join(config_file_path, ModelArchConfigs[self.model_arch])
        print(f"[INFO] Loading Q4NX config from {config_path}")
        self.q4nx_config = json.load(open(config_path))
        self.row_block_size = self.q4nx_config["q4nx_config"]["row_block_size"]
        self.col_block_size = self.q4nx_config["q4nx_config"]["col_block_size"]
        self.parallel_size = self.q4nx_config["q4nx_config"]["parallel_size"]
        self.keep_block_in_2D = self.q4nx_config["q4nx_config"]["keep_block_in_2D"]
        
        if self.q4nx_config["default_tensor_type"] == "Q4_0":
            self.default_tensor_type = GGMLQuantizationType.Q4_0
        elif self.q4nx_config["default_tensor_type"] == "Q4_1":
            self.default_tensor_type = GGMLQuantizationType.Q4_1
        elif self.q4nx_config["default_tensor_type"] == "Q8_0":
            self.default_tensor_type = GGMLQuantizationType.Q8_0
        else:
            raise ValueError("Unsupported default_tensor_type in config")
        
        if "vision_config" in self.q4nx_config:
            self.vision_MM_K = self.q4nx_config["vision_config"]["vision_MM_K"]
            self.vision_MM_N = self.q4nx_config["vision_config"]["vision_MM_N"]
        else:
            self.vision_MM_K = None
            self.vision_MM_N = None
        self._create_name_maps()

    def get_ggml_type(self, q4nx_name: str) -> GGMLQuantizationType:
        if q4nx_name == "Q4_0":
            return GGMLQuantizationType.Q4_0
        elif q4nx_name == "Q4_1":
            return GGMLQuantizationType.Q4_1
        elif q4nx_name == "Q8_0":
            return GGMLQuantizationType.Q8_0
        else:
            raise ValueError(f"Unsupported q4nx_name: {q4nx_name}")

    def _create_name_maps(self):
        print("[INFO] Creating name maps...")
        self.forward_name_map = {}
        self.backward_name_map = {}
        self.tensor_q4nx_type_map = {}
        for param_name, param_info in self.q4nx_config["name_map"].items():
            if ("{bid}" in param_info["gguf_name"]):
                for bid in range(self.num_layers):
                    self.forward_name_map[param_info["gguf_name"].format(bid=bid)] = param_info["q4nx_name"].format(bid=bid)
                    self.backward_name_map[param_info["q4nx_name"].format(bid=bid)] = param_info["gguf_name"].format(bid=bid)
                    if "default_tensor_type" in param_info:
                        self.tensor_q4nx_type_map[param_info["gguf_name"].format(bid=bid)] = self.get_ggml_type(param_info["default_tensor_type"])
                    else:
                        self.tensor_q4nx_type_map[param_info["gguf_name"].format(bid=bid)] = self.default_tensor_type
                    if bid == 0:
                        print(f"\tConverted {param_info['gguf_name'].format(bid=bid)} to {param_info['q4nx_name'].format(bid=bid)}")
            else:
                self.forward_name_map[param_info["gguf_name"]] = param_info["q4nx_name"]
                self.backward_name_map[param_info["q4nx_name"]] = param_info["gguf_name"]
                if "default_tensor_type" in param_info:
                    self.tensor_q4nx_type_map[param_info["gguf_name"]] = self.get_ggml_type(param_info["default_tensor_type"])
                else:
                    self.tensor_q4nx_type_map[param_info["gguf_name"]] = self.default_tensor_type

                print(f"\tConverted {param_info['gguf_name']} to {param_info['q4nx_name']}")

        self.forward_name_map = dict(sorted(self.forward_name_map.items(), key=lambda item: item[0]))
        self.backward_name_map = dict(sorted(self.backward_name_map.items(), key=lambda item: item[0]))
        self.tensor_q4nx_type_map = dict(sorted(self.tensor_q4nx_type_map.items(), key=lambda item: item[0]))

    def _has_lm_head(self) -> bool:
        for key in self.gguf_tensors.keys():
            if "lm_head.weight" in key or key == "output.weight":
                return True  
        return False

    def _export_q4nx_tensors(self, q4nx_path: str):
        print(f"[INFO] Saving streaming Q4NX tensors to {q4nx_path}/model.q4nx...")
        create_dir_if_not_exists(q4nx_path)
        
        output_file = os.path.join(q4nx_path, "model.q4nx")
        
        # Detect tied weights sharing the same storage pointer.
        # Safetensors requires every key to have a unique memory allocation.
        tensors_to_save = {}
        seen_storages = set()

        for name, tensor in self.q4nx_tensors.items():
            # Get the raw memory pointer for the underlying storage
            storage_id = tensor.untyped_storage().data_ptr()
            
            if storage_id in seen_storages:
                # This is a tied weight (like lm_head). We clone it to 
                # give it its own memory space so Safetensors won't complain.
                print(f"[INFO] Cloning tied weight for {name} to satisfy Safetensors requirements...")
                tensors_to_save[name] = tensor.clone()
            else:
                tensors_to_save[name] = tensor
                seen_storages.add(storage_id)

        # Ensure garbage collection runs before the disk-heavy save_file
        gc.collect()
        
        try:
            save_file(tensors_to_save, output_file)
        finally:
            # Clear everything to free up the memory/MMAP handles
            self.q4nx_tensors.clear()
            tensors_to_save.clear()
            gc.collect()

        if hasattr(self, "_mmap_temp_dir") and os.path.exists(self._mmap_temp_dir):
            print("[INFO] Cleaning up disk-backed memory cache...")
            shutil.rmtree(self._mmap_temp_dir, ignore_errors=True)

    def _pack_MXFP4_q4nx(self, scales: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        Pack MXFP4 blocks (Q4-like layout) while minimizing RAM spikes.
        
        Key insight: Only pad the *remainder* block at the end — never whole tensors.
        Streaming offload each chunk to avoid accumulation.
        """
        MXFP4_BLOCK_SIZE = 32
        MXFP4_BLOCK_SIZE_data_in_byte = 16

        # Ensure correct shapes before we start
        assert len(scales.shape) == 3, f"scales must be [batch, rows, cols], got {scales.shape}"
        assert len(data.shape) == 4, f"data must be [batch, rows, col_blocks, byte_block], got {data.shape}"

        batch_size, orig_rows, cols = scales.shape
        # data.shape[1] should match rows, data.shape[2]*MXFP4_BLOCK_SIZE should match cols
        assert data.shape[0] == batch_size and data.shape[1] == orig_rows

        # Step 1: Determine how many full blocks + remainder
        row_block_size = self.row_block_size
        col_block_size = self.col_block_size
        
        num_full_rows = orig_rows // row_block_size
        remainder_rows = orig_rows % row_block_size
        
        num_col_blocks = (cols + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
        effective_cols = num_col_blocks * MXFP4_BLOCK_SIZE

        # Prepare padded tensors only for the *remainder* if needed
        scales_padded = F.pad(scales, (0, 0, 0, remainder_rows), "constant", 0) if remainder_rows > 0 else scales
        data_padded = data
        
        if remainder_rows > 0:
            pad_cols = (effective_cols - cols) // MXFP4_BLOCK_SIZE
            # CRITICAL FIX: Use correct 6-element padding order for 4D tensor
            # [B, R, C_blks, B_blk] → pad in reverse dim order:
            #   byte_block (dim3): (0, 0)
            #   col_blocks (dim2): (0, pad_cols) ← only right-pad columns!
            #   rows       (dim1): (0, remainder_rows)
            if pad_cols > 0 or remainder_rows > 0:
                data_padded = F.pad(
                    data,
                    (
                        0, 0,           # byte_block axis: no padding needed
                        0, pad_cols,    # col_blocks axis: right-pad only
                        0, remainder_rows  # rows axis: bottom-pad only (not top!)
                    ),
                    "constant", 0
                )

        # Step 2: Process full blocks + remainder with streaming offload (no accumulation!)
        merged_chunks = []
        
        for i in range(num_full_rows):
            row_start = i * row_block_size
            row_end = row_start + row_block_size
            
            s_chunk = scales_padded[:, row_start:row_end, :]
            d_chunk = data_padded[:, row_start:row_end, :, :]

            packed_chunk = self._pack_MXFP4_q4nx_core(s_chunk, d_chunk, row_block_size)
            
            # CRITICAL: Offload chunk immediately — no accumulation!
            offloaded_chunk = self._offload_to_disk(packed_chunk)
            merged_chunks.append(offloaded_chunk)

        # Step 3: Handle remainder block (if any) — padded already above
        if remainder_rows > 0:
            s_rem = scales_padded[:, num_full_rows * row_block_size:, :]
            d_rem = data_padded[:, num_full_rows * row_block_size:, :, :]

            packed_rem = self._pack_MXFP4_q4nx_core(s_rem, d_rem, row_block_size)
            
            # CRITICAL: Offload remainder chunk immediately!
            offloaded_rem = self._offload_to_disk(packed_rem)
            merged_chunks.append(offloaded_rem)

        # Step 4: Concatenate final result — minimal peak RAM (only two chunks live at once)
        if len(merged_chunks) == 1:
            return merged_chunks[0]

        # Streaming concat via offload (no torch.cat!)
        merged = merged_chunks[0]
        for chunk in merged_chunks[1:]:
            filename = getattr(merged, "_mmap_filename", None)
            if filename and os.path.exists(filename):
                old_shape = merged.shape
                new_shape = (*old_shape[:-1], old_shape[-1] + chunk.shape[-1])

                # Reopen memmap with extended shape
                mapped_arr = np.memmap(filename, dtype=np.int8, mode='r+', shape=new_shape)
                mapped_tensor = torch.from_numpy(mapped_arr.view(np.int8)).view(torch.int8)

                # FIX: Load chunk into RAM first — avoids read-only → write conflict!
                chunk_cpu = chunk.detach().cpu().contiguous()  # ensures it's in host memory
                mapped_tensor[..., old_shape[-1]:] = chunk_cpu

                merged = mapped_tensor
            else:
                merged = torch.cat([merged, chunk], dim=-1).contiguous()
                merged = self._offload_to_disk(merged)

        return merged

    def _pack_MXFP4_q4nx_core(self, scales: torch.Tensor, data: torch.Tensor, row_block_size: int) -> torch.Tensor:
        MXFP4_BLOCK_SIZE = 32
        MXFP4_BLOCK_SIZE_data_in_byte = 16

        batch_size, rows, cols = scales.shape
        assert rows == row_block_size
        assert data.shape[0] == batch_size and data.shape[1] == rows
        num_col_blocks = data.shape[2]

        col_div = num_col_blocks // (self.col_block_size // MXFP4_BLOCK_SIZE)
        
        # Rearrange scales to [B, row_div, col_div, r, c]
        scales_rearranged = rearrange(
            scales,
            "batch (row_div r) (col_div c) -> batch row_div col_div r c",
            row_div=1, r=row_block_size,
            col_div=col_div, c=self.col_block_size // MXFP4_BLOCK_SIZE
        ).contiguous()

        # Reshape data to [B, R, col_div, q4_col, byte_block]
        d_reshaped = data.reshape(batch_size, rows, col_div, self.col_block_size // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE_data_in_byte)

        # Reorder to [B, row_div, col_div, r, c, q4_col, byte_block]
        d_reordered = rearrange(
            d_reshaped,
            "batch (row_div r) (col_div c) q4_col byte -> batch row_div col_div r c q4_col byte",
            row_div=1, r=row_block_size,
            col_div=col_div, c=self.col_block_size // MXFP4_BLOCK_SIZE
        ).contiguous()

        # Extract nibbles in *original order*: even bytes = low nibble, odd bytes = high nibble
        low_nibbles = d_reordered[..., 0::2] & 0x0F         # bits [3:0]
        high_nibbles = (d_reordered[..., 1::2] >> 4) & 0x0F # bits [7:4]
        
        # CRITICAL: match existing Q4NX convention → low nibble goes to upper half
        packed_data = (high_nibbles << 4) | low_nibbles

        # Flatten and pack with scales
        packed_flat = packed_data.reshape(batch_size, -1).contiguous()
        scales_flat = scales_rearranged.reshape(batch_size, -1).contiguous()

        # Pad scales: same length as packed bytes → 3× padding per scale byte
        pad_amount = packed_flat.shape[-1]
        scales_padded = F.pad(scales_flat, (0, pad_amount), "constant", 0)

        merged = torch.cat([scales_padded, packed_flat], dim=-1).contiguous()
        return merged
        
    def force_pack_q8_to_q4nx_size(self, tensor_data: GGUFTensor) -> torch.Tensor:
        """
        Stream: Q8_in → float → quantize(Q8_0) → unpack → pack(q4nx).
        
        Key fix: No double dequant! We only dequant once (to float), then re-quant to Q8_0,
                   and finally unpack *properly* before packing.
        """
        print(f"[INFO] Streaming Q8→Q4NX for {tensor_data.name}...")

        if len(tensor_data.shape) >= 2:
            rows, cols = tensor_data.shape[1], tensor_data.shape[0]
        else:
            rows, cols = 1, tensor_data.shape[0]

        chunk_rows = 256

        packed_chunks = []

        for i in range(0, rows, chunk_rows):
            start_row = i
            end_row = min(i + chunk_rows, rows)

            try:
                block_size = GGML_QUANT_SIZES.get(tensor_data.tensor_type, 32)
            except (KeyError, AttributeError):
                block_size = 32

            num_blocks_per_row = (cols + block_size - 1) // block_size
            start_block = start_row * num_blocks_per_row
            end_block   = end_row   * num_blocks_per_row

            raw_chunk = tensor_data.data[start_block:end_block]
            if raw_chunk.size == 0:
                continue

            # Step 2: Dequant → Float32 → Re-quant to Q8_0 (small peak)
            w_float = dequantize(raw_chunk, tensor_data.tensor_type)   # float32
            q8_np   = quantize(w_float, GGMLQuantizationType.Q8_0)     # int8, Q8_0 layout

            del w_float, raw_chunk
            gc.collect()

            # FIXED: unpack_q8_0 returns *tensors* → no from_numpy needed!
            d_tensor, m_tensor, qw_tensor = GGUFTensor.unpack_q8_0(q8_np, cols)

            del q8_np
            gc.collect()

            # Cast to correct dtypes (no numpy dance!)
            packed_chunk = self._pack_q4nx(
                d=d_tensor.to(torch.bfloat16),
                m=m_tensor.to(torch.bfloat16),
                qw=qw_tensor.to(torch.int8)
            )

            # Offload immediately
            packed_chunk = self._offload_to_disk(packed_chunk)
            packed_chunks.append(packed_chunk)
            gc.collect()

        return self._merge_offloaded_chunks(packed_chunks)

    def _merge_offloaded_chunks(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate offloaded memmap chunks vertically (along rows).
        """
        if not chunks:
            raise ValueError("Cannot merge empty chunk list")

        if len(chunks) == 1:
            return chunks[0]

        merged = chunks[0]
        for i, chunk in enumerate(chunks[1:], start=1):
            filename = getattr(merged, "_mmap_filename", None)
            
            if filename and os.path.exists(filename):
                old_shape = merged.shape
                # FIX: Grow Dimension 0 (rows), keep columns the same
                new_rows = old_shape[0] + chunk.shape[0]
                new_shape = (new_rows, *old_shape[1:])

                # Physically grow the file on disk
                with open(filename, "ab") as f:
                    current_size = os.path.getsize(filename)
                    # For int8/uint8, total bytes = product of all dimensions
                    target_size = np.prod(new_shape)
                    if target_size > current_size:
                        f.write(b"\x00" * (target_size - current_size))
                        f.flush()
                        os.fsync(f.fileno())

                # Re-map with the new vertical shape
                mapped_arr = np.memmap(filename, dtype=np.int8, mode='r+', shape=new_shape)
                mapped_tensor = torch.from_numpy(mapped_arr.view(np.int8)).view(torch.int8)

                # FIX: Insert the chunk at the bottom (new row indices)
                chunk_cpu = chunk.detach().cpu().contiguous()
                mapped_tensor[old_shape[0]:new_rows, ...] = chunk_cpu

                # Keep the mmap reference and filename alive
                merged = mapped_tensor
                merged._mmap_filename = filename
                
                # Manual cleanup of the chunk we just copied
                del chunk_cpu
            else:
                # Fallback: cat vertically
                merged = torch.cat([merged, chunk], dim=0).contiguous()
                merged = self._offload_to_disk(merged)

        return merged

    def _unpack_and_pack_streaming(self, w_chunk_np: np.ndarray, tensor_type: GGMLQuantizationType) -> torch.Tensor:
        """
        Unpack Q8 chunk in small pieces — streaming!
        
        FIX: After dequantizing, re-quantize to Q8_0 (or target type) before packing.
        """
        try:
            block_size = GGML_QUANT_SIZES.get(tensor_type, 32)
        except (KeyError):
            block_size = 32

        chunk_rows, cols = w_chunk_np.shape[0], w_chunk_np.shape[1]
        num_blocks_per_row = (cols + block_size - 1) // block_size
        unpack_chunk_rows = min(chunk_rows, 64)

        packed_parts = []

        for start_r in range(0, chunk_rows, unpack_chunk_rows):
            end_r = min(start_r + unpack_chunk_rows, chunk_rows)
            
            w_piece_np = w_chunk_np[start_r:end_r, :]
            
            # Step 1: Dequantize → get float weights
            dequantized_piece = dequantize(w_piece_np, tensor_type)   # dtype: float32
            
            # Step 2: Quantize back to Q8_0 (or target type) to restore proper layout
            if tensor_type == GGMLQuantizationType.Q8_0:
                # Already Q8 — no need to re-quantize, just ensure correct shape/layout
                q8_piece = dequantized_piece.astype(np.int8)
            else:
                # Re-quantize to target type (e.g., Q4_0 → we want raw Q8_0 layout first for compatibility)
                # But note: force_pack_q8_to_q4nx_size implies "Q8_in → q4nx_out", so we *must* quantize to Q8_0 here
                q8_piece = quantize(dequantized_piece, GGMLQuantizationType.Q8_0)  # This is key
                
            # Step 3: Pack using existing logic on *properly-quantized* data
            packed_part = self._pack_q4nx_streaming(q8_piece, tensor_type)
            packed_parts.append(packed_part)

            del w_piece_np, dequantized_piece, q8_piece
            gc.collect()

        return torch.cat(packed_parts, dim=0).contiguous()


    def _pack_q4nx_streaming(self, q8_data: np.ndarray, tensor_type: GGMLQuantizationType) -> torch.Tensor:
        """
        Pack Q8 data — streaming!
        """
        try:
            block_size = GGML_QUANT_SIZES.get(tensor_type, 32)
        except (KeyError):
            block_size = 32

        rows, cols = q8_data.shape[0], q8_data.shape[1]
        
        # Handle padding requirements
        if cols % block_size != 0:
            pad_cols = block_size - (cols % block_size)
            q8_data = np.pad(q8_data, ((0, 0), (0, pad_cols)), mode='constant')

        rows_padded, cols_padded = q8_data.shape[0], q8_data.shape[1]
        
        # Handle alignment constraints
        if self.keep_block_in_2D:
            # Ensure row/col alignment for 2D packing
            row_align = self.row_block_size
            col_align = block_size
            
            if rows_padded % row_align != 0:
                pad_rows = row_align - (rows_padded % row_align)
                q8_data = np.pad(q8_data, ((0, pad_rows), (0, 0)), mode='constant')

        # Pack using existing logic
        packed = self._pack_q4nx_impl(q8_data, tensor_type)
        
        del q8_data
        gc.collect()
        
        return packed


    def _pack_q4nx_impl(self, q8_data: np.ndarray, tensor_type: GGMLQuantizationType):
        """
        Core packing implementation — streaming!
        """
        rows, cols = q8_data.shape[0], q8_data.shape[1]
        
        # Handle alignment constraints
        if self.keep_block_in_2D:
            row_align = self.row_block_size
            col_align = block_size
            
            if rows % row_align != 0:
                pad_rows = row_align - (rows % row_align)
                q8_data = np.pad(q8_data, ((0, pad_rows), (0, 0)), mode='constant')

        # Extract scales/m/qw — streaming!
        scales, m, qw = self._extract_q4nx_components(q8_data, tensor_type)

        packed = torch.cat([scales, m, qw], dim=-1).contiguous()
        
        del q8_data
        gc.collect()
        
        return packed


    def _extract_q4nx_components(self, q8_data: np.ndarray, tensor_type: GGMLQuantizationType):
        """
        Extract scales/m/qw — streaming!
        """
        rows, cols = q8_data.shape[0], q8_data.shape[1]
        
        # Determine block size for this layer
        try:
            block_size = GGML_QUANT_SIZES.get(tensor_type, 32)
        except (KeyError):
            block_size = 32

        num_blocks_per_row = cols // block_size
        
        # Extract scales/m/qw — streaming!
        scales = q8_data[:, :num_blocks_per_row].copy()
        m = q8_data[:, num_blocks_per_row:2*num_blocks_per_row].copy()
        qw = q8_data[:, 2*num_blocks_per_row:].copy()

        # Convert to PyTorch
        scales_t = torch.from_numpy(scales).to(torch.bfloat16)
        m_t = torch.from_numpy(m).to(torch.bfloat16)
        qw_t = torch.from_numpy(qw).to(torch.int8)

        return scales_t, m_t, qw_t
            
    def _pack(self, d: torch.Tensor, m: torch.Tensor = None, qw: torch.Tensor = None, tensor_type: GGMLQuantizationType = None) -> torch.Tensor:
        if tensor_type == GGMLQuantizationType.Q8_0:
            result = self._pack_q4nx_8b(d, m, qw)
        else:
            result = self._pack_q4nx(d, m, qw)
        
        # Master offload: Intercepts all conversions right before they enter self.q4nx_tensors
        return self._offload_to_disk(result)
        
    def _pack_q4nx_8b(self, d: torch.Tensor, m: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
        """
        Pack Q8 weights into MXFP4 format — streaming!
        
        Key insight: Offload immediately after core packing, before padding to avoid RAM spikes.
        """
        col_block_size_old = self.col_block_size
        keep_block_in_2D_old = self.keep_block_in_2D
        
        # Step 1: Temporarily adjust config
        if self.col_block_size == 256:
            self.col_block_size = 128            
        else:
            raise ValueError("Undefine case for now")
        
        self.keep_block_in_2D = True
            
        cur_q4nx_block_byte_size = int((self.row_block_size * col_block_size_old) * (5/8))

        # Step 2: Core packing — offload immediately after
        q8nx_pack_result = self._pack_q8nx(data=qw, scales=d, m=m)

        # CRITICAL: Offload to disk BEFORE padding!
        packed_offloaded = self._offload_to_disk(q8nx_pack_result)
        
        # Step 3: Compute and apply minimal padding in-place
        current_size = packed_offloaded.shape[-1]
        padding_needed = cur_q4nx_block_byte_size - current_size

        if padding_needed > 0:
            # In-place expansion via memmap extension (no new tensor!)
            filename = getattr(packed_offloaded, "_mmap_filename", None)
            if filename and os.path.exists(filename):
                # Extend memmap to include padding
                old_shape = packed_offloaded.shape
                new_shape = (*old_shape[:-1], current_size + padding_needed)
                
                # Reopen memmap with extended shape
                mapped_arr = np.memmap(filename, dtype=np.int8, mode='r+', shape=new_shape)
                mapped_tensor = torch.from_numpy(mapped_arr.view(np.int8)).view(torch.int8)
                
                # Zero-pad new region in-place
                mapped_tensor[..., current_size:] = 0
                
                packed_offloaded = mapped_tensor
                
                # Reattach metadata
                packed_offloaded._mmap_filename = filename
            else:
                # Fallback: pad and offload (still better than RAM spike — offloads immediately)
                padded = F.pad(packed_offloaded, (0, padding_needed), "constant", 0)
                packed_offloaded = self._offload_to_disk(padded)
        else:
            # Already correct size — no action needed
            pass

        # Step 4: Restore config
        self.keep_block_in_2D = keep_block_in_2D_old
        self.col_block_size = col_block_size_old
        
        return packed_offloaded
        
    def _pack_q8nx(self, data: torch.Tensor, scales: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        Q8_group_size =  32
        
        if scales.shape[-1] == 1:
            scales = scales.reshape(*scales.shape[:-2], -1).contiguous()
            data = data.reshape(*data.shape[:-2], -1).contiguous()
            m = m.reshape(*m.shape[:-2], -1).contiguous()
        else:
            scales = scales.contiguous()
            data = data.contiguous()
            m = m.contiguous()  
            
        rows, cols = data.shape[0], data.shape[1]
        
        if cols % self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            
            scale_pad_amount = (cols_padded - cols) // Q8_group_size
            data_pad_amount = cols_padded - cols
            
            scales = F.pad(scales, (0, scale_pad_amount), "constant", 0)
            m    = F.pad(m,    (0, scale_pad_amount), "constant", 0)
            data = F.pad(data, (0, data_pad_amount), "constant", 0)

        if self.keep_block_in_2D:
            scales = rearrange(
                scales,
                "(row_div_r r) (col_div_c c) -> row_div_r col_div_c (c r)",
                r=self.row_block_size,
                c=self.col_block_size // Q8_group_size
            ).contiguous()
            
            m = rearrange(
                m,
                "(row_div_r r) (col_div_c c) -> row_div_r col_div_c (c r)",
                r=self.row_block_size,
                c=self.col_block_size // Q8_group_size                
            ).contiguous()
            
            assert self.row_block_size % self.parallel_size == 0
            
            data = rearrange(
                data,
                "(row_div_r r) (col_div_c c) -> row_div_r col_div_c r c",
                r=self.row_block_size,
                c=self.col_block_size  
            ).contiguous()
            
            data = rearrange(
                data,
                "row_div_r col_div_c (r_div_parallel parallel) c -> row_div_r col_div_c (r_div_parallel c parallel)",
                parallel=self.parallel_size,
            )
        else:
            raise ValueError("Only support keep_block_in_2D for now")
        
        # Use .view(torch.uint8) — not int8/bfloat16 — to ensure byte-accurate concatenation
        scales_bytes = scales.view(torch.uint8)
        m_bytes    = m.view(torch.uint8)
        data_bytes = data.view(torch.uint8)

        merged = torch.cat([scales_bytes, m_bytes, data_bytes], dim=-1).contiguous()
        return merged      
    
    def _pack_q4nx(self, d: torch.Tensor, m: torch.Tensor = None, qw: torch.Tensor = None) -> torch.Tensor:
        Q4_group_size =  32
        NUM_int4_in_byte = 2
        d = d.contiguous()
        m = m.contiguous() if m is not None else None
        qw = qw.contiguous() if qw is not None else None
        
        if m is None: 
            return d.to(torch.bfloat16)

        rows, cols = qw.shape

        if cols%self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            
            d_m_pad_amount = (cols_padded -cols) // Q4_group_size
            qw_pad_amount = cols_padded-cols
            
            d = F.pad(d, (0, d_m_pad_amount), "constant", 0)
            m = F.pad(m, (0, d_m_pad_amount), "constant", 0)
            qw = F.pad(qw, (0, qw_pad_amount), "constant", 0)
            
        if not self.keep_block_in_2D:
            d = rearrange(d, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()

            qw = rearrange(qw, '(p r) (q c) -> (p q) r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'n (g r) c -> n g r c', r = self.parallel_size)
            qw = rearrange(qw, 'n g (r b) c -> n g c r b', b = NUM_int4_in_byte).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'n g c r -> n (g c r)').contiguous()
        else:
            d = rearrange(d, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()

            qw = rearrange(qw, '(p r) (q c) -> p q r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'p q (g r) c -> p q g r c', r = self.parallel_size)
            qw = rearrange(qw, 'p q g (r b) c -> p q g c r b', b = NUM_int4_in_byte).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'p q g c r -> p q (g c r)').contiguous()
        
        # Use .view(torch.uint8) for raw byte-level concatenation — no shape doubling!
        d_bytes = d.to(torch.bfloat16).view(torch.uint8)
        m_bytes = m.to(torch.bfloat16).view(torch.uint8)
        qw_bytes = qw.view(torch.uint8)

        merged = torch.cat([d_bytes, m_bytes, qw_bytes], dim=-1).contiguous()
        return merged

    def vision_mm_weight_rearrange(self, weight: torch.Tensor) -> torch.Tensor:
        assert len(weight.shape) ==2
        assert self.vision_MM_K is not None and self.vision_MM_N is not None, "vision_MM_K and vision_MM_N must be set for vision model weight rearrangement"
        MM_N_K_padding = max(self.vision_MM_N, self.vision_MM_K)
        origin_N, origin_K = weight.shape
        
        pad_N = (MM_N_K_padding - origin_N % MM_N_K_padding) % MM_N_K_padding
        pad_K = (MM_N_K_padding - origin_K % MM_N_K_padding) % MM_N_K_padding
        
        if pad_N >0 or pad_K >0:
            weight = F.pad(weight, (0, pad_K, 0, pad_N), "constant", 0)
        weight = rearrange( weight,  
                       "(N mm_n) (K mm_k) -> N K (mm_n mm_k)",
                       mm_k = self.vision_MM_K, mm_n = self.vision_MM_N).contiguous()

        return weight
        
    def _extract_tokenizer_json(self, output_folder: str) -> dict:
        print("[INFO] Extracting tokenizer JSON...")
        tokenizer_json_path = os.path.join(output_folder, "tokenizer.json")
        
        tokenizer_model = None
        tokens = []
        scores = []
        token_types = []
        merges = []
        bos_token_id = None
        eos_token_id = None
        unk_token_id = None
        pad_token_id = None
        
        for field in self.gguf_reader.fields.values():
            if field.name == "tokenizer.ggml.model":
                tokenizer_model = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
            elif field.name == "tokenizer.ggml.tokens":
                tokens = [str(field.parts[i], encoding='utf-8') for i in field.data]
            elif field.name == "tokenizer.ggml.scores":
                scores = field.data.tolist() if hasattr(field.data, 'tolist') else list(field.data)
            elif field.name == "tokenizer.ggml.token_type":
                token_types = field.data.tolist() if hasattr(field.data, 'tolist') else list(field.data)
            elif field.name == "tokenizer.ggml.merges":
                merges = [str(field.parts[i], encoding='utf-8') for i in field.data]
            elif field.name == "tokenizer.ggml.bos_token_id":
                bos_token_id = int(field.contents())
                print(f"[INFO] BOS token ID: {bos_token_id}")
            elif field.name == "tokenizer.ggml.eos_token_id":
                eos_token_id = int(field.contents())
                print(f"[INFO] EOS token ID: {eos_token_id}")
            elif field.name == "tokenizer.ggml.unknown_token_id":
                unk_token_id = int(field.contents())
                print(f"[INFO] Unknown token ID: {unk_token_id}")
            elif field.name == "tokenizer.ggml.padding_token_id":
                pad_token_id = int(field.contents())
                print(f"[INFO] Padding token ID: {pad_token_id}")
        
        vocab = {}
        for idx, token in enumerate(tokens):
            vocab[token] = idx
        
        if tokenizer_model == "llama" or tokenizer_model == "gpt2":
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": self._build_added_tokens(tokens, bos_token_id, eos_token_id, unk_token_id, pad_token_id),
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True
                },
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": tokens[unk_token_id] if unk_token_id is not None and unk_token_id < len(tokens) else None,
                    "continuing_subword_prefix": "",
                    "end_of_word_suffix": "",
                    "fuse_unk": False,
                    "byte_fallback": False,
                    "vocab": vocab,
                    "merges": merges
                }
            }
        else:
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": self._build_added_tokens(tokens, bos_token_id, eos_token_id, unk_token_id, pad_token_id),
                "normalizer": {
                    "type": "Sequence",
                    "normalizers": []
                },
                "pre_tokenizer": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": True,
                    "prepend_scheme": "first"
                },
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {
                            "SpecialToken": {
                                "id": tokens[bos_token_id] if bos_token_id is not None and bos_token_id < len(tokens) else "<s>",
                                "type_id": 0
                            }
                        },
                        {"Sequence": {"id": "A", "type_id": 0}}
                    ],
                    "pair": [
                        {
                            "SpecialToken": {
                                "id": tokens[bos_token_id] if bos_token_id is not None and bos_token_id < len(tokens) else "<s>",
                                "type_id": 0
                            }
                        },
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {
                            "SpecialToken": {
                                "id": tokens[eos_token_id] if eos_token_id is not None and eos_token_id < len(tokens) else "</s>",
                                "type_id": 0
                            }
                        },
                        {"Sequence": {"id": "B", "type_id": 1}}
                    ],
                    "special_tokens": {}
                },
                "decoder": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": True,
                    "prepend_scheme": "first"
                },
                "model": {
                    "type": "Unigram",
                    "unk_id": unk_token_id,
                    "vocab": [[token, score] for token, score in zip(tokens, scores)] if scores else [[token, 0.0] for token in tokens]
                }
            }
        
        create_dir_if_not_exists(output_folder)
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Tokenizer saved to {tokenizer_json_path}")
        
        # Avoid keeping 150k+ strings in memory — nothing needs the returned dict here
        return None
    
    def _build_added_tokens(self, tokens: list, bos_id: int, eos_id: int, unk_id: int, pad_id: int) -> list:
        added_tokens = []
        special_token_ids = {
            bos_id: "bos_token",
            eos_id: "eos_token",  
            unk_id: "unk_token",
            pad_id: "pad_token"
        }
        
        for token_id, token_type in special_token_ids.items():
            if token_id is not None and token_id < len(tokens):
                added_tokens.append({
                    "id": token_id,
                    "content": tokens[token_id],
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                })
        
        return added_tokens

def get_model_arch_from_gguf(reader: GGUFReader, override_model_arch:str="") -> ModelArch:
    if override_model_arch != "":
        for arch_enum, arch_names in ModelArchNames.items():
            for arch_name in arch_names:            
                if override_model_arch.lower().startswith(arch_name.lower()):
                    return arch_enum
        print("Warning: Did not find matching override model arch, attempting to load base on gguf information")

    arch_str:str|None = None
    basename_str:str|None = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            arch_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
        elif field.name == 'general.basename':
            basename_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
    
    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if arch_str.lower() == arch_name.lower():
                return arch_enum
    
    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if basename_str and arch_name and basename_str.lower().startswith(arch_name.lower()):
                return arch_enum

    raise ValueError(f"Unsupported model architecture: {arch_str}")

def get_registered_models() -> Dict[ModelArch, Type['__Q4NX_Converter']]:
    return _MODEL_REGISTRY.copy()

def create_converter(gguf_path: str, override_model_arch:str) -> __Q4NX_Converter:
    reader = GGUFReader(gguf_path)
    model_arch = get_model_arch_from_gguf(reader,override_model_arch )
    
    if model_arch not in _MODEL_REGISTRY:
        available = [ModelArchNames.get(arch, str(arch)) for arch in _MODEL_REGISTRY.keys()]
        raise ValueError(
            f"No converter available for architecture: {ModelArchNames.get(model_arch, 'unknown')}. "
            f"Available converters: {available if available else 'None (no models imported?)'}"
        )
    
    converter_class = _MODEL_REGISTRY[model_arch]
    converter_instance = converter_class(reader)

    return converter_instance
