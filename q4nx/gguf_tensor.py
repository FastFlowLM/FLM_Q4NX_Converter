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
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8)

        d = torch.from_numpy(d)
        # m = torch.zeros_like(d)
        m = d * (-8)
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

    def unpack(self) -> np.ndarray:
        if self.tensor_type == GGMLQuantizationType.F32:
            return [torch.Tensor(np.array(self.data.view(np.float32)))]
        elif self.tensor_type == GGMLQuantizationType.F16:
            return [torch.Tensor(np.array(self.data.view(np.float16).astype(np.float32)))]
        elif self.tensor_type == GGMLQuantizationType.Q4_0:
            return self.unpack_q4_0(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q4_1:
            return self.unpack_q4_1(self.data, self.shape[0])
        else:
            """
                If the tensor type is not supported, try to dequantize it and then quantize it back to Q4_1
                This is a workaround for the fact that the GGUF format does not support all tensor types
                and we need to convert it to a supported type before converting to Q4NX
            """
            try:
                w = dequantize(self.data, self.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                
                w = w.to(torch.float32).numpy()
                data_q4_1 = quantize(w, GGMLQuantizationType.Q4_1).copy()
                d, m, qw = self.unpack_q4_1(data_q4_1, self.shape[0])
                return d, m, qw
            except Exception as e:
                print(f"Error unpacking {self.tensor_type.name}: {e}")
                return None, None, None