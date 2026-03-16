from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class Qwen3(__Q4NX_Converter, model_arch=ModelArch.QWEN3):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            # unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            # self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

            # # TODO: FIXME: DEBUG force convert Q80
            # w = dequantize(self.gguf_tensors["token_embd.weight"].data, self.gguf_tensors["token_embd.weight"].tensor_type)
            # w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
            # w = w.to(dtype=torch.float32).numpy()
            
            # data_q80 = quantize(w, GGMLQuantizationType.Q8_0).copy()
            
            
            
            # scales, data = GGUFTensor.unpack_q8_0( data_q80)
            
            # m_tmp = scales.clone()
            
            # # override 
            # row_block_size_old = self.row_block_size
            # col_block_size_old = self.col_block_size
            # keep_block_in_2D_old = self.keep_block_in_2D
            # self.row_block_size = 32
            # self.col_block_size= 128
            # self.keep_block_in_2D= True
            # self.q4nx_tensors["lm_head.weight"] = self._pack_q8nx(
            #     data, scales, m_tmp,
            # )
            
            # # now, we want to padd the last dimension to be size of 5120(for q4nx)
            # # Pad the last dimension from 4608 to 5120
            # import torch.nn.functional as F
            # lm_head = self.q4nx_tensors["lm_head.weight"]
            # padding_size = 5120 - lm_head.shape[-1]
            # self.q4nx_tensors["lm_head.weight"] = F.pad(lm_head, (0, padding_size))
            
            # self.keep_block_in_2D = keep_block_in_2D_old
            # self.row_block_size = row_block_size_old
            # self.col_block_size = col_block_size_old
            
            
            lm_head_q80= self.force_pack_q8_to_q4nx_size( self.gguf_tensors["token_embd.weight"] )
            self.q4nx_tensors["lm_head.weight"]  = lm_head_q80
            
            
            
            

        for key, gguf_tensor in self.gguf_tensors.items():
            if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)
        

            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
