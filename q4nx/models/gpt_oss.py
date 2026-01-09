from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class GPTOSS(__Q4NX_Converter, model_arch=ModelArch.GPT_OSS):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def merge_expert_weights(self):
        """
        Merge all expert up, gate, down weights to run on NPU.
        
        :param self: Description

        We have to merge all expert up, gate, down weights to run on NPU,
        We also need to handle MXFP4 format
        """
        for layer_id in range(self.num_layers):
            print(f"[INFO] Merging expert weights for layer {layer_id}, new name model.layers.{layer_id}.ffn_gate_up_down_exps.weight")
            # Just delete it for now
            del self.gguf_tensors[f"blk.{layer_id}.ffn_up_exps.weight"]
            del self.gguf_tensors[f"blk.{layer_id}.ffn_gate_exps.weight"]
            del self.gguf_tensors[f"blk.{layer_id}.ffn_down_exps.weight"]


    def convert(self, q4nx_path: str):
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack()
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        self.merge_expert_weights()

        for key, gguf_tensor in self.gguf_tensors.items():
            print(f"[INFO] Converting tensor {gguf_tensor.name} to {self.forward_name_map[gguf_tensor.name]}")
            if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                continue

            # unpacked = gguf_tensor.unpack()
        

            # self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        # Commet for now
        # self._export_q4nx_tensors(q4nx_path)
        # self._extract_tokenizer_json(q4nx_path)
