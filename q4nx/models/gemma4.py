import gguf

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange, unpack

class Gemma4(__Q4NX_Converter, model_arch=ModelArch.GEMMA4):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

        
        # read "embedding_length_per_layer_input" from gguf metadata
        field = self.gguf_reader.fields.get("gemma4.embedding_length_per_layer_input", None)
        self.embedding_length_per_layer_input = field.contents() if field is not None else None 
    def initialize(self):
        super().initialize()

    
    
    
    def reshape_matrix_to_block_matrix_for_mvm(self, weight: torch.Tensor, row_block_size: int=32) -> torch.Tensor:
        """
            Assume the weights is a 2D matrix of W x H
            The function will reshape weights into (blocks, H, row_block_size)
            Where blocks = W // row_block_size
            
            In addition, each block(row_block_size x H) is in column-major order
        """
        assert weight.ndim == 2, "Input weight must be a 2D matrix"
        
        W, H = weight.shape
        
        
        assert W % row_block_size == 0, f"Weight rows {W} must be divisible by row_block_size {row_block_size}"
        blocks = W // row_block_size
        
        
        # Step 1: Reshape to (blocks, row_block_size, H)
        weight = weight.contiguous()

        weight = rearrange(weight,
                           '(blocks row_block_size) H -> blocks row_block_size H',
                           blocks=blocks, row_block_size=row_block_size
                           )
        # Step 2: Transpose each block to be in column-major order
        weight = rearrange(weight,
                           'blocks row_block_size H -> blocks H row_block_size',
                           )
        return weight
        

    
    
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if weights_type == "language":
            
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
                self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)      

            for key, gguf_tensor in self.gguf_tensors.items():
                if "token_embd.weight"  == gguf_tensor.name: # this should be bf16
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = w * float(self.hidden_size) **0.5
                    w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue
                elif "per_layer_token_embd.weight"  ==  gguf_tensor.name:
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = w*float(self.embedding_length_per_layer_input)**0.5
                    w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue
                elif "per_layer_model_proj.weight" in gguf_tensor.name:
                    unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    w=unpacked[0]
                    
                    # since output projection == self.embedding_length_per_layer_input* self.num_layers, we want to actually split the matrxi
                    
                    assert w.shape[0] == self.embedding_length_per_layer_input * self.num_layers, f"Expected output projection weight shape[0] to be {self.embedding_length_per_layer_input * self.num_layers}, but got {w.shape[0]}"
                    
                    w_per_layer = w.reshape(self.num_layers, self.embedding_length_per_layer_input, w.shape[1])
                    
                    for layer_idx in range(self.num_layers):
                        layer_w = w_per_layer[layer_idx]
                        layer_w = self.reshape_matrix_to_block_matrix_for_mvm(layer_w)
                        layer_w = layer_w.contiguous().to(torch.bfloat16)
                        self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_layer{layer_idx}"] = layer_w
                    
                    
                    # w = self.reshape_matrix_to_block_matrix_for_mvm(unpacked[0])
                    # w = w.contiguous().to(torch.bfloat16)
                    # self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w

                    continue
                                        
                elif "inp_gate.weight" in gguf_tensor.name or "proj.weight" in gguf_tensor.name:
                    
                    unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    w = self.reshape_matrix_to_block_matrix_for_mvm(unpacked[0])
                    w = w.contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    
                    continue
                
                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])

                torch.set_printoptions(threshold=16, edgeitems=5, linewidth=200)

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

            self._extract_tokenizer_json(q4nx_path)
        elif weights_type == "vision":
            
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"

                
                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    # convert to bfloat16
                    weights = weights.to(torch.bfloat16)
                if gguf_tensor.name not in self.forward_name_map:
                    # check is not start with "v"
                    if not gguf_tensor.name.startswith("v"):
                        continue
                    else:
                        raise ValueError(f"Tensor name {gguf_tensor.name} not found in forward_name_map for vision model")
                new_name = self.forward_name_map[gguf_tensor.name]
                
                if new_name == "model.vision.embedding_projection.weight":
                    #TODO: no need transpose?
                    
                    weights = self.vision_mm_weight_rearrange(weights)                    
                elif new_name.endswith("ffn_down.weight") or new_name.endswith("ffn_gate.weight") or new_name.endswith("ffn_up.weight") \
                    or new_name.endswith("k_proj.weight") or new_name.endswith("q_proj.weight")\
                    or new_name.endswith("v_proj.weight") or new_name.endswith("out_proj.weight") \
                    or new_name.endswith("gate_proj.weight") or new_name.endswith("up_proj.weight") or new_name.endswith("down_proj.weight"):
                        
                        
                    weights = self.vision_mm_weight_rearrange(weights)


                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Gemma4 model")
        self._export_q4nx_tensors(q4nx_path)
