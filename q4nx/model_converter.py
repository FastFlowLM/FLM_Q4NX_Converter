

from abc import ABC, abstractmethod
from gguf import GGUFReader
from .constants import ModelArch, ModelArchNames
from .constants import ModelArchConfigs
from .gguf_tensor import GGUFTensor, GGMLQuantizationType
from typing import List, Dict, Type
import os
import json
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
from .utils import round_up_to_multiple, get_relativeL2, get_relativeL1, get_rmse, get_cosine_similarity, create_dir_if_not_exists
from safetensors.torch import save_file
from q4nx.gguf_tensor import GGUFTensor
from gguf import Q8_0, GGUFReader, dequantize, quantize, GGMLQuantizationType
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
    parallel_size: int   # vector len size for efficient parallel vector operation
    keep_block_in_2D: bool
    default_q4nx_tensor_type: GGMLQuantizationType 

    
    
    # specific for vision models
    vision_MM_K:int|None
    vision_MM_N:int|None
    
    forward_name_map: Dict[str, str]
    backward_name_map: Dict[str, str]

    def __init__(self):
        raise TypeError("This class is virtual, do not instantiate it directly")


    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.model_arch = model_arch
        # Register the subclass in the registry
        _MODEL_REGISTRY[model_arch] = cls

    def initialize(self):
        self._read_gguf_tensors()
        self._read_gguf_metadata()
        self._load_config()

    @abstractmethod
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        pass

    def _read_gguf_tensors(self):
        """
        Load the GGUF file and parse the tensors.

        Returns:
            None
        """
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

    def _create_name_maps(self):
        print("[INFO] Creating name maps...")
        self.forward_name_map = {}
        self.backward_name_map = {}
        for param_name, param_info in self.q4nx_config["name_map"].items():
            if ("{bid}" in param_info["gguf_name"]):
                for bid in range(self.num_layers):
                    self.forward_name_map[param_info["gguf_name"].format(bid=bid)] = param_info["q4nx_name"].format(bid=bid)
                    self.backward_name_map[param_info["q4nx_name"].format(bid=bid)] = param_info["gguf_name"].format(bid=bid)
                    if bid == 0:
                        print(f"\tConverted {param_info['gguf_name'].format(bid=bid)} to {param_info['q4nx_name'].format(bid=bid)}")
            else:
                self.forward_name_map[param_info["gguf_name"]] = param_info["q4nx_name"]
                self.backward_name_map[param_info["q4nx_name"]] = param_info["gguf_name"]
                print(f"\tConverted {param_info['gguf_name']} to {param_info['q4nx_name']}")

        # sort the name map by the name alphabetically
        self.forward_name_map = dict(sorted(self.forward_name_map.items(), key=lambda item: item[0]))
        self.backward_name_map = dict(sorted(self.backward_name_map.items(), key=lambda item: item[0]))


    def _has_lm_head(self) -> bool:
        for key in self.gguf_tensors.keys():
            if "lm_head.weight" in key or key == "output.weight":
                return True 
        return False

    def _export_q4nx_tensors(self, q4nx_path: str):
        print(f"[INFO] Saving Q4NX tensors to {q4nx_path}/model.q4nx...")
        create_dir_if_not_exists(q4nx_path)
        save_file(self.q4nx_tensors, os.path.join(q4nx_path, "model.q4nx"))

    def _pack_MXFP4_q4nx(self, scales:torch.Tensor, data:torch.Tensor,    
            )->torch.Tensor:
    
        MXFP4_BLOCK_SIZE= 32
        MXFP4_BLOCK_SIZE_data_in_byte = 16 # because 4 bit data, so 32 data is 16 byte
        

        
        #TODO: for now, only support safetensor with multiple expert, aka expect  scalse to be shape of [num_expert/batch, rows, cols]
        # expect data to be shape of [num_expert/batch, rows, cols_in_byte/MXFP4_BLOCK_SIZE_data_in_byte, MXFP4_BLOCK_SIZE_data_in_byte ]        
        assert len(scales.shape) == 3
        assert len(data.shape) ==4
        assert scales.shape[0] == data.shape[0] and scales.shape[1] == data.shape[1] and scales.shape[2] == data.shape[2]
        assert data.shape[3] == MXFP4_BLOCK_SIZE_data_in_byte
        
        
        if scales.shape[1] % self.row_block_size !=0:
            padd_size = (self.row_block_size - scales.shape[1]) % self.row_block_size
            scales = F.pad(scales, (0, 0, 0, padd_size), "constant", 0)
            data = F.pad(data, (0, 0, 0,0, 0, self.row_block_size - data.shape[-3] % self.row_block_size), "constant", 0)
            
        if (scales.shape[2] * MXFP4_BLOCK_SIZE) % self.col_block_size !=0:
            addition_padd_size = (self.col_block_size - ((scales.shape[2] * MXFP4_BLOCK_SIZE) % self.col_block_size)) // MXFP4_BLOCK_SIZE

            scales: torch.Tensor = F.pad(scales, (0, addition_padd_size, 0, 0), "constant", 0)
            data = F.pad(data, (0, 0, 0, addition_padd_size, 0, 0), "constant", 0)
        


        # # calcuate dimension for scales and biases

        scales= scales.contiguous()
        data = data.contiguous()
  
        

        row_div_q4_row = scales.shape[1] // self.row_block_size
        col_div_q4_col = scales.shape[2] // (self.col_block_size // MXFP4_BLOCK_SIZE)
        scales = rearrange(
            scales, 
            "batch (row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> batch row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
            row_div_q4_row=row_div_q4_row,  
            col_div_q4_col=col_div_q4_col,
            q4_row=self.row_block_size, 
            q4_col_div32=self.col_block_size//MXFP4_BLOCK_SIZE
        ).contiguous()            
        


        # combine the block dim
        # Get the dimensions *before* the last two and pass the full shape to view()
        data: torch.Tensor = data.reshape(*data.shape[:-2], -1)
        assert len(data.shape) == 3
        data_row_div = data.shape[1] // self.row_block_size
        data_col_div = data.shape[2] // (self.col_block_size // 2)
        data = rearrange(
            data,
            "batch (row_div_q4_row q4_row) (col_div_q4_col q4_col) -> batch row_div_q4_row col_div_q4_col q4_row q4_col",
            row_div_q4_row=data_row_div,
            col_div_q4_col=data_col_div,
            q4_row=self.row_block_size,
            q4_col=(self.col_block_size // 2)
        ).contiguous()

        # at this step, both scales and data are 
            # 1. row major within the blocks
            # 2. Also row major in block level


        assert self.row_block_size % self.parallel_size == 0
        
        # Now, rearrange data to column major order with stride of 16
        data = rearrange(
            data,
            "batch row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
            batch row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
            col_stride = self.parallel_size, # 16 element, since each data is half-byte
            one = 1,
        ).contiguous()
        
        # at this point, the data is shape of 
        # the data block is col-major order with col_stride
        #[batch, row_div, col_div (q4_row_div_col_stride x q4_col )  , col_stride]
        
        # however, for each col_stride(normally 16) the data is in uint8_t
        # thus, it actually mean the last_dim, col_stride is 2 column of 16
        """
        
        EX: if col_stride=16 of uint8_t
        [
            a0,
            a1,
            a2,
            a3,
            a4
            ...
        ]
        
        But in int4, this actuall represnt
        [
            a0_0, a0_1,
            a1_0, a1_1,
            a2_0, a2_1,
            ......
            
        ]
        where a0_0, a0_1 share a common scale
        Thus, this mean the AIE kernel need to do a even_odd filter, as show in code 
        https://github.com/ngdxzy/FastFlowLM_Dev/blob/30b43b59d77f5759e943cea52ff7a259ca0fa776/npu_framework/gpt_npu_bin/kernel/mvm_MXFP4.hpp#L76
        """
        
        # Thus, in this code, let us do the even odd filter for it
        """
        Thus, we reorder the col_stride from
        [
            a0_0, a0_1,a1_0, a1_1, a2_0, a2_1,
            ......
        ]
        to 
        [
            a0_0, a1_0, a2_0, .... a0_1, a1_1
            
        ]
        """
        
        # data shape: [..., col_stride] (e.g., [..., 16])
        # Assumes col_stride is an even number (like 16)

        # 1. Extract low and high nibble streams
        # low_nibbles shape: [..., 16], content: [a0_0, a1_0, a2_0, ..., a15_0]
        low_nibbles = data & 0x0F
        # high_nibbles shape: [..., 16], content: [a0_1, a1_1, a2_1, ..., a15_1]
        high_nibbles = (data >> 4) & 0x0F

        # 2. Pack the low_nibbles stream
        # [a0_0, a1_0, a2_0, ...] -> [ (a0_0 | a1_0<<4), (a2_0 | a3_0<<4), ... ]
        # Slicing [..., 0::2] gets evens: [a0_0, a2_0, ..., a14_0]
        # Slicing [..., 1::2] gets odds: [a1_0, a3_0, ..., a15_0]
        low_part_packed = (low_nibbles[..., 0::2] & 0x0F) | ((low_nibbles[..., 1::2] & 0x0F) << 4)
        # low_part_packed shape: [..., 8]

        # 3. Pack the high_nibbles stream
        # [a0_1, a1_1, a2_1, ...] -> [ (a0_1 | a1_1<<4), (a2_1 | a3_1<<4), ... ]
        high_part_packed = (high_nibbles[..., 0::2] & 0x0F) | ((high_nibbles[..., 1::2] & 0x0F) << 4)
        # high_part_packed shape: [..., 8]

        # 4. Concatenate the two 8-byte halves to form the new 16-byte col_stride
        # Final shape: [..., 16]
        data_reordered = torch.cat([low_part_packed, high_part_packed], dim=-1)
        data = data_reordered.contiguous()
        # NOW, this code change is reflected in https://github.com/ngdxzy/FastFlowLM_Dev/commit/028680d1f670d817fae0e7efe947bb1a4c19c8a3
        
        
        
        # but for scale, simply change to column major order, NOTE: the stride of 16 does not apply to scale here 
        # aka, scales remain to be  self.row_block_size x (self.col_block_size//MXFP4_BLOCK_SIZE) but in colum major order  
        scales = rearrange(
            scales,
            "batch row_div col_div q4_row q4_col -> \
                     batch row_div col_div q4_col q4_row"
        ).contiguous()
        
   


        # # Apply the padding.
        # scales_expanded = F.pad(scales, paddings, "constant", 0)
        scales = scales.reshape(*scales.shape[:-2], -1).contiguous()
        padding_amount = scales.shape[-1] * 3  # padding to align with q4nx requirement. Because in q41, 2byte for scale and 2byte for bias
                                                    # given mxfp4 there is only 1 byte for scale, so need to pad 3 byte
        paddings = (0, padding_amount)
        scales = F.pad(scales, paddings, "constant", 0).contiguous()


        
        # at this point, 
        # data should be shape of [batch, row_div col_div, q4_row, q4_col]
        # scales should be shape of [batch, row_div, col_div, (self.col_block_size//MXFP4_BLOCK_SIZE), q4_row*4]
        
        # now, given they are both uint8, first change data and scales shape by combine the last two dim
        # data now be [batch, row_div, col_div, q4_row*q4_col]
        #scale shoule be shape of [batch, row_div, col_div, (self.col_block_size//MXFP4_BLOCK_SIZE) * q4_row*4 ]

        data = data.reshape(*data.shape[:-2], -1).contiguous()
        # scales = scales_expanded.reshape(*scales_expanded.shape[:-2], -1).contiguous()
        
        
        merged = torch.cat([scales, data], dim=-1).contiguous()
        return merged
        
    
  
    def force_pack_q8_to_q4nx_size(self, tensor_data:GGUFTensor):
        
            # TODO: FIXME: DEBUG force convert Q80
            w = dequantize(tensor_data.data, tensor_data.tensor_type)
            w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
            w = w.to(dtype=torch.float32).numpy()
            
            data_q80 = quantize(w, GGMLQuantizationType.Q8_0).copy()
            # create a new GGUF type
            data_q80_gguf = GGUFTensor("data_q80", data_q80.shape, data_q80, GGMLQuantizationType.Q8_0)
                        
            unpacked =data_q80_gguf.unpack(self.default_tensor_type)

            
            # override to turn keep_block_in_2D = True
            old_keep_block_in_2D =  self.keep_block_in_2D
            self.keep_block_in_2D = True
            val = self._pack_q4nx_8b(*unpacked)
            self.keep_block_in_2D = old_keep_block_in_2D
            
            return val
            # scales, data = GGUFTensor.unpack_q8_0( data_q80)
            
            # m_tmp = scales.clone()
            
            # # override 

            # col_block_size_old = self.col_block_size
            # keep_block_in_2D_old = self.keep_block_in_2D
            
            # cur_q4nx_block_byte_size = int(( self.row_block_size* col_block_size_old   )*(5/8) )
            
            # if col_block_size_old == 256:
            #     self.col_block_size= 128
            # else:
            #     #TODO:
            #     raise ValueError("Undefine case for now")
            # self.keep_block_in_2D= True
            # q8nx_pack_result  = self._pack_q8nx(
            #     scales, m_tmp, data
            # )
            
            # # now, we want to padd the last dimension to be size of 5120(for q4nx)
            # # Pad the last dimension from 4608 to 5120

            
            # padding_size = cur_q4nx_block_byte_size    - q8nx_pack_result.shape[-1]
            # q8nx_pack_result = F.pad(q8nx_pack_result, (0, padding_size))
            
            # self.keep_block_in_2D = keep_block_in_2D_old
            # self.col_block_size = col_block_size_old
            # return q8nx_pack_result
            
            
    def _pack(self, d: torch.Tensor, m: torch.Tensor = None, qw: torch.Tensor = None, tensor_type: GGMLQuantizationType = None) -> torch.Tensor:
        if tensor_type == GGMLQuantizationType.Q8_0:
            return self._pack_q4nx_8b(d, m, qw)
        else:
            return self._pack_q4nx(d, m, qw)
        
    
    def _pack_q4nx_8b(self,  d: torch.Tensor,m: torch.Tensor, qw:torch.Tensor) -> torch.Tensor:
        #note, support q80 for now
        # d for scale
        # m for min
        
        # TODO: NOTE:
        col_block_size_old = self.col_block_size
        keep_block_in_2D_old = self.keep_block_in_2D        
        if self.col_block_size == 256:
            # force to 128 
            self.col_block_size= 128            
        else:
            raise ValueError("Undefine case for now")
        self.keep_block_in_2D= True
        
        cur_q4nx_block_byte_size = int(( self.row_block_size* col_block_size_old   )*(5/8) )
                    
        q8nx_pack_result = self._pack_q8nx(data=qw, scales=d, m = m )

        padding_size = cur_q4nx_block_byte_size    - q8nx_pack_result.shape[-1]
        q8nx_pack_result = F.pad(q8nx_pack_result, (0, padding_size))
        
        self.keep_block_in_2D = keep_block_in_2D_old
        self.col_block_size = col_block_size_old
        return q8nx_pack_result
        
    def _pack_q8nx(self,  data: torch.Tensor,scales: torch.Tensor, m:torch.Tensor) -> torch.Tensor:
        #note, support q80 for now
        """ Q8NX format similar to Q4NX

        Q8NX format:
            - chunk shape: row_block_size x col_block_size
            - scale shape: (col_block_size // Q8_group_size) x row_block_size
            - m shape: (col_block_size // Q8_group_size) x row_block_size
            - quant shape: (row_block_size // parallel) x col_block_size x (parallel) #NOTE: num_int8_in_byte is 1 for Q8
        
        
        Parameters
        ----------
        scales : torch.Tensor
            _description_
        data : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
            
        Notes
        -------
        Only support Q8_0 for now
        """
        
        
        Q8_group_size =  32
        assert len(scales.shape) == 3
        assert len(data.shape) ==3
        # data are shape of [row, col//Q8_group_size, Q8_group_size]
        # merge last two dim of scales and data
        scales = scales.reshape(*scales.shape[:-2], -1).contiguous()
        data = data.reshape(*data.shape[:-2], -1).contiguous()
        m = m.reshape(*m.shape[:-2], -1).contiguous()
        
        rows, cols = data.shape[0], data.shape[1]
        
        

        
        if cols % self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            
            scale_pad_amount = (cols_padded -cols) // Q8_group_size
            data_pad_amount = cols_padded - cols
            
            scales = F.pad(scales, (0, scale_pad_amount), "constant", 0)
            m = F.pad(m, (0, scale_pad_amount), "constant", 0)
            data = F.pad(data, (0, data_pad_amount), "constant", 0)
        

        
        if self.keep_block_in_2D:
            
            # does two things
            # 1. split scales to logical block of (row_div_r, col_div_c, r, c) but in row major order
            # 2. change the block of (r,c) to column major order as (c,r)
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
            # similar, for the data block
            data = rearrange(
                data,
                "(row_div_r r) (col_div_c c) -> row_div_r col_div_c r c",
                r=self.row_block_size,
                c=self.col_block_size 
            ).contiguous()
            # at this point, data is in row major order within each block
            
            data = rearrange(
                data,
                #" row_div_r col_div_c (r_div_parallel parallel) c -> row_div_r col_div_c (c r_div_parallel parallel)",
                
                "row_div_r col_div_c (r_div_parallel parallel) c -> row_div_r col_div_c (r_div_parallel c parallel)",
                parallel=self.parallel_size,
            )
            # at this point, data is in column major order within each block, and strided by parallel size
            
            # now, convert scales from float16 to bfloat16
            assert(scales.dtype == torch.float16)
            scales = scales.to(torch.bfloat16)
            
            # also convert m from float16 to bfloat16
            assert(m.dtype == torch.float16)
            m = m.to(torch.bfloat16)
        
            
        else:
            raise ValueError("Only support keep_block_in_2D for now")
        
        scales = scales.view(torch.int8)
        m = m.view(torch.int8)
        data = data.view(torch.int8)
        
        scales_np = scales.numpy()
        m_np = m.numpy()
        data_np = data.numpy()
        # do  a copy of scales_np for now, for padding space
        merged = np.concatenate([scales_np,  m_np, data_np], axis = -1).copy()
        
        return torch.from_numpy(merged)       
    
    def _pack_q4nx(self, d: torch.Tensor, m: torch.Tensor = None, qw: torch.Tensor = None) -> torch.Tensor:
        """

        
        Q4NX format:
            - chunk shape: row_block_size x col_block_size
            - scale shape: (col_block_size // Q4_group_size) x row_block_size
            - min shape:   (col_block_size // Q4_group_size) x row_block_size
            - quant shape: (row_block_size // parallel) x col_block_size x (parallel // NUM_int4_in_byte)

        Therefore:
            - d: (rows, cols // Q4_group_size) -> (rows // row_block_size, cols // (col_block_size * Q4_group_size), col_block_size // Q4_group_size, Q4_group_size)
            - m: (rows, cols // Q4_group_size) -> (rows // row_block_size, cols // (col_block_size * Q4_group_size), col_block_size // Q4_group_size, Q4_group_size)
        """
        Q4_group_size =  32
        NUM_int4_in_byte = 2
        d = d.contiguous()
        m = m.contiguous() if m is not None else None
        qw = qw.contiguous() if qw is not None else None
        
        if m is None: # not packed, could be a float or bf16 tensor
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
            # chunk wise
            d = rearrange(d, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            # dm done

            qw = rearrange(qw, '(p r) (q c) -> (p q) r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'n (g r) c -> n g r c', r = self.parallel_size)
            qw = rearrange(qw, 'n g (r b) c -> n g c r b', b = NUM_int4_in_byte).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'n g c r -> n (g c r)').contiguous()
        else:
            # chunk wise
            d = rearrange(d, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            # dm done

            qw = rearrange(qw, '(p r) (q c) -> p q r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'p q (g r) c -> p q g r c', r = self.parallel_size)
            qw = rearrange(qw, 'p q g (r b) c -> p q g c r b', b = NUM_int4_in_byte).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'p q g c r -> p q (g c r)').contiguous()
        d = d.to(torch.bfloat16).view(torch.int8)
        m = m.to(torch.bfloat16).view(torch.int8)
        qw = qw.view(torch.int8)

        d = d.numpy()
        m = m.numpy()
        qw = qw.numpy()

        merged = np.concatenate([d, m, qw], axis = -1).copy()

        merged = torch.from_numpy(merged)
        return merged
    def vision_mm_weight_rearrange(self, weight: torch.Tensor) -> torch.Tensor:
        """Rearrange the vision MM weights


        In vision MM, the weights matrix is used as the B matrix in A@B operation.    
        Given a weight tensor of shape (out_features, in_features) correspond to (N, K)
        
        The function assume input weights is in row-major of (N, K),
        which should view as a K x N matrix in col-major.
        
        For efficent memory access, the function reorder the matrix to be in block of (N//MM_N, K//MM_K)( MM_N, MM_K) where each
        block is in row-major order and the blocks are also in row-major order.
        
        Then data can also be viewed as in col-major order with block of (K//MM_K, N//MM_N) within each (MM_K, MM_N) block in col-major order.

        For simplicity, the row, column dimension of the matrix is padded to be multiple of max(MM_N, MM_K). 
        Because in MLP operations, the output dim of up/gate projection needs to match the input dim of down projection.
        In other words, the N dim of up/gate projection equals to the K dim of down projection.
    
    
        Parameters
        ----------
        weight : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            Reorder vision weights 

        Notes
        ----------
        mm_n and mm_k are the data block size that each CT can process in one time.
        
        """
        
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
        """
        Extract tokenizer information from GGUF file and convert to HuggingFace tokenizer.json format.
        
        Args:
            output_folder: Path to save the tokenizer.json file
            
        Returns:
            Dictionary containing tokenizer configuration
        """
        print("[INFO] Extracting tokenizer JSON...")
        tokenizer_json_path = os.path.join(output_folder, "tokenizer.json")
        
        # Extract tokenizer metadata from GGUF
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
            # Tokenizer model type
            if field.name == "tokenizer.ggml.model":
                tokenizer_model = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
            
            # Token list
            elif field.name == "tokenizer.ggml.tokens":
                tokens = [str(field.parts[i], encoding='utf-8') for i in field.data]
            
            # Token scores (for SentencePiece)
            elif field.name == "tokenizer.ggml.scores":
                scores = field.data.tolist() if hasattr(field.data, 'tolist') else list(field.data)
            
            # Token types
            elif field.name == "tokenizer.ggml.token_type":
                token_types = field.data.tolist() if hasattr(field.data, 'tolist') else list(field.data)
            
            # BPE merges
            elif field.name == "tokenizer.ggml.merges":
                merges = [str(field.parts[i], encoding='utf-8') for i in field.data]
            
            # Special token IDs
            elif field.name == "tokenizer.ggml.bos_token_id":
                bos_token_id = int(field.contents())
                print(f"[INFO] BOSS token ID: {bos_token_id}")
            elif field.name == "tokenizer.ggml.eos_token_id":
                eos_token_id = int(field.contents())
                print(f"[INFO] EOS token ID: {eos_token_id}")
            elif field.name == "tokenizer.ggml.unknown_token_id":
                unk_token_id = int(field.contents())
                print(f"[INFO] Unknown token ID: {unk_token_id}")
            elif field.name == "tokenizer.ggml.padding_token_id":
                pad_token_id = int(field.contents())
                print(f"[INFO] Padding token ID: {pad_token_id}")
        
        # Build vocab dictionary for HuggingFace format
        vocab = {}
        for idx, token in enumerate(tokens):
            vocab[token] = idx
        
        # Determine tokenizer type and create appropriate HuggingFace config
        if tokenizer_model == "llama" or tokenizer_model == "gpt2":
            # BPE tokenizer
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
            # Default to SentencePiece-like tokenizer (for llama, etc.)
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
        
        # Save tokenizer.json
        create_dir_if_not_exists(output_folder)
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Tokenizer saved to {tokenizer_json_path}")
        return tokenizer_json
    
    def _build_added_tokens(self, tokens: list, bos_id: int, eos_id: int, unk_id: int, pad_id: int) -> list:
        """
        Build the added_tokens list for HuggingFace tokenizer format.
        
        Args:
            tokens: List of all tokens
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            unk_id: Unknown token ID
            pad_id: Padding token ID
            
        Returns:
            List of added token dictionaries
        """
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
    """
    Read the model architecture from a GGUF file.
    
    Args:
        gguf_path: Path to the GGUF file
        override_model_arch: Non-empty string override this model architecture type
        
    Returns:
        ModelArch enum value
        
    Raises:
        ValueError: If the architecture is not recognized or supported
    """
    
    if override_model_arch != "":
        for arch_enum, arch_names in ModelArchNames.items():
            for arch_name in arch_names:            
                if override_model_arch.lower().startswith(arch_name.lower()):
                    return arch_enum
        print("Warning: Did not find matching override model arch, attempting to load base on gguf information")

    
    # Get architecture from GGUF metadata
    # The architecture is typically stored in the 'general.architecture' field
    arch_str:str|None = None
    basename_str:str|None = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            arch_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
        elif field.name == 'general.basename':
            basename_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
    

    # Map the architecture string to ModelArch enum
    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if arch_str.lower() == arch_name.lower():
                return arch_enum
    

    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if basename_str.lower().startswith(arch_name.lower()):
                return arch_enum


    raise ValueError(f"Unsupported model architecture: {arch_str}")


def get_registered_models() -> Dict[ModelArch, Type['__Q4NX_Converter']]:
    """
    Get the dictionary of registered model converters.
    
    Returns:
        Dictionary mapping ModelArch to converter classes
    """
    return _MODEL_REGISTRY.copy()


def create_converter(gguf_path: str, override_model_arch:str) -> __Q4NX_Converter:
    """
    Factory function to create the appropriate converter based on the GGUF model architecture.
    
    Args:
        gguf_path: Path to the GGUF file
        
    Returns:
        An instance of the appropriate converter class
        
    Raises:
        ValueError: If the architecture is not recognized or no converter is available
        
    Example:
        >>> converter = create_converter("model.gguf")
        >>> converter.load_gguf("model.gguf")
        >>> converter.convert("configs")
    """
    # Read the architecture from the GGUF file
    reader = GGUFReader(gguf_path)
    model_arch = get_model_arch_from_gguf(reader,override_model_arch )
    
    # Look up the appropriate converter class
    if model_arch not in _MODEL_REGISTRY:
        available = [ModelArchNames.get(arch, str(arch)) for arch in _MODEL_REGISTRY.keys()]
        raise ValueError(
            f"No converter available for architecture: {ModelArchNames.get(model_arch, 'unknown')}. "
            f"Available converters: {available if available else 'None (no models imported?)'}"
        )
    
    converter_class = _MODEL_REGISTRY[model_arch]

    converter_instance = converter_class(reader)

    return converter_instance