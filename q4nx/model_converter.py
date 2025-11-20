

from abc import ABC, abstractmethod
from gguf import GGUFReader
from .constants import ModelArch, ModelArchNames
from .constants import ModelArchConfigs
from .gguf_tensor import GGUFTensor
from typing import List, Dict, Type
import os
import json

from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
from .utils import round_up_to_multiple, get_relativeL2, get_relativeL1, get_rmse, get_cosine_similarity, create_dir_if_not_exists
from safetensors.torch import save_file

# Registry to store model classes by architecture
_MODEL_REGISTRY: Dict[ModelArch, Type['__Q4NX_Converter']] = {}

class __Q4NX_Converter(ABC):
    model_arch: ModelArch
    gguf_reader: GGUFReader
    gguf_tensors: Dict[str, GGUFTensor]
    q4nx_tensors: Dict[str, torch.Tensor]
    hidden_size: int
    num_layers: int
    q4nx_config: json

    row_block_size: int
    col_block_size: int
    parallel_size: int
    keep_block_in_2D: bool

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
    def convert(self, q4nx_path: str):
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
            if field.name == f"{ModelArchNames[self.model_arch]}.embedding_length":
                self.hidden_size = field.contents()
            elif field.name == f"{ModelArchNames[self.model_arch]}.feed_forward_length":
                self.intermediate_size = field.contents()
            elif field.name == f"{ModelArchNames[self.model_arch]}.block_count":
                self.num_layers = field.contents()

    def _load_config(self, config_file_path: str = "configs"):
        config_path = os.path.join(config_file_path, ModelArchConfigs[self.model_arch])
        self.q4nx_config = json.load(open(config_path))
        self.row_block_size = self.q4nx_config["q4nx_config"]["row_block_size"]
        self.col_block_size = self.q4nx_config["q4nx_config"]["col_block_size"]
        self.parallel_size = self.q4nx_config["q4nx_config"]["parallel_size"]
        self.keep_block_in_2D = self.q4nx_config["q4nx_config"]["keep_block_in_2D"]
        self._create_name_maps()

    def _create_name_maps(self):
        self.forward_name_map = {}
        self.backward_name_map = {}
        for param_name, param_info in self.q4nx_config["name_map"].items():
            if ("{bid}" in param_info["gguf_name"]):
                for bid in range(self.num_layers):
                    self.forward_name_map[param_info["gguf_name"].format(bid=bid)] = param_info["q4nx_name"].format(bid=bid)
                    self.backward_name_map[param_info["q4nx_name"].format(bid=bid)] = param_info["gguf_name"].format(bid=bid)
                    if bid == 0:
                        print(f"Converted {param_info['gguf_name'].format(bid=bid)} to {param_info['q4nx_name'].format(bid=bid)}")
            else:
                self.forward_name_map[param_info["gguf_name"]] = param_info["q4nx_name"]
                self.backward_name_map[param_info["q4nx_name"]] = param_info["gguf_name"]
                print(f"Converted {param_info['gguf_name']} to {param_info['q4nx_name']}")

        # sort the name map by the name alphabetically
        self.forward_name_map = dict(sorted(self.forward_name_map.items(), key=lambda item: item[0]))
        self.backward_name_map = dict(sorted(self.backward_name_map.items(), key=lambda item: item[0]))


    def _has_lm_head(self) -> bool:
        for key in self.gguf_tensors.keys():
            if "lm_head.weight" in key:
                return True 
        return False

    def _export_q4nx_tensors(self, q4nx_path: str):
        create_dir_if_not_exists(q4nx_path)
        save_file(self.q4nx_tensors, q4nx_path)

    # def _convert_tensor(self, gguf_tensor: GGUFTensor) -> torch.Tensor:
        

    def _pack_q4nx(self, d: torch.Tensor, m: torch.Tensor = None, qw: torch.Tensor = None) -> torch.Tensor:
        """
        Q4NX format:
            - chunk shape: 32 x 256
            - parallel size: 16
            - scale shape: (256 // 32) x 32
            - min shape:   (256 // 32) x 32
            - quant shape: (32 // 16) x 256 x (16 // 2)

        Therefore:
            - d: (rows, cols // 32) -> (rows // 32, cols // 32 // 256, 256 // 32, 32)
            - m: (rows, cols // 32) -> (rows // 32, cols // 32 // 256, 256 // 32, 32)
        """
        if m is None: # not packed, could be a float or bf16 tensor
            return d.to(torch.bfloat16)

        rows, cols = qw.shape
        group_size =  32

        if cols%self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            
            d_m_pad_amount = (cols_padded -cols) // 32
            qw_pad_amount = cols_padded-cols
            
            d = F.pad(d, (0, d_m_pad_amount), "constant", 0)
            m = F.pad(m, (0, d_m_pad_amount), "constant", 0)
            qw = F.pad(qw, (0, qw_pad_amount), "constant", 0)
            
            
        if not self.keep_block_in_2D:
            # chunk wise
            d = rearrange(d, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // group_size).contiguous()
            # dm done

            qw = rearrange(qw, '(p r) (q c) -> (p q) r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'n (g r) c -> n g r c', r = self.parallel_size)
            qw = rearrange(qw, 'n g (r b) c -> n g c r b', b = 2).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'n g c r -> n (g c r)').contiguous()
        else:
            # chunk wise
            d = rearrange(d, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // group_size).contiguous()
            # dm done

            qw = rearrange(qw, '(p r) (q c) -> p q r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'p q (g r) c -> p q g r c', r = self.parallel_size)
            qw = rearrange(qw, 'p q g (r b) c -> p q g c r b', b = 2).contiguous().to(torch.int8)

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


def get_model_arch_from_gguf(reader: GGUFReader) -> ModelArch:
    """
    Read the model architecture from a GGUF file.
    
    Args:
        gguf_path: Path to the GGUF file
        
    Returns:
        ModelArch enum value
        
    Raises:
        ValueError: If the architecture is not recognized or supported
    """
    
    # Get architecture from GGUF metadata
    # The architecture is typically stored in the 'general.architecture' field
    arch_str = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            arch_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
            break
    
    # Map the architecture string to ModelArch enum
    for arch_enum, arch_name in ModelArchNames.items():
        if arch_str.lower() == arch_name.lower():
            return arch_enum
    
    raise ValueError(f"Unsupported model architecture: {arch_str}")


def get_registered_models() -> Dict[ModelArch, Type['__Q4NX_Converter']]:
    """
    Get the dictionary of registered model converters.
    
    Returns:
        Dictionary mapping ModelArch to converter classes
    """
    return _MODEL_REGISTRY.copy()


def create_converter(gguf_path: str) -> __Q4NX_Converter:
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
    model_arch = get_model_arch_from_gguf(reader)
    
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