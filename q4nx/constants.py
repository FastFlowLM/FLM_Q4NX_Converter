from enum import IntEnum, auto
from typing import Tuple

from gguf.constants import GGMLQuantizationType
import numpy as np

class ModelArch(IntEnum):
    QWEN3VL = auto()
    QWEN3   = auto()
    GEMMA3  = auto()
    LLAMA   = auto()
    LFM2    = auto()


ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3VL: "qwen3vl",
    ModelArch.QWEN3:   "qwen3",
    ModelArch.GEMMA3:  "gemma3",
    ModelArch.LLAMA:   "llama",
    ModelArch.LFM2:    "lfm2"
}

ModelArchConfigs: dict[ModelArch, str] = {
    ModelArch.QWEN3VL: "qwen3vl.json",
    ModelArch.QWEN3:   "qwen3.json",
    ModelArch.GEMMA3:  "gemma3.json",
    ModelArch.LLAMA:   "llama.json",
    ModelArch.LFM2:    "lfm2.json"
}