from enum import IntEnum, auto
from typing import Tuple

from gguf.constants import GGMLQuantizationType
import numpy as np

class ModelArch(IntEnum):
    QWEN3VL = auto()
    QWEN3   = auto()
    QWEN35   = auto()
    QWEN2   = auto()
    QWEN2VL = auto()
    GEMMA3  = auto()
    LLAMA   = auto()
    LFM2    = auto()
    PHI4    = auto()
    GPT_OSS = auto()
    NANBEIGE = auto()


ModelArchNames: dict[ModelArch, list[str]] = {
    ModelArch.QWEN35:  ["qwen35","qwen3.5"],    
    ModelArch.QWEN3VL: ["qwen3vl", "qwen3-vl"],
    ModelArch.QWEN2VL: ["qwen2.5-Vl"],    
    ModelArch.QWEN2:   ["qwen2"],
    ModelArch.QWEN3:   ["qwen3"],
    ModelArch.GEMMA3:  ["gemma3", "Medgemma", "Gemma-3"],
    ModelArch.LLAMA:   ["llama"],
    ModelArch.LFM2:    ["lfm2"],
    ModelArch.PHI4:    ["phi3"],
    ModelArch.GPT_OSS: ["gpt-oss"],
    ModelArch.NANBEIGE: ["nanbeige"]
}

ModelArchConfigs: dict[ModelArch, str] = {
    ModelArch.QWEN3VL: "qwen3vl.json",
    ModelArch.QWEN2:   "qwen2.json",
    ModelArch.QWEN2VL: "qwen2vl.json",
    ModelArch.QWEN3:   "qwen3.json",
    ModelArch.QWEN35:  "qwen3.5.json",
    ModelArch.GEMMA3:  "gemma3.json",
    ModelArch.LLAMA:   "llama.json",
    ModelArch.LFM2:    "lfm2.json",
    ModelArch.PHI4:    "phi4.json",
    ModelArch.GPT_OSS: "gpt-oss.json",
    ModelArch.NANBEIGE: "nanbeige.json"
}