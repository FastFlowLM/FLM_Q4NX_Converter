from enum import IntEnum, auto
from typing import Tuple

from gguf.constants import GGMLQuantizationType
import numpy as np

class ModelArch(IntEnum):
    QWEN3VL = auto()
    QWEN3   = auto()
    QWEN35_08B = auto()
    QWEN35_2B = auto()
    QWEN35_4B = auto()
    QWEN35_9B = auto()
    QWEN2   = auto()
    QWEN2VL = auto()
    GEMMA3  = auto()
    GEMMA4  = auto()
    LLAMA   = auto()
    LFM2    = auto()
    PHI4    = auto()
    GPT_OSS = auto()
    NANBEIGE = auto()


ModelArchNames: dict[ModelArch, list[str]] = {
    ModelArch.QWEN35_08B: ["qwen35-0.8B","qwen3.5-0.8B"],
    ModelArch.QWEN35_4B:  ["qwen35-4B","qwen3.5-4B"],    
    ModelArch.QWEN35_9B:  ["qwen35-9B","qwen3.5-9B"],       
    ModelArch.QWEN35_2B:  ["qwen35-2B","qwen3.5-2B"],       
    ModelArch.QWEN3VL: ["qwen3vl", "qwen3-vl"],
    ModelArch.QWEN2VL: ["qwen2.5-Vl"],    
    ModelArch.QWEN2:   ["qwen2"],
    ModelArch.QWEN3:   ["qwen3"],
    ModelArch.GEMMA3:  ["gemma3", "Medgemma", "Gemma-3"],
    ModelArch.GEMMA4:  ["gemma4"],
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
    ModelArch.QWEN35_4B: "qwen3.5_4b.json",
    ModelArch.QWEN35_9B: "qwen3.5_9b.json",
    ModelArch.QWEN35_2B: "qwen3.5_2b.json",
    ModelArch.QWEN35_08B: "qwen3.5_0.8b.json",
    ModelArch.GEMMA3:  "gemma3.json",
    ModelArch.GEMMA4:  "gemma4.json",
    ModelArch.LLAMA:   "llama.json",
    ModelArch.LFM2:    "lfm2.json",
    ModelArch.PHI4:    "phi4.json",
    ModelArch.GPT_OSS: "gpt-oss.json",
    ModelArch.NANBEIGE: "nanbeige.json"
}