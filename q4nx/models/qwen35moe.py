"""Qwen3.5 MoE (qwen35moe / qwen3.6-moe) converter: HF safetensors -> Q4NX.

The target Q4NX layout mirrors the official FastFlowLM Qwen3.6-35B-A3B model.
Derived from the proven dense Qwen3.5 converter conventions and the official
Q4NX header (733 tensors, dtype policy):

- V heads stay in HF grouped order. The dense converter (num_k=16 != num_v=32,
  which llama.cpp tiles in GGUF) is re-untiled to grouped order by its proven
  (g q) reorders, so the engine consumes grouped order. No V reorder here.
- Full-attn q_proj only: (g p h) -> (p g h) head reorder, p=2, h=head_dim,
  matching the dense converter.
- Layernorm weights stored as weight + 1 (llama.cpp / dense-converter
  convention), except linear_attn.norm (ssm_norm) which is stored raw.
- A_log -> -exp(A_log); conv1d squeezed + transposed; alpha/beta transposed.
- dtype policy: BF16 norms/router/gates/alpha/beta/conv1d/ssm_norm/embed,
  F32 ssm_a + ssm_dt.bias, Q4_1 the three big expert mats, Q8_0 everything
  else quantized.
"""

from pathlib import Path
import json
import os

import numpy as np
import torch
from einops import rearrange
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors import safe_open

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from ..model_converter import __Q4NX_Converter

# Known names (suffixes) that appear without the trailing ".weight".
_HF_GATE_UP = "mlp.experts.gate_up_proj"
_HF_EXPERT_DOWN = "mlp.experts.down_proj"


class Qwen35Moe(__Q4NX_Converter, model_arch=ModelArch.QWEN35MOE):
    FULL_ATTENTION_INTERVAL = 4
    LINEAR_NUM_KEY_HEADS = 16
    LINEAR_NUM_VALUE_HEADS = 32
    LINEAR_KEY_HEAD_DIM = 128
    LINEAR_VALUE_HEAD_DIM = 128
    HEAD_DIM = 256
    Q_PROJ_P = 2
    NUM_ATTN_HEADS = 16
    NUM_KV_HEADS = 2

    def __init__(self, source: str | GGUFReader, config_json_path: str | None = None):
        print("[INFO] Using Qwen35Moe converter")
        self.gguf_reader: GGUFReader | None = None
        self.gguf_tensors: dict = {}
        self.hf_source: str | None = None
        self.hf_dir: Path | None = None
        self.weight_map: dict = {}
        self.hf_shards: dict = {}
        self.q4nx_tensors: dict = {}
        if isinstance(source, GGUFReader):
            print("[INFO] Qwen35Moe converter (GGUF source)")
            self.gguf_reader = source
            self.gguf_tensors = {t.name: t for t in source.tensors}
            self.initialize(config_json_path=config_json_path)
        else:
            print("[INFO] Qwen35Moe converter (HF safetensors source)")
            self.hf_source = source
            self.hf_dir = self._resolve_source(source)
            self.initialize(config_json_path=config_json_path)

    # ------------------------------------------------------------------ setup

    def initialize(self, config_json_path: str | None = None):
        self._load_config(config_file_path=config_json_path)
        if self.gguf_reader is None:
            self._read_hf_index()

    def _resolve_source(self, source: str) -> Path:
        path = Path(source)
        if path.is_dir():
            return path
        if "/" in source:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError("huggingface_hub is required to download HF sources")
            print(f"[INFO] Downloading {source} to HF cache...")
            local = snapshot_download(repo_id=source, allow_patterns=[
                "*.safetensors", "model.safetensors.index.json",
                "config.json", "tokenizer*", "*.json", "*.jinja",
            ])
            return Path(local)
        raise FileNotFoundError(f"HF source not found: {source}")

    def _read_hf_index(self):
        idx_path = self.hf_dir / "model.safetensors.index.json"
        if idx_path.is_file():
            index = json.loads(idx_path.read_text())
            self.weight_map = index["weight_map"]
            shards = sorted(set(self.weight_map.values()))
        else:
            single = self.hf_dir / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"No safetensors weights found in {self.hf_dir}"
                )
            with safe_open(single, framework="torch") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
            shards = ["model.safetensors"]
        self.hf_shards = {s: self.hf_dir / s for s in shards}
        print(f"[INFO] Loaded {len(self.weight_map)} HF tensors across {len(shards)} shards")

    def _load_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.hf_shards[shard], framework="torch") as f:
            return f.get_tensor(name).contiguous()

    # ------------------------------------------------------------- conversion

    def convert(self, q4nx_path: str, weights_type: str = "language"):
        if weights_type != "language":
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35Moe")
        self.q4nx_tensors = {}
        if self.gguf_reader is not None:
            for name in sorted(self.gguf_tensors):
                self._process_gguf_tensor(name)
            print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
            self._export_weights(q4nx_path, weights_type)
            self._extract_tokenizer_json(q4nx_path)
        else:
            self._convert_hf(q4nx_path, weights_type)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        """HF safetensors path. Supports fused experts and per-expert gate/up/down."""
        # Collect per-expert pieces so we can stack expert-major once per layer.
        expert_gate: dict[int, dict[int, torch.Tensor]] = {}
        expert_up: dict[int, dict[int, torch.Tensor]] = {}
        expert_down: dict[int, dict[int, torch.Tensor]] = {}

        for name in sorted(self.weight_map):
            key = name.replace("model.language_model.", "")
            if key.startswith("visual.") or key.startswith("model.visual."):
                continue
            if ".mlp.experts." in key and key.split(".")[-1] == "weight":
                # layers.{bid}.mlp.experts.{eid}.{gate,up,down}_proj.weight
                parts = key.split(".")
                if len(parts) >= 6 and parts[2] == "mlp" and parts[3] == "experts":
                    bid = int(parts[1])
                    eid = int(parts[4])
                    kind = parts[5]  # gate_proj / up_proj / down_proj
                    w = self._load_tensor(name)
                    if kind == "gate_proj":
                        expert_gate.setdefault(bid, {})[eid] = w
                    elif kind == "up_proj":
                        expert_up.setdefault(bid, {})[eid] = w
                    elif kind == "down_proj":
                        expert_down.setdefault(bid, {})[eid] = w
                    else:
                        print(f"[WARN] Unhandled expert tensor: {name}")
                    continue
            self._process_tensor(name)

        for bid in sorted(set(expert_gate) | set(expert_up) | set(expert_down)):
            prefix = f"model.layer.{bid}."
            if bid in expert_gate:
                eids = sorted(expert_gate[bid])
                gate = torch.stack([expert_gate[bid][e] for e in eids], dim=0)
                self._store_q(
                    prefix + "mlp.gate_exps_proj.weight",
                    gate.reshape(-1, gate.shape[-1]),
                )
            if bid in expert_up:
                eids = sorted(expert_up[bid])
                up = torch.stack([expert_up[bid][e] for e in eids], dim=0)
                self._store_q(
                    prefix + "mlp.up_exps_proj.weight",
                    up.reshape(-1, up.shape[-1]),
                )
            if bid in expert_down:
                eids = sorted(expert_down[bid])
                down = torch.stack([expert_down[bid][e] for e in eids], dim=0)
                self._store_q(
                    prefix + "mlp.down_exps_proj.weight",
                    down.reshape(-1, down.shape[-1]),
                )

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        self._export_weights(q4nx_path, weights_type)

    # ------------------------------------------------------- GGUF conversion

    def _deq_gguf(self, name: str) -> torch.Tensor:
        """Dequantize a GGUF tensor to a float32 torch tensor, GGUF orientation.

        gguf.dequantize() already returns a correctly-shaped numpy array
        (row-major, e.g. [n_vocab, n_embd] for token_embd.weight). GGUFTensor.shape
        reports GGML's reversed ne[] axis order (e.g. [n_embd, n_vocab]) and must
        NOT be used to reshape the dequantized array - doing so reinterprets the
        buffer with the wrong strides and scrambles every element while leaving
        summary statistics (mean/abs) misleadingly unchanged.
        """
        gt = self.gguf_tensors[name]
        w = dequantize(gt.data, gt.tensor_type).copy()
        return torch.from_numpy(w).to(torch.float32).contiguous()

    def _untile_head_params(self, w: torch.Tensor) -> torch.Tensor:
        """llama.cpp tiled head params [NVP, NK] -> engine grouped order [NK, NVP]."""
        nvp = self.LINEAR_NUM_VALUE_HEADS // self.LINEAR_NUM_KEY_HEADS
        nk = self.LINEAR_NUM_KEY_HEADS
        w = w.reshape(nvp, nk).T.reshape(nk * nvp)
        return w.to(torch.float32).contiguous()

    def _untile_linear_rows(self, w: torch.Tensor) -> torch.Tensor:
        """Undo llama.cpp value-major tiling on linear-attn row dim: (q g p)->(g q p)."""
        return rearrange(
            w, "(q g p) c -> (g q p) c", p=self.LINEAR_KEY_HEAD_DIM, q=2
        ).contiguous()

    def _untile_linear_cols(self, w: torch.Tensor) -> torch.Tensor:
        """Undo llama.cpp value-major tiling on linear-attn col dim: (q g p)->(g q p)."""
        return rearrange(
            w, "r (q g p) -> r (g q p)", p=self.LINEAR_KEY_HEAD_DIM, q=2
        ).contiguous()

    def _untile_linear_heads(self, w: torch.Tensor) -> torch.Tensor:
        """Undo llama.cpp tiling on rank/head vectors: (q g)->(g q)."""
        return rearrange(w, "(q g) c -> (g q) c", q=2).contiguous()

    def _process_gguf_tensor(self, gguf_name: str):
        # --- globals ---
        # gguf.dequantize() already returns the natural [out, in] / [vocab, hidden]
        # numpy shape (same orientation as the HF safetensors path) - no transpose.
        if gguf_name == "token_embd.weight":
            self.q4nx_tensors["model.embed_tokens.weight"] = self._bf16(self._deq_gguf(gguf_name))
            return
        if gguf_name == "output.weight":
            self._store_q("lm_head.weight", self._deq_gguf(gguf_name))
            return
        if gguf_name == "output_norm.weight":
            self.q4nx_tensors["model.norm.weight"] = self._bf16(self._deq_gguf(gguf_name))
            return
        if gguf_name.startswith("blk."):
            self._process_gguf_layer_tensor(gguf_name)
            return
        print(f"[WARN] Unhandled GGUF tensor: {gguf_name}")

    def _process_gguf_layer_tensor(self, gguf_name: str):
        # name like: blk.0.attn_qkv.weight
        parts = gguf_name.split(".")
        bid = int(parts[1])
        rest = ".".join(parts[2:])
        prefix = f"model.layer.{bid}."

        w = self._deq_gguf(gguf_name)

        # --- norms (GGUF already stores the Q4NX weight + 1 convention) ---
        if rest == "attn_norm.weight":
            self.q4nx_tensors[prefix + "input_layernorm.weight"] = self._bf16(w)
            return
        if rest == "post_attention_norm.weight":
            self.q4nx_tensors[prefix + "post_attention_layernorm.weight"] = self._bf16(w)
            return
        if rest == "ssm_norm.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_norm.weight"] = self._bf16(w)
            return
        if rest == "attn_q_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.q_norm.weight"] = self._bf16(w)
            return
        if rest == "attn_k_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.k_norm.weight"] = self._bf16(w)
            return

        # --- linear-attention scalars / small projections (BF16/F32) ---
        # llama.cpp tiles linear-attn heads value-major; engine wants HF grouped
        # order. Matches dense qwen35.py reorder_linear_required path.
        # GGUF already stores A_log as -exp(A_log).
        if rest == "ssm_a":
            self.q4nx_tensors[prefix + "linear_attn.ssm_a"] = self._untile_head_params(w)
            return
        if rest == "ssm_dt.bias":
            self.q4nx_tensors[prefix + "linear_attn.ssm_dt.bias"] = self._untile_head_params(w)
            return
        if rest == "ssm_alpha.weight":
            w = self._untile_linear_heads(w)
            self.q4nx_tensors[prefix + "linear_attn.ssm_alpha_proj.weight"] = self._bf16(w.t())
            return
        if rest == "ssm_beta.weight":
            w = self._untile_linear_heads(w)
            self.q4nx_tensors[prefix + "linear_attn.ssm_beta_proj.weight"] = self._bf16(w.t())
            return
        if rest == "ssm_conv1d.weight":
            # Second half of channels is value-tiled; first half stays.
            w0, w1 = w.chunk(2, dim=0)
            w = torch.cat([w0, self._untile_linear_rows(w1)], dim=0).contiguous()
            self.q4nx_tensors[prefix + "linear_attn.ssm_conv1d.weight"] = self._bf16(w.t())
            return

        # --- router / shared-expert gates (BF16) ---
        # gguf.dequantize() returns ffn_gate_inp as natural [n_experts, hidden];
        # Q4NX wants [hidden, n_experts] (matches the HF mlp.gate.weight.t() convention).
        if rest == "ffn_gate_inp.weight":
            self.q4nx_tensors[prefix + "moe_router.weight"] = self._bf16(w.t())
            return
        if rest == "ffn_gate_inp_shexp.weight":
            self.q4nx_tensors[prefix + "shared_expert_gate.weight"] = self._bf16(w.reshape(-1))
            return

        # --- quantized 2D weights ---
        # Linear-attn mats need the same llama.cpp untile as dense qwen35.
        # Full-attn q in GGUF is HF-ordered (g p h); the engine wants (p g h)
        # (matches the dense converter and the official Q4NX layout).
        if rest == "attn_qkv.weight":
            w0, w1 = w.chunk(2, dim=0)
            w = torch.cat([w0, self._untile_linear_rows(w1)], dim=0).contiguous()
            self._store_q(prefix + "linear_attn.qkv_proj.weight", w)
            return
        if rest == "attn_gate.weight":
            w = self._untile_linear_rows(w)
            self._store_q(prefix + "self_attn.gate_proj.weight", w)
            return
        if rest == "ssm_out.weight":
            w = self._untile_linear_cols(w)
            self._store_q(prefix + "linear_attn.ssm_out_proj.weight", w)
            return
        if rest == "attn_q.weight":
            w = rearrange(w, "(g p h) c -> (p g h) c", p=self.Q_PROJ_P, h=self.HEAD_DIM).contiguous()
            self._store_q(prefix + "self_attn.q_proj.weight", w)
            return
        if rest == "attn_k.weight":
            self._store_q(prefix + "self_attn.k_proj.weight", w)
            return
        if rest == "attn_v.weight":
            self._store_q(prefix + "self_attn.v_proj.weight", w)
            return
        if rest == "attn_output.weight":
            self._store_q(prefix + "self_attn.o_proj.weight", w)
            return
        if rest == "ffn_gate_shexp.weight":
            self._store_q(prefix + "mlp.share_gate_exps_proj.weight", w)
            return
        if rest == "ffn_up_shexp.weight":
            self._store_q(prefix + "mlp.share_up_exps_proj.weight", w)
            return
        if rest == "ffn_down_shexp.weight":
            self._store_q(prefix + "mlp.share_down_exps_proj.weight", w)
            return

        # --- routed experts: gguf.dequantize() natural shape is already
        # [n_experts, inter, hidden] - just flatten expert-major, no permute. ---
        if rest == "ffn_gate_exps.weight":
            w = w.reshape(-1, w.shape[-1]).contiguous()
            self._store_q(prefix + "mlp.gate_exps_proj.weight", w)
            return
        if rest == "ffn_up_exps.weight":
            w = w.reshape(-1, w.shape[-1]).contiguous()
            self._store_q(prefix + "mlp.up_exps_proj.weight", w)
            return
        if rest == "ffn_down_exps.weight":
            w = w.reshape(-1, w.shape[-1]).contiguous()
            self._store_q(prefix + "mlp.down_exps_proj.weight", w)
            return

        print(f"[WARN] Unhandled GGUF layer tensor: {gguf_name}")

    def _process_tensor(self, hf_name: str):
        key = hf_name.replace("model.language_model.", "")
        if key.startswith("layers."):
            self._process_layer_tensor(hf_name, key)
            return
        # globals
        if key == "embed_tokens.weight":
            self.q4nx_tensors["model.embed_tokens.weight"] = self._bf16(self._load_tensor(hf_name))
        elif key == "lm_head.weight":
            self._store_q("lm_head.weight", self._load_tensor(hf_name))
        elif key == "norm.weight":
            w = self._bf16(self._load_tensor(hf_name).float() + 1)
            self.q4nx_tensors["model.norm.weight"] = w
        else:
            print(f"[WARN] Unhandled global tensor: {hf_name}")

    def _process_layer_tensor(self, hf_name: str, key: str):
        # key like: layers.0.linear_attn.in_proj_qkv.weight
        parts = key.split(".")
        bid = int(parts[1])
        rest = ".".join(parts[2:])
        prefix = f"model.layer.{bid}."

        w = self._load_tensor(hf_name)

        # --- layernorms (weight + 1, except linear_attn.norm) ---
        if rest == "input_layernorm.weight":
            self.q4nx_tensors[prefix + "input_layernorm.weight"] = self._bf16(w.float() + 1)
            return
        if rest == "post_attention_layernorm.weight":
            self.q4nx_tensors[prefix + "post_attention_layernorm.weight"] = self._bf16(w.float() + 1)
            return

        # --- linear attention ---
        if rest == "linear_attn.in_proj_qkv.weight":
            self._store_q(prefix + "linear_attn.qkv_proj.weight", w)
            return
        if rest == "linear_attn.in_proj_z.weight":
            self._store_q(prefix + "self_attn.gate_proj.weight", w)
            return
        if rest == "linear_attn.in_proj_a.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_alpha_proj.weight"] = self._bf16(w.t())
            return
        if rest == "linear_attn.in_proj_b.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_beta_proj.weight"] = self._bf16(w.t())
            return
        if rest == "linear_attn.out_proj.weight":
            self._store_q(prefix + "linear_attn.ssm_out_proj.weight", w)
            return
        if rest == "linear_attn.conv1d.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_conv1d.weight"] = self._bf16(w.squeeze().t())
            return
        if rest == "linear_attn.A_log":
            self.q4nx_tensors[prefix + "linear_attn.ssm_a"] = (-torch.exp(w.float())).contiguous()
            return
        if rest == "linear_attn.dt_bias":
            self.q4nx_tensors[prefix + "linear_attn.ssm_dt.bias"] = w.to(torch.float32).contiguous()
            return
        if rest == "linear_attn.norm.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_norm.weight"] = self._bf16(w)
            return

        # --- full attention ---
        if rest == "self_attn.q_proj.weight":
            w = rearrange(w, "(g p h) c -> (p g h) c", p=self.Q_PROJ_P, h=self.HEAD_DIM).contiguous()
            self._store_q(prefix + "self_attn.q_proj.weight", w)
            return
        if rest == "self_attn.k_proj.weight":
            self._store_q(prefix + "self_attn.k_proj.weight", w)
            return
        if rest == "self_attn.v_proj.weight":
            self._store_q(prefix + "self_attn.v_proj.weight", w)
            return
        if rest == "self_attn.o_proj.weight":
            self._store_q(prefix + "self_attn.o_proj.weight", w)
            return
        if rest == "self_attn.q_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.q_norm.weight"] = self._bf16(w.float() + 1)
            return
        if rest == "self_attn.k_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.k_norm.weight"] = self._bf16(w.float() + 1)
            return

        # --- MLP / MoE ---
        if rest == "mlp.gate.weight":
            self.q4nx_tensors[prefix + "moe_router.weight"] = self._bf16(w.t())
            return
        if rest == "mlp.shared_expert_gate.weight":
            self.q4nx_tensors[prefix + "shared_expert_gate.weight"] = self._bf16(w.reshape(-1))
            return
        if rest == "mlp.shared_expert.gate_proj.weight":
            self._store_q(prefix + "mlp.share_gate_exps_proj.weight", w)
            return
        if rest == "mlp.shared_expert.up_proj.weight":
            self._store_q(prefix + "mlp.share_up_exps_proj.weight", w)
            return
        if rest == "mlp.shared_expert.down_proj.weight":
            self._store_q(prefix + "mlp.share_down_exps_proj.weight", w)
            return
        if rest == _HF_GATE_UP:
            # [num_experts, 2 * moe_intermediate, hidden] -> gate + up, flattened expert-major
            gate, up = w.chunk(2, dim=1)
            self._store_q(prefix + "mlp.gate_exps_proj.weight", gate.reshape(-1, gate.shape[-1]))
            self._store_q(prefix + "mlp.up_exps_proj.weight", up.reshape(-1, up.shape[-1]))
            return
        if rest == _HF_EXPERT_DOWN:
            self._store_q(prefix + "mlp.down_exps_proj.weight", w.reshape(-1, w.shape[-1]))
            return

        print(f"[WARN] Unhandled tensor: {hf_name}")

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _bf16(w: torch.Tensor) -> torch.Tensor:
        return w.to(torch.bfloat16).contiguous()

    _Q4_1_NAMES = {
        "mlp.gate_exps_proj.weight",
        "mlp.up_exps_proj.weight",
        "mlp.down_exps_proj.weight",
    }

    def _store_q(self, q4nx_name: str, w: torch.Tensor):
        """Quantize + pack a 2D weight into the Q4NX block layout."""
        target = (
            GGMLQuantizationType.Q4_1
            if q4nx_name.endswith(tuple(Qwen35Moe._Q4_1_NAMES))
            else GGMLQuantizationType.Q8_0
        )
        w_np = w.to(torch.float32).numpy()
        quantized = quantize(w_np, target).copy()
        columns = w_np.shape[1]
        if target == GGMLQuantizationType.Q4_1:
            d, m, qw = GGUFTensor.unpack_q4_1(quantized, columns)
            self.q4nx_tensors[q4nx_name] = self._pack(d, m, qw, tensor_type=target)
        else:
            d, _, qw = GGUFTensor.unpack_q8_0(quantized, columns)
            self.q4nx_tensors[q4nx_name] = self._pack(d, None, qw, tensor_type=target)
