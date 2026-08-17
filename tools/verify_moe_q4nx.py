#!/usr/bin/env python3
"""Verify a Qwen3.5/3.6-MoE Q4NX conversion against reference weights.

Reference may be a GGUF (dequantized on the fly) or an HF safetensors dir /
cached repo id. Every non-quantized (BF16/F32) Q4NX tensor is compared by
cosine + max-abs error against the reference, re-applying the exact storage
transform qwen35moe.py uses (un-tiling, transpose, weight+1, -exp(A_log), ...).
Quantized (Q4_1/Q8_0) tensors get structural checks: shape and packed-block
byte size (Q4_1 = 5120 B/block, Q8_0 = 8704 B/block).

Usage:
  venv/bin/python tools/verify_moe_q4nx.py --q4nx-dir <dir> --ref-gguf <gguf>
  venv/bin/python tools/verify_moe_q4nx.py --q4nx-dir <dir> --ref-hf <dir-or-repo>
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from gguf import GGUFReader, dequantize
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from q4nx.models.qwen35moe import Qwen35Moe  # noqa: E402


# ------------------------------------------------------------------ reference

class GGUFRef:
    def __init__(self, path):
        self.reader = GGUFReader(path)
        self.by_name = {t.name: t for t in self.reader.tensors}
        self.is_gguf = True

    def get(self, name):
        t = self.by_name[name]
        return torch.from_numpy(dequantize(t.data, t.tensor_type)).to(torch.float32)

    def has(self, name):
        return name in self.by_name


class HFRef:
    def __init__(self, path):
        from q4nx.model_assets import cached_snapshot_dir

        self.path = Path(path)
        if not self.path.is_dir():
            snapshot = cached_snapshot_dir(path)
            if snapshot is None:
                raise FileNotFoundError(f"HF source not found locally: {path}")
            self.path = snapshot
        self.q = Qwen35Moe(str(self.path))
        self.q._read_hf_index()
        self.is_gguf = False

    def get(self, name):
        return self.q._load_tensor(name)

    def has(self, name):
        return name in self.q.weight_map


# ------------------------------------------- per-tensor source + transforms

# name suffix -> (gguf_source, hf_source)
SRC = {
    "linear_attn.ssm_a": ("blk.{b}.ssm_a", "layers.{b}.linear_attn.A_log"),
    "linear_attn.ssm_dt.bias": ("blk.{b}.ssm_dt.bias", "layers.{b}.linear_attn.dt_bias"),
    "linear_attn.ssm_alpha_proj.weight": ("blk.{b}.ssm_alpha.weight", "layers.{b}.linear_attn.in_proj_a"),
    "linear_attn.ssm_beta_proj.weight": ("blk.{b}.ssm_beta.weight", "layers.{b}.linear_attn.in_proj_b"),
    "linear_attn.ssm_conv1d.weight": ("blk.{b}.ssm_conv1d.weight", "layers.{b}.linear_attn.conv1d"),
    "linear_attn.ssm_norm.weight": ("blk.{b}.ssm_norm.weight", "layers.{b}.linear_attn.norm"),
    "input_layernorm.weight": ("blk.{b}.attn_norm.weight", "layers.{b}.input_layernorm"),
    "post_attention_layernorm.weight": ("blk.{b}.post_attention_norm.weight", "layers.{b}.post_attention_layernorm"),
    "self_attn.q_norm.weight": ("blk.{b}.attn_q_norm.weight", "layers.{b}.self_attn.q_norm"),
    "self_attn.k_norm.weight": ("blk.{b}.attn_k_norm.weight", "layers.{b}.self_attn.k_norm"),
    "moe_router.weight": ("blk.{b}.ffn_gate_inp.weight", "layers.{b}.mlp.gate"),
    "shared_expert_gate.weight": ("blk.{b}.ffn_gate_inp_shexp.weight", "layers.{b}.mlp.shared_expert_gate"),
    # quantized (structural only) - GGUF names listed for shape reference
    "linear_attn.qkv_proj.weight": ("blk.{b}.attn_qkv.weight", None),
    "self_attn.gate_proj.weight": ("blk.{b}.attn_gate.weight", None),
    "linear_attn.ssm_out_proj.weight": ("blk.{b}.ssm_out.weight", None),
    "self_attn.q_proj.weight": ("blk.{b}.attn_q.weight", None),
    "self_attn.k_proj.weight": ("blk.{b}.attn_k.weight", None),
    "self_attn.v_proj.weight": ("blk.{b}.attn_v.weight", None),
    "self_attn.o_proj.weight": ("blk.{b}.attn_output.weight", None),
    "mlp.share_gate_exps_proj.weight": ("blk.{b}.ffn_gate_shexp.weight", None),
    "mlp.share_up_exps_proj.weight": ("blk.{b}.ffn_up_shexp.weight", None),
    "mlp.share_down_exps_proj.weight": ("blk.{b}.ffn_down_shexp.weight", None),
    "mlp.gate_exps_proj.weight": ("blk.{b}.ffn_gate_exps.weight", None),
    "mlp.up_exps_proj.weight": ("blk.{b}.ffn_up_exps.weight", None),
    "mlp.down_exps_proj.weight": ("blk.{b}.ffn_down_exps.weight", None),
}

# suffixes whose Q4NX dtype is Q4_1 (rest quantized are Q8_0)
Q4_1 = Qwen35Moe._Q4_1_NAMES


def _t(w):
    return w.t()


def make_transforms(q):
    """Return {suffix: {'gguf': fn, 'hf': fn}} for float tensors."""

    def g_untile_head(w):
        return q._untile_head_params(w).to(torch.float32)

    return {
        "linear_attn.ssm_a": {
            "gguf": g_untile_head,
            "hf": lambda w: (-torch.exp(w.float())).contiguous(),
        },
        "linear_attn.ssm_dt.bias": {
            "gguf": g_untile_head,
            "hf": lambda w: w.to(torch.float32).contiguous(),
        },
        "linear_attn.ssm_alpha_proj.weight": {
            "gguf": lambda w: q._untile_linear_heads(w).t().to(torch.bfloat16).contiguous(),
            "hf": lambda w: w.t().to(torch.bfloat16).contiguous(),
        },
        "linear_attn.ssm_beta_proj.weight": {
            "gguf": lambda w: q._untile_linear_heads(w).t().to(torch.bfloat16).contiguous(),
            "hf": lambda w: w.t().to(torch.bfloat16).contiguous(),
        },
        "linear_attn.ssm_conv1d.weight": {
            "gguf": lambda w: torch.cat(
                [w[: w.shape[0] // 2], q._untile_linear_rows(w[w.shape[0] // 2:])], dim=0
            ).t().to(torch.bfloat16).contiguous(),
            "hf": lambda w: w.squeeze().t().to(torch.bfloat16).contiguous(),
        },
        "linear_attn.ssm_norm.weight": {
            "gguf": lambda w: w.to(torch.bfloat16).contiguous(),
            "hf": lambda w: w.to(torch.bfloat16).contiguous(),
        },
        "input_layernorm.weight": {
            "gguf": lambda w: w.to(torch.bfloat16).contiguous(),
            "hf": lambda w: (w.float() + 1).to(torch.bfloat16).contiguous(),
        },
        "post_attention_layernorm.weight": {
            "gguf": lambda w: w.to(torch.bfloat16).contiguous(),
            "hf": lambda w: (w.float() + 1).to(torch.bfloat16).contiguous(),
        },
        "self_attn.q_norm.weight": {
            "gguf": lambda w: w.to(torch.bfloat16).contiguous(),
            "hf": lambda w: (w.float() + 1).to(torch.bfloat16).contiguous(),
        },
        "self_attn.k_norm.weight": {
            "gguf": lambda w: w.to(torch.bfloat16).contiguous(),
            "hf": lambda w: (w.float() + 1).to(torch.bfloat16).contiguous(),
        },
        "moe_router.weight": {
            "gguf": lambda w: w.t().to(torch.bfloat16).contiguous(),
            "hf": lambda w: w.t().to(torch.bfloat16).contiguous(),
        },
        "shared_expert_gate.weight": {
            "gguf": lambda w: w.reshape(-1).to(torch.bfloat16).contiguous(),
            "hf": lambda w: w.reshape(-1).to(torch.bfloat16).contiguous(),
        },
    }


# ------------------------------------------------------------------ compare

def cosine(a, b):
    a = a.float().reshape(-1).double()
    b = b.float().reshape(-1).double()
    if a.shape != b.shape:
        return -2.0
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


def maxabs(a, b):
    a = a.float().reshape(-1).double()
    b = b.float().reshape(-1).double()
    if a.shape != b.shape:
        return -1.0
    return float((a - b).abs().max())


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q4nx-dir", required=True)
    ap.add_argument("--ref-gguf", default=None)
    ap.add_argument("--ref-hf", default=None)
    ap.add_argument("--limit", default=None, type=int, help="limit float comparisons")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    q4nx_dir = Path(args.q4nx_dir)
    q4nx_path = q4nx_dir / "model.q4nx"
    if not q4nx_path.is_file():
        ap.error(f"model.q4nx not found in {q4nx_dir}")
    config = json.loads((q4nx_dir / "config.json").read_text())
    n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)

    st = load_file(str(q4nx_path))
    print(f"[INFO] Loaded {len(st)} Q4NX tensors from {q4nx_path}")

    ref = None
    if args.ref_gguf:
        ref = GGUFRef(args.ref_gguf)
        print(f"[INFO] Reference: GGUF {args.ref_gguf}")
    elif args.ref_hf:
        ref = HFRef(args.ref_hf)
        print(f"[INFO] Reference: HF {ref.path}")
    else:
        print("[WARN] No reference; structural checks only.")

    q = Qwen35Moe.__new__(Qwen35Moe)
    q.LINEAR_NUM_KEY_HEADS = Qwen35Moe.LINEAR_NUM_KEY_HEADS
    q.LINEAR_NUM_VALUE_HEADS = Qwen35Moe.LINEAR_NUM_VALUE_HEADS
    q.LINEAR_KEY_HEAD_DIM = Qwen35Moe.LINEAR_KEY_HEAD_DIM
    q.Q_PROJ_P = Qwen35Moe.Q_PROJ_P
    q.HEAD_DIM = Qwen35Moe.HEAD_DIM
    q._untile_head_params = Qwen35Moe._untile_head_params.__get__(q)
    q._untile_linear_heads = Qwen35Moe._untile_linear_heads.__get__(q)
    q._untile_linear_rows = Qwen35Moe._untile_linear_rows.__get__(q)
    transforms = make_transforms(q)

    checked = skipped = failures = float_n = 0
    cos_sum = 0.0
    worst = []

    def compare(name, src, tf):
        nonlocal checked, skipped, failures, float_n, cos_sum
        if name not in st:
            return
        checked += 1
        t = st[name]
        if tf is None or ref is None or not ref.has(src):
            kind = "Q4_1" if any(name.endswith(s) for s in Q4_1) else "Q8_0"
            # structural: packed block byte size
            nbytes = t.numel()
            block = 5120 if kind == "Q4_1" else 8704
            status = "ok" if nbytes % block == 0 else "MISALIGNED"
            print(f"[CHK] {name}: shape={tuple(t.shape)} bytes={nbytes} ({nbytes % block} mod {kind} block) {status}")
            if status != "ok":
                failures += 1
            return
        r = tf(ref.get(src))
        if r.shape != t.shape:
            sl = tuple(slice(0, min(a, b)) for a, b in zip(r.shape, t.shape))
            if len(r.shape) == len(t.shape):
                r2, t2 = r[sl], t[sl]
            else:
                r2, t2 = r, t
            if r2.shape != t2.shape:
                failures += 1
                worst.append((name, -3.0, -1.0))
                print(f"[FAIL] {name}: shape mismatch q4nx={tuple(t.shape)} ref={tuple(r.shape)}")
                return
            r, t = r2, t2
        c = cosine(t, r)
        m = maxabs(t, r)
        float_n += 1
        cos_sum += c
        if c < 0.999:
            failures += 1
            worst.append((name, c, m))
            print(f"[FAIL] {name}: cos={c:.6f} maxerr={m:.6f} shape={tuple(t.shape)}")
        elif args.verbose:
            print(f"[ OK ] {name}: cos={c:.6f} maxerr={m:.6f} shape={tuple(t.shape)}")

    # globals
    g_embed = "model.embed_tokens.weight"
    g_norm = "model.norm.weight"
    if g_embed in st:
        checked += 1
        if ref is not None and ref.has("token_embd.weight" if ref.is_gguf else "model.language_model.embed_tokens.weight"):
            src = "token_embd.weight" if ref.is_gguf else "model.language_model.embed_tokens.weight"
            r = ref.get(src).to(torch.bfloat16).contiguous()
            t = st[g_embed]
            if r.shape == t.shape:
                c = cosine(t, r)
                m = maxabs(t, r)
                float_n += 1
                cos_sum += c
                if c < 0.999:
                    failures += 1
                    worst.append((g_embed, c, m))
                    print(f"[FAIL] {g_embed}: cos={c:.6f} maxerr={m:.6f}")
                elif args.verbose:
                    print(f"[ OK ] {g_embed}: cos={c:.6f} maxerr={m:.6f}")
            else:
                print(f"[CHK] {g_embed}: shape {tuple(t.shape)} vs ref {tuple(r.shape)} (skip)")
    if g_norm in st:
        checked += 1
        if ref is not None and ref.has("output_norm.weight" if ref.is_gguf else "model.language_model.norm.weight"):
            src = "output_norm.weight" if ref.is_gguf else "model.language_model.norm.weight"
            r = ref.get(src)
            r = r if ref.is_gguf else (r.float() + 1)
            r = r.to(torch.bfloat16).contiguous()
            t = st[g_norm]
            if r.shape == t.shape:
                c = cosine(t, r)
                m = maxabs(t, r)
                float_n += 1
                cos_sum += c
                if c < 0.999:
                    failures += 1
                    worst.append((g_norm, c, m))
                    print(f"[FAIL] {g_norm}: cos={c:.6f} maxerr={m:.6f}")
                elif args.verbose:
                    print(f"[ OK ] {g_norm}: cos={c:.6f} maxerr={m:.6f}")

    for bid in range(n_layers):
        for suffix, (gguf_src, hf_src) in SRC.items():
            if args.limit and float_n >= args.limit:
                break
            name = f"model.layer.{bid}.{suffix}"
            if ref is not None and ref.is_gguf:
                src = gguf_src.format(b=bid)
            elif ref is not None:
                src = f"model.language_model." + hf_src.format(b=bid) if hf_src else None
            else:
                src = gguf_src.format(b=bid)
            tf = transforms.get(suffix, {}).get("gguf" if (ref is None or ref.is_gguf) else "hf")
            compare(name, src, tf)

    print("\n=== Summary ===")
    print(f"Tensor count: {len(st)} (expected 733)")
    print(f"Layers: {n_layers}")
    print(f"Float-compared: {float_n}, mean cos: {cos_sum / max(1, float_n):.6f}")
    print(f"Checked: {checked}, failures: {failures}")
    if worst:
        worst.sort(key=lambda x: x[1])
        for name, c, m in worst[:10]:
            print(f"  worst: {name} cos={c:.6f} maxerr={m:.6f}")
    print("PASS" if failures == 0 and float_n > 0 else "FAIL")


if __name__ == "__main__":
    main()
