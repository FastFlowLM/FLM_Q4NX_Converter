#!/usr/bin/env python3
"""Numerically verify the QUANTIZED (Q4_1 / Q8_0) tensors of a Q4NX model.

The float verifier (verify_moe_q4nx.py) only structure-checks quantized
tensors. This tool decodes the packed blocks INDEPENDENTLY of the converter's
pack code and compares against the reference weight the converter intended to
store (GGUF dequant + the same untile transforms), re-quantized with the gguf
package's own quantize()/dequantize().

If the untiles, the pack layout, or the block order were wrong, cosine drops
well below 1.0 and the tensor fails.

Usage:
  venv/bin/python tools/verify_q4nx_quant.py --q4nx-dir <dir> --ref-gguf <gguf> [--layers 0,19,39]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from q4nx.models.qwen35moe import Qwen35Moe  # noqa: E402

Q4_1 = Qwen35Moe._Q4_1_NAMES


def is_q4_1(name):
    return any(name.endswith(s) for s in Q4_1)


def bf16_view(buf):
    return torch.frombuffer(buf, dtype=torch.bfloat16).float()


def decode_q8_0_block(buf, r=32, c=256, par=16, gs=32):
    """buf: 8704 bytes -> (32, 256) float32. v = q * d[cg, row]."""
    ng = c // gs  # col-groups per block
    d = bf16_view(buf[0 : r * ng * 2]).reshape(ng, r)  # [cg, row]
    data = torch.frombuffer(buf[r * ng * 2 :], dtype=torch.int8).reshape(
        r // par, c, par
    ).float()  # [g, col, p]
    g = torch.arange(r // par).view(r // par, 1, 1)
    cidx = torch.arange(c).view(1, c, 1)
    p = torch.arange(par).view(1, 1, par)
    row = g * par + p  # broadcast (g, c, p) -> absolute row
    cg = cidx // gs
    v = data * d[cg, row]
    return v.permute(0, 2, 1).reshape(r, c)  # [g,p,c] -> [row, c]


def decode_q4_1_block(buf, r=32, c=256, par=16, gs=32):
    """buf: 5120 bytes -> (32, 256) float32. v = q*d + m, q in [0,15]."""
    ng = c // gs
    d = bf16_view(buf[0 : r * ng * 2]).reshape(ng, r)  # [cg, row]
    m = bf16_view(buf[r * ng * 2 : r * ng * 4]).reshape(ng, r)
    qw = torch.frombuffer(buf[r * ng * 4 :], dtype=torch.uint8).reshape(
        r // par, c, r // 2 // (r // par)
    )  # [g, col, row_pair]
    low = (qw & 0x0F).float()
    high = ((qw >> 4) & 0x0F).float()
    g = torch.arange(r // par).view(r // par, 1, 1)
    cidx = torch.arange(c).view(1, c, 1)
    rp = torch.arange(r // 2 // (r // par)).view(1, 1, -1)  # row_pairs per half
    cg = cidx // gs
    row = g * par + 2 * rp  # absolute row for the low nibble
    dl = d[cg, row]
    ml = m[cg, row]
    dlo = d[cg, row + 1]
    mlo = m[cg, row + 1]
    v0 = low * dl + ml  # [g, c, rp] -> out[row, c]
    v1 = high * dlo + mlo
    out = torch.empty(r, c, dtype=torch.float32)
    for gg in range(r // par):
        for rrr in range(r // 2 // (r // par)):
            rr = gg * par + 2 * rrr
            out[rr, :] = v0[gg, :, rrr]
            out[rr + 1, :] = v1[gg, :, rrr]
    return out


def decode_tensor(t):
    """t: safetensors int8 tensor shaped (R, C, block_bytes) -> (R*32, C*256) float32."""
    buf = t.numpy().tobytes()
    R, C, B = t.shape
    block = 5120 if B == 5120 else 8704
    dec = decode_q4_1_block if block == 5120 else decode_q8_0_block
    cols = C * 256
    rows = R * 32
    out = torch.empty(rows, cols, dtype=torch.float32)
    for p in range(R):
        for q in range(C):
            off = (p * C + q) * block
            out[p * 32 : (p + 1) * 32, q * 256 : (q + 1) * 256] = dec(
                buf[off : off + block]
            )
    return out


def cosine(a, b):
    a = a.float().reshape(-1).double()
    b = b.float().reshape(-1).double()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float((a @ b) / denom) if denom else 0.0


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q4nx-dir", required=True)
    ap.add_argument("--ref-gguf", required=True)
    ap.add_argument("--layers", default="0,19,39", help="comma list of layer ids to check")
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    q4nx_path = Path(args.q4nx_dir) / "model.q4nx"
    st = load_file(str(q4nx_path))
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    reader = GGUFReader(args.ref_gguf)
    by_name = {t.name: t for t in reader.tensors}

    def ref_float(gguf_name):
        return torch.from_numpy(dequantize(by_name[gguf_name].data, by_name[gguf_name].tensor_type)).to(torch.float32)

    q = Qwen35Moe.__new__(Qwen35Moe)
    q.LINEAR_NUM_KEY_HEADS = Qwen35Moe.LINEAR_NUM_KEY_HEADS
    q.LINEAR_NUM_VALUE_HEADS = Qwen35Moe.LINEAR_NUM_VALUE_HEADS
    q.LINEAR_KEY_HEAD_DIM = Qwen35Moe.LINEAR_KEY_HEAD_DIM
    q.Q_PROJ_P = Qwen35Moe.Q_PROJ_P
    q.HEAD_DIM = Qwen35Moe.HEAD_DIM
    q._untile_head_params = Qwen35Moe._untile_head_params.__get__(q)
    q._untile_linear_heads = Qwen35Moe._untile_linear_heads.__get__(q)
    q._untile_linear_rows = Qwen35Moe._untile_linear_rows.__get__(q)
    q._untile_linear_cols = Qwen35Moe._untile_linear_cols.__get__(q)

    # Q4NX suffix -> (gguf_src, ref-transform fn)
    T = {
        "linear_attn.qkv_proj.weight": lambda w, q: torch.cat(
            [w[: w.shape[0] // 2], q._untile_linear_rows(w[w.shape[0] // 2 :])], dim=0
        ).contiguous(),
        "self_attn.gate_proj.weight": lambda w, q: q._untile_linear_rows(w),
        "linear_attn.ssm_out_proj.weight": lambda w, q: q._untile_linear_cols(w),
        "self_attn.q_proj.weight": lambda w, q: w.contiguous(),
        "self_attn.k_proj.weight": lambda w, q: w.contiguous(),
        "self_attn.v_proj.weight": lambda w, q: w.contiguous(),
        "self_attn.o_proj.weight": lambda w, q: w.contiguous(),
        "mlp.share_gate_exps_proj.weight": lambda w, q: w.contiguous(),
        "mlp.share_up_exps_proj.weight": lambda w, q: w.contiguous(),
        "mlp.share_down_exps_proj.weight": lambda w, q: w.reshape(-1, w.shape[-1]).contiguous(),
        "mlp.gate_exps_proj.weight": lambda w, q: w.reshape(-1, w.shape[-1]).contiguous(),
        "mlp.up_exps_proj.weight": lambda w, q: w.reshape(-1, w.shape[-1]).contiguous(),
        "mlp.down_exps_proj.weight": lambda w, q: w.reshape(-1, w.shape[-1]).contiguous(),
    }
    G = {
        "linear_attn.qkv_proj.weight": "blk.{b}.attn_qkv.weight",
        "self_attn.gate_proj.weight": "blk.{b}.attn_gate.weight",
        "linear_attn.ssm_out_proj.weight": "blk.{b}.ssm_out.weight",
        "self_attn.q_proj.weight": "blk.{b}.attn_q.weight",
        "self_attn.k_proj.weight": "blk.{b}.attn_k.weight",
        "self_attn.v_proj.weight": "blk.{b}.attn_v.weight",
        "self_attn.o_proj.weight": "blk.{b}.attn_output.weight",
        "mlp.share_gate_exps_proj.weight": "blk.{b}.ffn_gate_shexp.weight",
        "mlp.share_up_exps_proj.weight": "blk.{b}.ffn_up_shexp.weight",
        "mlp.share_down_exps_proj.weight": "blk.{b}.ffn_down_shexp.weight",
        "mlp.gate_exps_proj.weight": "blk.{b}.ffn_gate_exps.weight",
        "mlp.up_exps_proj.weight": "blk.{b}.ffn_up_exps.weight",
        "mlp.down_exps_proj.weight": "blk.{b}.ffn_down_exps.weight",
    }

    failures = 0
    checked = 0
    worst = []

    def check(name, gguf_name, tf):
        nonlocal failures, checked
        if name not in st or gguf_name not in by_name:
            return
        checked += 1
        w = ref_float(gguf_name)
        w = tf(w, q)
        qt = GGMLQuantizationType.Q4_1 if is_q4_1(name) else GGMLQuantizationType.Q8_0
        w_req = torch.from_numpy(dequantize(quantize(w.numpy(), qt), qt)).float()
        w_dec = decode_tensor(st[name])
        if w_req.shape != w_dec.shape:
            print(f"[FAIL] {name}: shape q4nx={tuple(w_dec.shape)} ref={tuple(w_req.shape)}")
            failures += 1
            return
        c = cosine(w_dec, w_req)
        m = maxabs(w_dec, w_req)
        status = "OK " if c >= args.threshold else "FAIL"
        if c < args.threshold:
            failures += 1
            worst.append((name, c, m))
        print(f"[{status}] {name}: cos={c:.6f} maxerr={m:.6f} q={qt.name}")

    # globals
    check("lm_head.weight", "output.weight", lambda w, q: w.contiguous())

    for bid in layers:
        for suffix, tf in T.items():
            name = f"model.layer.{bid}.{suffix}"
            check(name, G[suffix].format(b=bid), tf)

    print("\n=== Summary ===")
    print(f"Quant-checked: {checked}, failures: {failures}")
    for name, c, m in sorted(worst, key=lambda x: x[1])[:10]:
        print(f"  worst: {name} cos={c:.6f} maxerr={m:.6f}")
    print("PASS" if failures == 0 and checked > 0 else "FAIL")


if __name__ == "__main__":
    main()
