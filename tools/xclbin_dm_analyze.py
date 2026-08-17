#!/usr/bin/env python3
"""
xclbin_dm_analyze.py — deep analysis of NPU2 xclbin data-memory (DM) contents.

Usage:
  python3 xclbin_dm_analyze.py [expert.xclbin [reference.xclbin ...]]

Produces:
  - DM size per tile
  - Tile homogeneity check (are all tiles identical?)
  - Config word decode at key offsets (0x0090, 0x0094)
  - Byte-level diff between first and subsequent xclbins
  - Full hex dump of tile(0,2) DM
"""

import argparse
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axlf import load
from cdo  import parse_xclbin_cdo, CdoDmaWriteCmd, CdoWriteCmd, CdoMaskWriteCmd

CORE_DM_BASE = 0x20000

# ── helpers ───────────────────────────────────────────────────────────────

def collect_dm_all_tiles(path):
    """Return dict (col,row) -> bytes of DM content, populated from DmaWrite cmds."""
    ax  = load(path)
    cdo = parse_xclbin_cdo(ax.aie_pdi.data)
    tiles = {}
    for cmd in cdo.cmds:
        if not (isinstance(cmd, CdoDmaWriteCmd) and cmd.is_dm_init):
            continue
        key = (cmd.col, cmd.row)
        buf = tiles.setdefault(key, bytearray())
        off = cmd.local_off - CORE_DM_BASE
        raw = struct.pack(f"<{len(cmd.data)}I", *cmd.data)
        need = off + len(raw)
        if need > len(buf):
            buf.extend(b"\x00" * (need - len(buf)))
        buf[off : off + len(raw)] = raw
    return {k: bytes(v) for k, v in tiles.items()}


def hexdump(data, start=0, limit=None, as_u32=True):
    """Print a human-readable hex dump with optional uint32 annotation."""
    end = len(data) if limit is None else min(len(data), start + limit)
    for off in range(start, end, 16):
        chunk = data[off : off + 16]
        hex_  = " ".join(f"{b:02x}" for b in chunk)
        ann   = ""
        if as_u32:
            words = [struct.unpack_from("<I", chunk, i)[0]
                     for i in range(0, len(chunk), 4) if i + 4 <= len(chunk)]
            ann = "  " + " ".join(f"0x{w:08x}" for w in words)
        print(f"  {off:04x}: {hex_:<48}{ann}")


# ── analysis functions ────────────────────────────────────────────────────

def analyze(paths):
    if not paths:
        print("Usage: xclbin_dm_analyze.py <primary.xclbin> [ref1.xclbin ...]")
        sys.exit(1)

    primary_name = os.path.basename(paths[0])
    primary_tiles = collect_dm_all_tiles(paths[0])

    ref_tile = primary_tiles.get((0, 2), b"")
    tile_keys = sorted(primary_tiles.keys())

    print(f"\n{'='*72}")
    print(f"Primary: {primary_name}")
    print(f"{'='*72}")

    # ── 1. Tile list + sizes
    print(f"\n[1] Tiles with DM: {tile_keys}")
    sizes = set(len(v) for v in primary_tiles.values())
    print(f"    DM sizes: {sizes} bytes")

    # ── 2. Tile homogeneity
    all_same = all(bytes(v) == ref_tile for v in primary_tiles.values())
    print(f"\n[2] All {len(tile_keys)} tiles identical: {all_same}")
    if not all_same:
        print("    Differing tiles:")
        for key, dm in primary_tiles.items():
            if bytes(dm) != ref_tile:
                diffs = [i for i in range(min(len(dm), len(ref_tile))) if dm[i] != ref_tile[i]]
                print(f"      tile{key}: first diff at byte 0x{diffs[0]:04x}")

    # ── 3. Config word decode
    print(f"\n[3] Critical config words in tile(0,2) DM:")
    _config_decode(ref_tile, primary_name)

    # ── 4. Cross-xclbin diff
    if len(paths) > 1:
        print(f"\n[4] Byte-level diff vs. reference xclbins (tile 0,2, first 256 bytes):")
        for ref_path in paths[1:]:
            ref_name = os.path.basename(ref_path)
            ref_tiles = collect_dm_all_tiles(ref_path)
            ref_dm    = ref_tiles.get((0, 2), b"")
            diffs = [(i, ref_tile[i], ref_dm[i])
                     for i in range(min(256, len(ref_tile), len(ref_dm)))
                     if ref_tile[i] != ref_dm[i]]
            print(f"  vs {ref_name}: {len(diffs)} byte diffs in first 256B "
                  f"(first at 0x{diffs[0][0]:04x})" if diffs else
                  f"  vs {ref_name}: IDENTICAL in first 256B")
            for off, v1, v2 in diffs:
                print(f"      0x{off:04x}: {primary_name}=0x{v1:02x}  {ref_name}=0x{v2:02x}")

    # ── 5. Full DM dump of tile(0,2)
    print(f"\n[5] Full DM hex dump — tile(0,2), {len(ref_tile)} bytes:")
    hexdump(ref_tile, as_u32=True)


def _config_decode(dm, kernel_name):
    """Decode the two model-specific config words at DM 0x0090/0x0094."""
    if len(dm) < 0x9c:
        print("    DM too short for config decode.")
        return

    w90 = struct.unpack_from("<I", dm, 0x90)[0]
    w94 = struct.unpack_from("<I", dm, 0x94)[0]

    imm8     = (w90 >> 24) & 0xFF
    loop_cnt = w94 & 0xFFFF
    high94   = (w94 >> 16) & 0xFFFF

    COL_BLOCK = 128   # GPT-OSS col_block_size
    hidden_guess  = (loop_cnt + 1) * COL_BLOCK if loop_cnt > 0 else None
    ffn_guess     = hidden_guess

    print(f"    0x0090 = 0x{w90:08x}  imm8=0x{imm8:02x}={imm8}")
    print(f"    0x0094 = 0x{w94:08x}  loop_count={loop_cnt}  high=0x{high94:04x}")
    print(f"    Interpretation (col_block=128):")
    if loop_cnt == 4 and 'expert' in kernel_name.lower():
        print(f"      → expert kernel: loop_count=4 = NUM_CT_PER_COLUMN")
        print(f"        (4 compute tiles per column process experts in parallel)")
        print(f"        32 tiles × 1 expert/tile = 32 experts/hardware-pass")
        print(f"        With 128 experts: 4 hardware passes (matches loop_count!)")
    elif loop_cnt > 0:
        print(f"      → loop_count {loop_cnt}+1={loop_cnt+1} col-blocks × {COL_BLOCK}B = {hidden_guess} cols")
        print(f"        (likely hidden_size or intermediate_size)")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xclbins", nargs="+",
                        help="Primary xclbin first, then optional reference xclbins for diff")
    args = parser.parse_args()
    analyze(args.xclbins)
