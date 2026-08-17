#!/usr/bn/env python3
"""
xclbin_inspect.py — CLI tool: inspect an xclbin, extract sections, dump DMA topology

Usage:
  python xclbin_inspect.py <path/to/kernel.xclbin> [options]

Options:
  --sections        Print section table (default)
  --connectivity    Print kernel interface (name, args)
  --cdo-summary     Summarise CDO tile configuration
  --tile-programs   List extracted core-tile VLIW programs
  --dump-pm <dir>   Dump all tile program memory blobs to <dir>/
  --compare <b>     Compare two xclbins (diff section sizes and topology)
  --json            Output as JSON where applicable
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# allow running from the tools/ dir directly
sys.path.insert(0, str(Path(__file__).parent))
import axlf as axlf_mod
import cdo as cdo_mod
import txn as txn_mod


def cmd_inspect(args):
    xc = axlf_mod.load(args.xclbin)
    show_all = not any([args.connectivity, args.cdo_summary,
                        args.tile_programs, args.dump_pm])

    # ── section table ─────────────────────────────────────────
    if show_all or args.sections:
        print(xc.summary())
        print()

    # ── connectivity / kernel interface ───────────────────────
    if show_all or args.connectivity:
        print("── Kernel Interface ───────────────────────────────────")
        print(f"  Kernel name : {xc.kernel_name}")
        for a in xc.kernel_args:
            print(f"  arg[{a['id']:>2}]  {a['name']:<12}  {a['type']:<16}  "
                  f"size={a['size']}  addr_qualifier={a['addressQualifier']}")
        print()
        # Memory topology
        mem = xc.section(0x06)
        if mem:
            topo = mem.mem_topology
            if topo:
                print("── Memory Regions ─────────────────────────────────────")
                for t in topo:
                    print(f"  {t['name']:<8}  base=0x{t['base']:016x}  "
                          f"size={t['size_kb']} KB  used={t['used']}")
                print()

        # AIE partition descriptor
        aie = xc.section(0x19)
        if aie:
            info = aie.aie_partition_info
            if info:
                print("── AIE Partition Descriptor ───────────────────────────")
                print(f"  version     : {info['version']}")
                print(f"  num_configs : {info['num_configs']}")
                print(f"  cols        : {info['cols']}")
                print()

        if args.json:
            out = {
                "kernel_name": xc.kernel_name,
                "args": xc.kernel_args,
                "uuid": xc.uuid.hex(),
            }
            print(json.dumps(out, indent=2))

    # ── CDO / AIE tile configuration ──────────────────────────
    if show_all or args.cdo_summary or args.tile_programs or args.dump_pm:
        pdi_sec = xc.aie_pdi
        if pdi_sec is None:
            print("[!] No AIE partition PDI section found (kind=0x20).")
        else:
            parser = cdo_mod.parse_xclbin_cdo(pdi_sec.data)

            if show_all or args.cdo_summary:
                print("── CDO / AIE Hardware Configuration ──────────────────")
                print(parser.topology_summary())
                print()

            if show_all or args.tile_programs:
                progs = parser.extract_tile_programs()
                if progs:
                    print("── Extracted Core Tile VLIW Programs ─────────────────")
                    for p in progs:
                        print(f"  {p.describe()}")
                    print()
                else:
                    print("[i] No program-memory writes detected in CDO body.")
                    print("    (The CDO may use an indirect/PDI loading path.)")
                    print()

            if args.dump_pm:
                progs = parser.extract_tile_programs()
                out_dir = Path(args.dump_pm)
                out_dir.mkdir(parents=True, exist_ok=True)
                for p in progs:
                    name = f"tile_{p.col}_{p.row}_pm.bin"
                    dest = out_dir / name
                    p.save(str(dest))
                    print(f"  Saved {dest}  ({len(p.data)} bytes)")
                if not progs:
                    print("[!] Nothing to dump.")


def cmd_compare(args):
    a = axlf_mod.load(args.xclbin)
    b = axlf_mod.load(args.compare)

    print(f"── Section size diff: {Path(args.xclbin).name}  vs  {Path(args.compare).name}")
    print(f"{'Kind':<28}  {'A size':>12}  {'B size':>12}  {'Δ':>12}")
    print("-" * 70)

    a_map = {s.kind: s for s in a.sections}
    b_map = {s.kind: s for s in b.sections}
    all_kinds = sorted(set(a_map) | set(b_map))
    for k in all_kinds:
        name = axlf_mod.SECTION_KINDS.get(k, f"0x{k:02x}")
        sa = a_map[k].size if k in a_map else 0
        sb = b_map[k].size if k in b_map else 0
        delta = sb - sa
        sign = "+" if delta >= 0 else ""
        flag = "  ◄" if delta != 0 else ""
        print(f"{name:<28}  {sa:>12,}  {sb:>12,}  {sign}{delta:>11,}{flag}")
    print()

    # Compare kernel interface
    if a.kernel_name == b.kernel_name:
        print(f"[=] Kernel interface identical: {a.kernel_name}")
    else:
        print(f"[≠] Kernel A={a.kernel_name}  B={b.kernel_name}")

    # Quick CDO topology comparison
    for label, xc in [("A", a), ("B", b)]:
        pdi = xc.aie_pdi
        if pdi:
            parser = cdo_mod.parse_xclbin_cdo(pdi.data)
            progs = parser.extract_tile_programs()
            total_pm = sum(len(p.data) for p in progs)
            print(f"  [{label}] tiles with PM loads: {len(progs)}  "
                  f"total_pm_bytes={total_pm}")


def main():
    ap = argparse.ArgumentParser(description="NPU2 xclbin inspector")
    ap.add_argument("xclbin", help="Path to .xclbin file")
    ap.add_argument("--sections",      action="store_true")
    ap.add_argument("--connectivity",  action="store_true")
    ap.add_argument("--cdo-summary",   action="store_true", dest="cdo_summary")
    ap.add_argument("--tile-programs", action="store_true", dest="tile_programs")
    ap.add_argument("--dump-pm",       metavar="DIR",       dest="dump_pm")
    ap.add_argument("--compare",       metavar="XCLBIN_B",  dest="compare")
    ap.add_argument("--json",          action="store_true")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(args)
    else:
        cmd_inspect(args)


if __name__ == "__main__":
    main()
