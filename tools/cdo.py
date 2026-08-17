#!/usr/bin/env python3
"""
cdo.py — AMD AIE2 CDO (Configuration Data Object) parser

The AIE partition section (kind=0x20) inside an NPU2 xclbin holds the
static hardware-configuration CDO.  It programmes every AIE tile before
any inference runs: DMA channels, locks, stream-switch connections, and —
most importantly — the VLIW program memory of each compute core.

CDO v2 format  (confirmed from XRT cdo_cmd.h in aie-pdi-transform)
─────────────────────────────────────────────────────────────────────────
The AIE partition section starts with a ~540-byte wrapper; the CDO
proper begins at CDO_HEADER_OFFSET = 0x21C from the section start:

  [0x21C] framing marker  0xfdfb4175
  [0x220] framing count   0x00000004
  [0x224] CDO magic       0x004F4443  ("CDO\\0")
  [0x228] version         0x00000200  (v2.0)
  [0x22C] total_len_words
  [0x230] checksum
  [0x234] command stream starts here  (CDO_BODY_OFFSET)

CDO v2 command word (little-endian):
  bits[ 7: 0] = opcode
  bits[15: 8] = module_id
  bits[23:16] = payload_length  (words after this header)
                If == 0xFF → long command: next word = real payload length
  bits[31:24] = reserved

Opcodes (from cdo_cmd.h XCDO_CMD_*):
  0x02  MASK_WRITE    payload: [addr, mask, value]
  0x03  WRITE         payload: [addr, value]
  0x05  DMAWRITE      payload: [hi_addr, lo_addr, data*N]
  0x07  MASKWRITE64   payload: [hi_addr, lo_addr, mask, value]
  0x08  WRITE64       payload: [hi_addr, lo_addr, value]
  0x11  NOP           payload: (none)
  0x01FF END marker   (entire command word value)

AIE2 tile address (NPU2 / XDNA / Phoenix):
  addr = (col << 25) | (row << 20) | local_offset
  row 0 = shim, row 1 = memory tile, rows 2-5 = compute cores
"""

from __future__ import annotations
import struct
from dataclasses import dataclass, field


# ── AIE2 address constants ────────────────────────────────────────────────────

AIE2_COL_SHIFT = 25
AIE2_ROW_SHIFT = 20

# Core tile (rows 2-5) local address regions
CORE_REG_BASE  = 0x00000   # stream switch, locks, etc.
CORE_REG_END   = 0x10000
CORE_PM_BASE   = 0x10000   # program memory (VLIW code, if loaded statically)
CORE_PM_END    = 0x1D000
CORE_DMA_BD_BASE = 0x1D000  # core-tile DMA buffer descriptors
CORE_DMA_BD_END  = 0x20000
CORE_DM_BASE   = 0x20000   # core tile data memory
CORE_DM_END    = 0x30000

# Memory tile (row 1) local address regions
MEM_DM_BASE    = 0x00000   # 512 KB SRAM
MEM_DM_END     = 0x80000
MEM_DMA_BD_BASE = 0xA0000  # memory-tile DMA buffer descriptors
MEM_DMA_BD_END  = 0xB0000

SHIM_DMA_BASE  = 0x1D000   # shim tile (row 0) DMA registers

SHIM_ROW     = 0
MEM_ROW      = 1
CORE_ROW_MIN = 2
CORE_ROW_MAX = 5


def decode_addr(addr: int) -> tuple:
    """Return (col, row, local_offset) from a 32-bit NPU2 AIE tile address."""
    col    = (addr >> AIE2_COL_SHIFT) & 0x7F
    row    = (addr >> AIE2_ROW_SHIFT) & 0x1F
    offset = addr & ((1 << AIE2_ROW_SHIFT) - 1)
    return col, row, offset


def area_name(row: int, off: int) -> str:
    if row == SHIM_ROW:
        return "SHIM_DMA" if SHIM_DMA_BASE <= off < SHIM_DMA_BASE + 0x3000 else "SHIM"
    if row == MEM_ROW:
        if MEM_DM_BASE <= off < MEM_DM_END:   return "MEM_DM"
        if MEM_DMA_BD_BASE <= off < MEM_DMA_BD_END: return "MEM_DMA_BD"
        return "MEM_REG"
    if CORE_ROW_MIN <= row <= CORE_ROW_MAX:
        if CORE_PM_BASE <= off < CORE_PM_END:     return "CORE_PM"
        if CORE_DMA_BD_BASE <= off < CORE_DMA_BD_END: return "CORE_DMA_BD"
        if CORE_DM_BASE <= off < CORE_DM_END:     return "CORE_DM"
        return "CORE_REG"
    return f"ROW{row}_0x{off:05x}"


# ── CDO constants ─────────────────────────────────────────────────────────────

CDO_MAGIC         = 0x004F4443
CDO_HEADER_OFFSET = 0x21C   # framing words start here in AIE partition section
CDO_BODY_OFFSET   = 0x234   # first command word (= 0x21C + 6*4)

CDO_OP_MASK_WRITE  = 0x02
CDO_OP_WRITE       = 0x03
CDO_OP_DMAWRITE    = 0x05
CDO_OP_MASKWRITE64 = 0x07
CDO_OP_WRITE64     = 0x08
CDO_OP_NOP         = 0x11
CDO_OP_END_WORD    = 0x01FF  # full command word value for END marker


# ── Command dataclasses ───────────────────────────────────────────────────────

@dataclass
class CdoCmd:
    body_off: int
    opcode: int
    module_id: int
    payload: list = field(repr=False, default_factory=list)

    def describe(self) -> str:
        return f"[+0x{self.body_off:06x}] CDO op=0x{self.opcode:02x} mod=0x{self.module_id:02x}"


@dataclass
class CdoWriteCmd(CdoCmd):
    addr: int = 0
    value: int = 0
    col: int = 0
    row: int = 0
    local_off: int = 0

    def describe(self) -> str:
        a = area_name(self.row, self.local_off)
        return (f"[+0x{self.body_off:06x}] WRITE"
                f"  tile({self.col},{self.row})"
                f"  reg=0x{self.local_off:05x}({a})"
                f"  val=0x{self.value:08x}")


@dataclass
class CdoMaskWriteCmd(CdoCmd):
    addr: int = 0
    mask: int = 0
    value: int = 0
    col: int = 0
    row: int = 0
    local_off: int = 0

    def describe(self) -> str:
        a = area_name(self.row, self.local_off)
        return (f"[+0x{self.body_off:06x}] MASKWRITE"
                f"  tile({self.col},{self.row})"
                f"  reg=0x{self.local_off:05x}({a})"
                f"  mask=0x{self.mask:08x}  val=0x{self.value:08x}")


@dataclass
class CdoDmaWriteCmd(CdoCmd):
    dest_hi: int = 0
    dest_lo: int = 0
    col: int = 0
    row: int = 0
    local_off: int = 0
    data: list = field(repr=False, default_factory=list)

    @property
    def dest_addr(self) -> int:
        return (self.dest_hi << 32) | self.dest_lo

    @property
    def is_core_dma_bd(self) -> bool:
        """Configures a core-tile (rows 2-5) DMA buffer descriptor."""
        return (CORE_ROW_MIN <= self.row <= CORE_ROW_MAX
                and CORE_DMA_BD_BASE <= self.local_off < CORE_DMA_BD_END)

    @property
    def is_pm_load(self) -> bool:
        """Loads code into core-tile program memory (rare; usually done via DPU instr buffer)."""
        return (CORE_ROW_MIN <= self.row <= CORE_ROW_MAX
                and CORE_PM_BASE <= self.local_off < CORE_PM_END)

    @property
    def is_dm_init(self) -> bool:
        """Loads static weights/constants into core-tile data memory."""
        return CORE_DM_BASE <= self.local_off < CORE_DM_END and CORE_ROW_MIN <= self.row

    @property
    def is_mem_dma_bd(self) -> bool:
        """Configures a memory-tile DMA buffer descriptor."""
        return self.row == MEM_ROW and MEM_DMA_BD_BASE <= self.local_off < MEM_DMA_BD_END

    def describe(self) -> str:
        a = area_name(self.row, self.local_off)
        note = " [PM_LOAD]" if self.is_pm_load else (
               " [CORE_DMA_BD]" if self.is_core_dma_bd else (
               " [DM_INIT]" if self.is_dm_init else (
               " [MEM_DMA_BD]" if self.is_mem_dma_bd else "")))
        return (f"[+0x{self.body_off:06x}] DMAWRITE"
                f"  tile({self.col},{self.row})"
                f"  reg=0x{self.local_off:05x}({a}){note}"
                f"  words={len(self.data)}")


@dataclass
class TileProgram:
    col: int
    row: int
    base_reg: int
    data: bytes

    @property
    def num_vliw_instrs(self) -> int:
        """AIE2 VLIW instruction = 128 bits = 16 bytes."""
        return len(self.data) // 16

    def describe(self) -> str:
        return (f"tile({self.col},{self.row})"
                f"  pm_offset=0x{self.base_reg:05x}"
                f"  {len(self.data)} bytes"
                f"  ({self.num_vliw_instrs} VLIW instrs)")

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.data)


# ── CDO parser ────────────────────────────────────────────────────────────────

class CdoParser:
    """
    Parses CDO v2 commands from the raw bytes of an AIE partition section.

        p = CdoParser(section_data)
        p.parse()
        programs = p.extract_tile_programs()
        print(p.topology_summary())
    """

    def __init__(self, section_data: bytes):
        self.data = section_data
        self.cmds: list = []

    def parse(self) -> "CdoParser":
        data = self.data

        # Locate CDO body: default at CDO_BODY_OFFSET, scan on mismatch
        body_start = CDO_BODY_OFFSET
        if len(data) >= CDO_HEADER_OFFSET + 12:
            magic = struct.unpack_from("<I", data, CDO_HEADER_OFFSET + 8)[0]
            if magic != CDO_MAGIC:
                # Scan for magic and skip 4 header words after it
                for i in range(0, min(len(data) - 4, CDO_HEADER_OFFSET + 0x80), 4):
                    if struct.unpack_from("<I", data, i)[0] == CDO_MAGIC:
                        body_start = i + 16
                        break

        n = (len(data) - body_start) // 4
        if n <= 0:
            return self
        words = list(struct.unpack_from(f"<{n}I", data, body_start))

        pos = 0
        while pos < len(words):
            body_off = pos * 4
            w0 = words[pos]

            if (w0 & 0xFFFF) == CDO_OP_END_WORD:
                break

            opcode    = w0 & 0xFF
            module_id = (w0 >> 8) & 0xFF
            plen      = (w0 >> 16) & 0xFF

            if plen == 0xFF:  # long command
                if pos + 1 >= len(words):
                    break
                plen = words[pos + 1]
                data_pos = pos + 2
            else:
                data_pos = pos + 1

            if data_pos + plen > len(words):
                break
            payload = words[data_pos: data_pos + plen]
            advance = data_pos + plen

            if opcode == CDO_OP_NOP:
                self.cmds.append(CdoCmd(body_off, opcode, module_id, []))

            elif opcode == CDO_OP_WRITE and len(payload) >= 2:
                addr, val = payload[0], payload[1]
                col, row, off = decode_addr(addr)
                self.cmds.append(CdoWriteCmd(body_off, opcode, module_id,
                                              payload, addr, val, col, row, off))

            elif opcode == CDO_OP_MASK_WRITE and len(payload) >= 3:
                addr, mask, val = payload[0], payload[1], payload[2]
                col, row, off = decode_addr(addr)
                self.cmds.append(CdoMaskWriteCmd(body_off, opcode, module_id,
                                                   payload, addr, mask, val,
                                                   col, row, off))

            elif opcode == CDO_OP_DMAWRITE and len(payload) >= 2:
                hi, lo = payload[0], payload[1]
                col, row, off = decode_addr(lo)
                self.cmds.append(CdoDmaWriteCmd(body_off, opcode, module_id,
                                                  payload, hi, lo, col, row, off,
                                                  payload[2:]))

            elif opcode in (CDO_OP_WRITE64, CDO_OP_MASKWRITE64) and len(payload) >= 3:
                hi, lo = payload[0], payload[1]
                col, row, off = decode_addr(lo)
                val = payload[2]
                self.cmds.append(CdoWriteCmd(body_off, opcode, module_id,
                                              payload, (hi << 32) | lo, val,
                                              col, row, off))

            else:
                self.cmds.append(CdoCmd(body_off, opcode, module_id, payload))

            pos = advance

        return self

    def extract_tile_programs(self) -> list:
        """Collect DMAWRITE commands targeting core-tile program memory (0x10000-0x1CFFF).
        Note: In practice, VLIW programs are usually loaded via the DPU runtime instr
        buffer, not the static CDO. Core-tile DMA BDs (0x1D000+) are separate from PM."""
        pm_segs: dict = {}
        for cmd in self.cmds:
            if isinstance(cmd, CdoDmaWriteCmd) and cmd.is_pm_load:
                key = (cmd.col, cmd.row)
                pm_segs.setdefault(key, []).append((cmd.local_off, cmd.data))

        programs = []
        for (col, row), segs in sorted(pm_segs.items()):
            segs.sort(key=lambda x: x[0])
            base = segs[0][0]
            all_words: list = []
            for _, d in segs:
                all_words.extend(d)
            raw = struct.pack(f"<{len(all_words)}I", *all_words)
            programs.append(TileProgram(col=col, row=row, base_reg=base, data=raw))
        return programs

    def extract_static_weights(self) -> dict:
        """Extract all static data written to core-tile data memory (DM, 0x20000+).
        Returns dict: (col, row) -> list of (local_offset, bytes)
        These are the model weights/constants baked into the xclbin."""
        result: dict = {}
        for cmd in self.cmds:
            if isinstance(cmd, CdoDmaWriteCmd) and cmd.is_dm_init:
                key = (cmd.col, cmd.row)
                data_bytes = struct.pack(f"<{len(cmd.data)}I", *cmd.data)
                result.setdefault(key, []).append((cmd.local_off, data_bytes))
        return result

    def topology_summary(self) -> str:
        tiles: set = set()
        shim_dma = 0
        pm_bytes: dict = {}
        counts: dict = {}

        dm_bytes: dict = {}
        dma_bd_tiles: set = set()
        for cmd in self.cmds:
            n = type(cmd).__name__
            counts[n] = counts.get(n, 0) + 1
            if hasattr(cmd, 'col'):
                tiles.add((cmd.col, cmd.row))
            if isinstance(cmd, CdoDmaWriteCmd):
                if cmd.is_pm_load:
                    k = (cmd.col, cmd.row)
                    pm_bytes[k] = pm_bytes.get(k, 0) + len(cmd.data) * 4
                if cmd.is_dm_init:
                    k = (cmd.col, cmd.row)
                    dm_bytes[k] = dm_bytes.get(k, 0) + len(cmd.data) * 4
                if cmd.is_core_dma_bd or cmd.is_mem_dma_bd:
                    dma_bd_tiles.add((cmd.col, cmd.row))
            if isinstance(cmd, (CdoWriteCmd, CdoMaskWriteCmd)):
                if cmd.row == SHIM_ROW and SHIM_DMA_BASE <= cmd.local_off < SHIM_DMA_BASE + 0x3000:
                    shim_dma += 1

        lines = [
            "── AIE Tile Configuration ──────────────────────────────────",
            f"  Total CDO commands : {len(self.cmds)}",
            f"  Tiles touched      : {len(tiles)}",
            f"  Shim DMA reg writes: {shim_dma}",
            f"  Tiles with DMA BDs : {len(dma_bd_tiles)}",
            "",
            "  Static weights in core-tile DM (0x20000+):",
        ]
        for (col, row), nb in sorted(dm_bytes.items()):
            lines.append(f"    tile({col},{row})  {nb} bytes")
        if not dm_bytes:
            lines.append("    (none found)")
        lines += ["", "  Program memory loads (0x10000-0x1CFFF, static VLIW code):"]
        for (col, row), nb in sorted(pm_bytes.items()):
            lines.append(f"    tile({col},{row})  {nb} bytes  ({nb//24} VLIW instrs @ 24B each)")
        if not pm_bytes:
            lines.append("    (none — VLIW programs loaded at runtime via DPU instr buffer)")

        lines += ["", "  CDO command breakdown:"]
        for n, c in sorted(counts.items()):
            lines.append(f"    {n:<24} {c}")
        return "\n".join(lines)


# ── public helper ─────────────────────────────────────────────────────────────

def parse_xclbin_cdo(aie_section_data: bytes) -> CdoParser:
    """
    Parse the CDO from the raw bytes of an AIE partition section
    (section kind=0x20 returned by axlf.load().aie_pdi.data).
    """
    return CdoParser(aie_section_data).parse()
