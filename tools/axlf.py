#!/usr/bin/env python3
"""
axlf.py — AMD/Xilinx AXLF (xclbin2) container parser

The AXLF format is documented in XRT's xclbin.h.
This module provides read-only parsing; xclbinutil (part of XRT) handles
creation/modification.

Reference: https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/include/xclbin.h
"""

from __future__ import annotations
import struct
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import xml.etree.ElementTree as ET

# ── Section kind IDs ────────────────────────────────────────────────────────
SECTION_KINDS: dict[int, str] = {
    0x00: "BITSTREAM",
    0x01: "CLEARING_BITSTREAM",
    0x02: "EMBEDDED_METADATA",      # CONNECTIVITY XML lives here
    0x03: "FIRMWARE",
    0x04: "DEBUG_DATA",
    0x05: "SCHED_FIRMWARE",
    0x06: "MEM_TOPOLOGY",
    0x07: "IP_LAYOUT",
    0x08: "DEBUG_IP_LAYOUT",
    0x09: "DESIGN_CHECK_POINT",
    0x0A: "PDI",
    0x0B: "BITSTREAM_PARTIAL_PDI",
    0x0C: "DTC",
    0x0D: "EMULATION_DATA",
    0x0E: "SYSTEM_METADATA",
    0x0F: "CLOCK_FREQ_TOPOLOGY",
    0x10: "MCSBIN",
    0x11: "BME_SUPPORT",
    0x12: "PSP_IMAGES",
    0x13: "HMAC",
    0x14: "PARTITION_METADATA",
    0x15: "EXE_AFFINITY_MAP",
    0x16: "SYSTEM_DT",
    0x17: "SOFT_KERNEL",            # In NPU xclbins: memory topology
    0x18: "ASK_FLASH",
    0x19: "AIE_PARTITION",          # Partition descriptor (col/row map)
    0x1A: "ASK_GROUP_TOPOLOGY",
    0x1B: "ASK_GROUP_CONNECTIVITY",
    0x1C: "AIEBU",                  # AIE Binary Utilities payload
    0x1D: "DEBUG_IP_LAYOUT_EXTENDED",
    0x20: "AIE_PARTITION_PDI",      # The big CDO/PDI blob for NPU2
    # Aliases/alternative values observed in NPU2 xclbins
    0x02: "CONNECTIVITY",           # Same slot re-used for XML connectivity
}

# ── AXLF header layout ───────────────────────────────────────────────────────
# struct axlf {
#   char     magic[8];           // "xclbin2\0"
#   int32_t  signature_length;   // -1 if no signature
#   uint8_t  reserved[28];
#   uint8_t  keyBlock[256];
#   uint64_t uniqueId;
#   axlf_header header;
# };
#
# struct axlf_header {
#   uint64_t timeStamp;
#   uint64_t featureRomTimeStamp;
#   uint16_t versionPatch;
#   uint8_t  versionMajor;
#   uint8_t  versionMinor;
#   uint32_t mode;
#   union { uint64_t platformId; ... } rom;
#   uint8_t  platformVBNV[64];
#   uint8_t  actionMask;
#   uint8_t  uuid[16];
#   char     debug_bin[16];
#   uint32_t numSections;
#   // section table follows immediately
# };
#
# Each section entry (axlf_section_header):
#   uint32_t sectionKind;
#   uint32_t flags;
#   uint64_t sectionOffset;  (bytes from start of file)
#   uint64_t sectionSize;    (bytes)
#   char     sectionName[16];

AXLF_MAGIC = b"xclbin2\x00"

# Official layout from XRT xclbin.h  (40-byte struct with alignment padding):
#   [+0x00]  uint32   m_sectionKind
#   [+0x04]  char[16] m_sectionName
#   [+0x14]  uint8[4] (implicit compiler padding for uint64 alignment)
#   [+0x18]  uint64   m_sectionOffset  (from start of file)
#   [+0x20]  uint64   m_sectionSize    (bytes)
# Total: 40 bytes per entry.
AXLF_SECTION_ENTRY_SIZE = 40

AXLF_NSECTIONS_OFFSET     = 0x1C0   # offset of uint32 section count
AXLF_SECTION_TABLE_OFFSET = 0x1C8   # section entries start here


@dataclass
class AxlfSection:
    kind: int
    kind_name: str
    offset: int       # bytes from file start
    size: int         # bytes
    name: str         # embedded name string from header entry
    data: bytes = field(repr=False, default=b"")

    @property
    def is_aie_partition(self) -> bool:
        return self.kind in (0x19, 0x20)

    @property
    def is_connectivity(self) -> bool:
        # The CONNECTIVITY section contains the kernel interface XML
        return self.kind == 0x02

    @property
    def connectivity_xml(self) -> Optional[ET.Element]:
        if not self.is_connectivity:
            return None
        try:
            return ET.fromstring(self.data.rstrip(b"\x00"))
        except ET.ParseError:
            return None

    @property
    def mem_topology(self) -> Optional[list[dict]]:
        """Parse MEM_TOPOLOGY binary section."""
        if self.kind != 0x06:
            return None
        if len(self.data) < 8:
            return None
        count = struct.unpack_from("<I", self.data, 4)[0]
        entries = []
        base = 8
        entry_size = 32
        for i in range(count):
            off = base + i * entry_size
            if off + entry_size > len(self.data):
                break
            mtype, used = struct.unpack_from("<BB", self.data, off)
            name = self.data[off + 4: off + 4 + 16].split(b"\x00")[0].decode("ascii", errors="replace")
            base_addr = struct.unpack_from("<Q", self.data, off + 20)[0]
            size_kb = struct.unpack_from("<Q", self.data, off + 28)[0]
            entries.append({"type": mtype, "used": used, "name": name,
                             "base": base_addr, "size_kb": size_kb})
        return entries

    @property
    def aie_partition_info(self) -> Optional[dict]:
        """Parse AIE_PARTITION binary descriptor (kind=0x19)."""
        if self.kind != 0x19:
            return None
        if len(self.data) < 12:
            return None
        # First 12 bytes: version(4), num_configs(4), reserved(4)
        version = struct.unpack_from("<I", self.data, 0)[0]
        num = struct.unpack_from("<I", self.data, 4)[0]
        # Each 12-byte entry: col(4), num_rows(4), reserved(4)
        cols = []
        for i in range(num):
            off = 12 + i * 12
            if off + 4 > len(self.data):
                break
            col = struct.unpack_from("<I", self.data, off)[0]
            cols.append(col)
        return {"version": version, "num_configs": num, "cols": cols}


@dataclass
class Axlf:
    path: Path
    uuid: bytes
    num_sections: int
    sections: list[AxlfSection]

    def section(self, kind: int) -> Optional[AxlfSection]:
        for s in self.sections:
            if s.kind == kind:
                return s
        return None

    def section_by_name(self, name: str) -> Optional[AxlfSection]:
        for s in self.sections:
            if s.kind_name == name or s.name == name:
                return s
        return None

    @property
    def aie_pdi(self) -> Optional[AxlfSection]:
        """Returns the big AIE partition PDI section (kind=0x20)."""
        return self.section(0x20)

    @property
    def kernel_name(self) -> Optional[str]:
        """Extract the MLIR_AIE kernel name from CONNECTIVITY XML."""
        conn = self.section(0x02)
        if conn is None:
            return None
        xml = conn.connectivity_xml
        if xml is None:
            return None
        for kernel in xml.iter("kernel"):
            return kernel.get("name")
        return None

    @property
    def kernel_args(self) -> list[dict]:
        """Return ordered list of kernel argument descriptors."""
        conn = self.section(0x02)
        if conn is None:
            return []
        xml = conn.connectivity_xml
        if xml is None:
            return []
        args = []
        for arg in xml.iter("arg"):
            args.append({k: arg.get(k) for k in ("name", "id", "type", "size", "offset",
                                                   "addressQualifier")})
        return args

    def summary(self) -> str:
        lines = [
            f"File   : {self.path}",
            f"UUID   : {self.uuid.hex()}",
            f"Kernel : {self.kernel_name}",
            f"Args   : {[a['name'] for a in self.kernel_args]}",
            "",
            f"{'#':<3}  {'Kind':<28}  {'Name':<20}  {'Offset':>10}  {'Size':>10}",
            "-" * 80,
        ]
        for i, s in enumerate(self.sections):
            lines.append(
                f"{i:<3}  {s.kind_name:<28}  {s.name:<20}  "
                f"0x{s.offset:08x}  {s.size:>10,}"
            )
        return "\n".join(lines)


def load(path: str | Path) -> Axlf:
    """Parse an AXLF/xclbin file and return an Axlf object with all section data."""
    path = Path(path)
    data = path.read_bytes()

    if data[:8] != AXLF_MAGIC:
        raise ValueError(f"Not an AXLF file (magic mismatch): {path}")

    num_sections = struct.unpack_from("<I", data, AXLF_NSECTIONS_OFFSET)[0]

    # UUID is at a fixed offset: after magic+sig+reserved+keyBlock+uniqueId+timestamps...
    # Empirically confirmed at 0x1a0 for NPU2 xclbins
    UUID_OFFSET = 0x1A0
    uuid = data[UUID_OFFSET: UUID_OFFSET + 16]

    sections: list[AxlfSection] = []
    table_base = AXLF_SECTION_TABLE_OFFSET
    for i in range(num_sections):
        entry_off = table_base + i * AXLF_SECTION_ENTRY_SIZE
        if entry_off + AXLF_SECTION_ENTRY_SIZE > len(data):
            break
        kind       = struct.unpack_from("<I", data, entry_off)[0]
        name_raw   = data[entry_off + 4: entry_off + 20]          # 16-byte name
        # +20 = 4 bytes implicit alignment padding (skipped)
        sec_offset = struct.unpack_from("<Q", data, entry_off + 24)[0]
        sec_size   = struct.unpack_from("<Q", data, entry_off + 32)[0]
        name = name_raw.split(b"\x00")[0].decode("ascii", errors="replace")

        sec_data = data[sec_offset: sec_offset + sec_size] if sec_size else b""
        kind_name = SECTION_KINDS.get(kind, f"UNKNOWN_0x{kind:02x}")

        sections.append(AxlfSection(
            kind=kind,
            kind_name=kind_name,
            offset=sec_offset,
            size=sec_size,
            name=name,
            data=sec_data,
        ))

    return Axlf(path=path, uuid=uuid, num_sections=num_sections, sections=sections)
