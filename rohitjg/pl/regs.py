"""AXI-Lite register map mirror of hls/regs.h. FROZEN — see docs/SHARED.md §4."""
from __future__ import annotations

import struct
from enum import IntEnum

MAX_ROIS, MAX_LANES = 64, 16
PL_W, PL_H = 320, 240               # tables are authored at PL scale, not 640x480
ROI_TABLE_BYTES = MAX_ROIS * 8      # {x,y,w,h} u16, 320x240 scale
LANE_TABLE_BYTES = MAX_LANES * 8


class Reg(IntEnum):
    CTRL = 0x00                     # bit0 start, bit1 reset bg, bit2 dump mask, bit3 result+mask
    STREAM_ID = 0x04                # 0 overhead, 1 shelf
    MOG2_VAR_THRESH = 0x08          # Q8.8, default 16.0
    MOG2_LR = 0x0C                  # Q0.16, default 0.005
    MIN_BLOB_AREA = 0x10            # at 320x240, default 150
    MORPH_ITERS = 0x14              # 0..3
    LATENCY = 0x18                  # read-only
    DEBUG_STAGE = 0x1C              # 0 gray, 1 mask, 2 morph
    ROI_TABLE = 0x100               # 0x100-0x2FF, 512 bytes
    LANE_TABLE = 0x300              # 0x300-0x37F, 128 bytes


def q8_8(v: float) -> int:
    if not 0.0 <= v < 256.0:
        raise ValueError(f"{v} out of Q8.8 range")
    return round(v * 256)


def q0_16(v: float) -> int:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{v} out of Q0.16 range")
    return min(round(v * 65536), 0xFFFF)    # ponytail: saturate; 1.0 has no exact Q0.16


def _pack(entries, limit, nbytes, what) -> bytes:
    if len(entries) > limit:
        raise ValueError(f"{len(entries)} {what} exceeds hardware table of {limit}")
    out = []
    for e in entries:
        x, y, w, h = (int(e[k]) for k in ("x", "y", "w", "h"))
        # Trust boundary: these come from hand-edited JSON. Authoring at 640x480
        # instead of PL scale reads the wrong shelf and looks like a CV bug.
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > PL_W or y + h > PL_H:
            raise ValueError(
                f"{what} {e.get('id', '?')} = ({x},{y},{w},{h}) is outside the "
                f"{PL_W}x{PL_H} PL frame; author tables at PL scale"
            )
        out.append(struct.pack("<4H", x, y, w, h))
    return b"".join(out).ljust(nbytes, b"\x00")


def pack_roi_table(rois) -> bytes:
    """rois.json order == roi_fill[] index. Adding a facing never needs a rebuild."""
    return _pack(rois, MAX_ROIS, ROI_TABLE_BYTES, "ROIs")


def pack_lane_table(lanes) -> bytes:
    """lanes.json order == lane_occupancy[] index."""
    return _pack(lanes, MAX_LANES, LANE_TABLE_BYTES, "lane cells")
