"""Contract A: FrameResult, PL -> PS, 856 bytes. FROZEN — see docs/SHARED.md §4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

BLOB_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("w", "<u2"), ("h", "<u2"), ("area", "<u4")])
FRAME_RESULT_DT = np.dtype([
    ("frame_id", "<u4"), ("num_blobs", "<u2"), ("flags", "<u2"),
    ("blobs", BLOB_DT, (64,)), ("roi_fill", "u1", (64,)), ("lane_occupancy", "u1", (16,)),
])
assert FRAME_RESULT_DT.itemsize == 856

MAX_BLOBS = 64
FLAG_BG_WARM = 1 << 0
FLAG_TRUNCATED = 1 << 1
STREAMS = ("overhead", "shelf")


def blob_foot_point(blob) -> tuple:
    """Ground contact of a blob in full-res image px. docs/SHARED.md §6."""
    return (int(blob["x"]) + int(blob["w"]) // 2, int(blob["y"]) + int(blob["h"]))


@dataclass
class Frame:
    stream: str
    frame_id: int
    t_ns: int              # time.monotonic_ns() from the capture thread
    result: Any            # np.void of FRAME_RESULT_DT
    image: Optional[Any] = None   # never persisted, never serialised

    def __post_init__(self):
        if self.stream not in STREAMS:
            raise ValueError(f"unknown stream {self.stream!r}, expected one of {STREAMS}")
        if int(self.result["frame_id"]) != self.frame_id:
            raise ValueError(f"frame_id {self.frame_id} does not echo PL {int(self.result['frame_id'])}")

    @property
    def blobs(self):
        return self.result["blobs"][: int(self.result["num_blobs"])]

    def to_dict(self) -> dict:
        return {
            "stream": self.stream,
            "frame_id": int(self.frame_id),
            "t_ns": int(self.t_ns),
            "flags": int(self.result["flags"]),
            "blobs": [{k: int(b[k]) for k in ("x", "y", "w", "h", "area")} for b in self.blobs],
            "roi_fill": self.result["roi_fill"].tolist(),
            "lane_occupancy": self.result["lane_occupancy"].tolist(),
        }
