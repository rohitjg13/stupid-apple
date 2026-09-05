"""Contract B: the event envelope. FROZEN — see docs/SHARED.md §5.

Payload keys are exact: no missing keys, no invented ones. If you need a new
field, change docs/SHARED.md §5 and this table first, in a PR Rohit reviews.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

STREAMS = ("overhead", "shelf", "-")

EVENT_TYPES = {
    "tripwire":            {"dir", "tripwire_id"},
    "occupancy":           {"count"},
    "visit":               {"track_id", "t_enter", "t_exit"},
    "heatmap":             {"t_bucket", "tiles"},
    "shelf_fill":          {"fill"},
    "stockout_start":      {"sku", "facing"},
    "stockout_end":        {"sku", "facing"},
    "planogram_violation": {"expected_sku", "observed"},
    "lane_occ":            {"lane", "cells"},
    "queue_estimate":      {"lane", "count", "pred_wait_s"},
    "pos_txn":             {"lane", "items", "amount", "zone_hint"},
    "alert":               {"severity", "rule", "message"},
}


@dataclass
class Event:
    t: float                    # unix epoch seconds
    store_id: str
    stream: str                 # "overhead" | "shelf" | "-"
    zone_id: Optional[str]      # zone / roi / lane id
    event_type: str
    payload: dict

    def __post_init__(self):
        expected = EVENT_TYPES.get(self.event_type)
        if expected is None:
            raise ValueError(f"unknown event_type {self.event_type!r}; see docs/SHARED.md §5")
        if self.stream not in STREAMS:
            raise ValueError(f"unknown stream {self.stream!r}, expected one of {STREAMS}")
        got = set(self.payload)
        if got != expected:
            raise ValueError(
                f"{self.event_type} payload mismatch: missing {sorted(expected - got)}, "
                f"unexpected {sorted(got - expected)}"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self))
