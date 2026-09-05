"""Tuning knobs for the shopper pipeline, from `<clipset>/tracker.yaml`.

Loaded here rather than in core/config.py on purpose: `load_clipset` validates a
fixed set of files and adding to it would mean changing Rohit's schema and his
tests for parameters only this package reads. An absent tracker.yaml is fine --
the defaults below are the tuned values for config/sim.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# pl/regs.py defaults MIN_BLOB_AREA to 150 *at 320x240*. pl/reference.py reports
# `area * SCALE * SCALE` with SCALE=2, so the same blob arrives here as 600.
MIN_BLOB_AREA_FULL_RES = 150 * 4


@dataclass
class TrackerParams:
    # --- association ---
    n_init: int = 3                     # hits before a track is confirmed
    max_age: int = 30                   # frames a lost track is predicted forward
    iou_high: float = 0.5               # first pass, big blobs
    iou_low: float = 0.3                # second pass, everything else
    high_area: int = 2 * MIN_BLOB_AREA_FULL_RES   # area is the pseudo-score
    overlap_grace: int = 15             # frames two tracks may share one blob

    # --- prefilter ---
    merge_gap_px: int = 8               # vertical gap that still means one person
    merge_x_tol: float = 0.5            # horizontal overlap needed to merge, 0..1
    max_aspect: float = 3.0             # wider than this is a shadow, not a person

    # --- events ---
    visit_debounce_s: float = 1.0       # continuous presence before a zone counts
    min_visit_s: float = 2.0            # shorter than this is pass-through
    tripwire_debounce_s: float = 2.0    # one crossing per track per this long
    occupancy_period_s: float = 1.0
    heatmap_period_s: float = 10.0
    heatmap_decay_per_s: float = 0.98   # live tile only; the event is undecayed
    entrance_zone: str = "entrance"     # excluded from the occupancy count

    # --- guards ---
    flood_area_frac: float = 0.40       # blob area over this fraction = bad frame
    flood_pause_s: float = 2.0

    @classmethod
    def load(cls, clipset=None) -> "TrackerParams":
        """Read `<clipset>/tracker.yaml` if it exists, else use the defaults."""
        if clipset is None:
            return cls()
        p = Path(clipset) / "tracker.yaml"
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p}: tracker.yaml must be a mapping, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            # A typo'd key silently doing nothing is how you lose an evening.
            raise ValueError(f"{p}: unknown tracker.yaml keys {sorted(unknown)}; "
                             f"expected some of {sorted(known)}")
        log.info("tracker params loaded", extra={"path": str(p), "overrides": sorted(data)})
        return cls(**data)
