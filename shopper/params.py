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
    # Tuned on config/sim, where nobody ever fragments, so every merge there is a
    # false one: 0.95 cuts false merges from 1118 frames in 3900 to 192. Fusing
    # two shoppers loses one from the count permanently, while splitting one
    # person is something the tracker recovers from, so this stays conservative.
    merge_x_tol: float = 0.95           # horizontal overlap needed to merge, 0..1
    max_v_overlap_frac: float = 0.25    # above this they are two people, not two
                                        # fragments of one; see blob_prefilter
    max_aspect: float = 3.0             # wider than this is a shadow, not a person

    # --- events ---
    visit_debounce_s: float = 1.0       # continuous presence before a zone counts
    min_visit_s: float = 2.0            # shorter than this is pass-through
    tripwire_debounce_s: float = 2.0    # one crossing per track per this long
    tripwire_deadband_px: float = 12.0  # hysteresis band around the wire; a track
                                        # must clear it before a crossing counts
    occupancy_period_s: float = 1.0
    heatmap_period_s: float = 10.0
    heatmap_decay_per_s: float = 0.98   # live tile only; the event is undecayed
    entrance_zone: str = "entrance"     # excluded from the occupancy count

    # --- standing still ---
    # A background subtractor sees *change*, so it goes blind to anyone who stops
    # moving -- which is precisely the shopper dwell time exists to measure. But
    # a track that loses its blob while stationary almost certainly did not leave,
    # because leaving requires moving. Hold it, and keep counting it.
    static_speed_px: float = 1.5        # px/frame; a walking person is ~6
    static_max_age: int = 450           # 30 s at 15 fps before we admit defeat
    # Arriving requires moving too. A shopper standing at a shelf walked there; a
    # blob that materialised in place and never moved is a basket or a reflection
    # that flickered, and holding it for 30 s invents a browser. Roughly one body
    # width of real movement away from where it first appeared, measured as peak
    # excursion so jitter cannot accumulate into a fake journey.
    static_min_travel_px: float = 40.0
    # ...and it has to *stay* that far away for this many frames. A single merge
    # with a passing shopper yanks one measurement across the aisle and would
    # otherwise count as a 162 px journey for a basket that never moved.
    static_min_far_frames: int = 5
    # Confirmed for this many frames without ever moving that far, and it is not a
    # shopper at all: a trolley wheel or a glossy basket the background model keeps
    # flagging. Shorter than min_visit_s so it never registers as a visit.
    furniture_age: int = 25
    # Retiring furniture is not enough -- the blob is still there next frame and
    # spawns a fresh track, counted for 23 of every 26 frames. Remember where
    # furniture was and refuse to start tracks there for this long (40 s at 15 fps).
    furniture_memory: int = 600
    # When a track dies and a new one is born on the same spot moments later, it
    # is the same person -- id churn, not a new arrival. The child inherits that
    # its predecessor walked here, so a shopper standing at a shelf whose track
    # got recycled is not mistaken for furniture. Ids stay ephemeral regardless.
    relink_memory: int = 60             # 4 s at 15 fps
    # Re-identifying someone who was occluded, on motion and geometry alone --
    # no appearance, no embeddings, so the privacy claim in docs/DPDP.md holds.
    # A person cannot teleport, does not change size, and does not usually double
    # back the instant they are out of sight.
    relink_max_speed_px: float = 22.0   # px/frame ceiling on how far they could get
    relink_size_ratio: float = 1.8      # box may differ by up to this factor

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
