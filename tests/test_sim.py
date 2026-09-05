"""sources/sim.py — the day-1 dev source and the demo fallback."""
import numpy as np
import pytest

from core.config import load_clipset
from pl.contract import FRAME_RESULT_DT, MAX_BLOBS
from sources.sim import SimSource

CFG = "config/sim"


def frames(n, seed=1, stream="overhead", **kw):
    src = SimSource(load_clipset(CFG), stream=stream, seed=seed, **kw)
    try:
        return [f for _, f in zip(range(n), src.frames())]
    finally:
        src.close()


def test_yields_valid_frames():
    for f in frames(30):
        assert f.stream == "overhead"
        assert f.result.dtype == FRAME_RESULT_DT
        assert 0 <= f.result["num_blobs"] <= MAX_BLOBS
        assert f.image is None or f.image.shape == (480, 640, 3)


def test_frame_ids_are_contiguous_from_zero():
    assert [f.frame_id for f in frames(20)] == list(range(20))


def test_t_ns_is_monotonic():
    ts = [f.t_ns for f in frames(20)]
    assert ts == sorted(ts)


def test_same_seed_same_run():
    a, b = frames(40, seed=7), frames(40, seed=7)
    assert [f.result.tobytes() for f in a] == [f.result.tobytes() for f in b]


def test_different_seed_different_run():
    a, b = frames(40, seed=1), frames(40, seed=2)
    assert [f.result.tobytes() for f in a] != [f.result.tobytes() for f in b]


def test_blobs_are_inside_the_full_res_image_and_sorted_by_area():
    for f in frames(60):
        blobs = f.result["blobs"][: f.result["num_blobs"]]
        areas = [int(b["area"]) for b in blobs]
        assert areas == sorted(areas, reverse=True)
        for b in blobs:
            assert 0 <= b["x"] and b["x"] + b["w"] <= 640
            assert 0 <= b["y"] and b["y"] + b["h"] <= 480
            assert b["area"] > 0


def test_people_actually_come_and_go():
    counts = [int(f.result["num_blobs"]) for f in frames(600, seed=3)]
    assert max(counts) >= 2, "nobody ever showed up"
    assert min(counts) < max(counts), "occupancy never changed"


def test_bg_warm_flag_sets_after_warmup():
    fs = frames(150)
    assert not (fs[0].result["flags"] & 1)
    assert fs[-1].result["flags"] & 1


def test_shelf_stream_drops_a_roi_to_stockout():
    """A facing empties for 30-200 s; the shelf lead's detector needs this."""
    cfg = load_clipset(CFG)
    n_rois = len(cfg.rois)
    fills = np.array([f.result["roi_fill"][:n_rois] for f in frames(2000, seed=5, stream="shelf")])
    assert fills.max() > 150, "shelf never looks stocked"
    assert fills.min() < 60, "no facing ever empties"
    assert (fills.min(axis=0) < 60).sum() >= 1


def test_unused_roi_slots_stay_zero():
    cfg = load_clipset(CFG)
    for f in frames(50, stream="shelf"):
        assert not f.result["roi_fill"][len(cfg.rois):].any()


def test_lane_occupancy_is_a_percentage():
    for f in frames(200, seed=4):
        assert f.result["lane_occupancy"].max() <= 100


def test_overhead_stream_leaves_shelf_fields_alone():
    for f in frames(50, stream="overhead"):
        assert not f.result["roi_fill"].any()


def test_history_backfill_spans_the_requested_days():
    """sources/sim.py --history-days 7 for the backend lead's backfill."""
    src = SimSource(load_clipset(CFG), stream="overhead", seed=1)
    events = list(src.history(days=2))
    assert events, "no history generated"
    span = max(e.t for e in events) - min(e.t for e in events)
    assert 1.5 * 86400 < span <= 2 * 86400
    assert {e.event_type for e in events} <= {"occupancy", "tripwire", "shelf_fill", "pos_txn"}
