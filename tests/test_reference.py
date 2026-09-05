"""pl/reference.py — OpenCV clone of the PL chain. Khushwant's golden model.

Stage order is fixed so he can diff stage by stage:
gray -> resize 320x240 -> bg -> morph open -> CCL -> filter/sort/cap/x2
     -> Sobel mean per ROI -> fg % per lane cell
"""
import numpy as np
import pytest

from core.config import load_clipset
from pl.contract import FLAG_BG_WARM, FRAME_RESULT_DT, MAX_BLOBS
from pl.reference import ReferenceBackend

CFG = load_clipset("config/sim")


def blank(v=30):
    return np.full((480, 640, 3), v, np.uint8)


def with_box(img, x, y, w, h, v=220):
    out = img.copy()
    out[y:y + h, x:x + w] = v
    return out


def warm(be, n=None, img=None):
    """Run past the model's own warm-up counter, or blobs are suppressed."""
    img = blank() if img is None else img
    for i in range(n or be.warmup_frames + 10):
        be.process(img, stream_id=0, frame_id=i)
    return be


def test_returns_a_valid_frame_result():
    be = ReferenceBackend(CFG)
    r = be.process(blank(), stream_id=0, frame_id=7)
    assert r.dtype == FRAME_RESULT_DT and r.nbytes == 856
    assert int(r["frame_id"]) == 7


def test_bg_warm_flag_follows_the_warmup_counter():
    be = ReferenceBackend(CFG, warmup_frames=10)
    flags = [int(be.process(blank(), 0, i)["flags"]) & FLAG_BG_WARM for i in range(15)]
    assert flags[0] == 0 and flags[-1] == FLAG_BG_WARM


def test_static_scene_has_no_blobs():
    be = warm(ReferenceBackend(CFG))
    assert int(be.process(blank(), 0, 99)["num_blobs"]) == 0


def test_a_moving_box_becomes_one_blob_at_full_res_coords():
    be = warm(ReferenceBackend(CFG))
    r = be.process(with_box(blank(), 200, 150, 80, 120), 0, 100)
    assert int(r["num_blobs"]) == 1
    b = r["blobs"][0]
    # PL works at 320x240 and doubles back to full res, so expect ~2 px granularity
    assert abs(int(b["x"]) - 200) <= 6 and abs(int(b["y"]) - 150) <= 6
    assert abs(int(b["w"]) - 80) <= 8 and abs(int(b["h"]) - 120) <= 8
    assert int(b["area"]) > 0


def test_blobs_are_sorted_by_area_descending():
    be = warm(ReferenceBackend(CFG))
    img = blank()
    img = with_box(img, 20, 20, 30, 30)
    img = with_box(img, 300, 200, 120, 140)
    img = with_box(img, 500, 100, 60, 60)
    r = be.process(img, 0, 100)
    areas = [int(b["area"]) for b in r["blobs"][: r["num_blobs"]]]
    assert len(areas) == 3
    assert areas == sorted(areas, reverse=True)


def test_min_blob_area_filters_specks():
    be = warm(ReferenceBackend(CFG, min_blob_area=150))
    r = be.process(with_box(blank(), 300, 200, 8, 8), 0, 100)     # 4x4 at 320x240 = 16 px
    assert int(r["num_blobs"]) == 0


def test_more_than_64_blobs_sets_the_truncation_flag():
    be = warm(ReferenceBackend(CFG, min_blob_area=4))
    img = blank()
    for i in range(70):                       # 70 well-separated squares
        x, y = 10 + (i % 10) * 62, 10 + (i // 10) * 62
        img = with_box(img, x, y, 20, 20)
    r = be.process(img, 0, 100)
    assert int(r["num_blobs"]) == MAX_BLOBS
    assert int(r["flags"]) & (1 << 1)


def test_roi_fill_is_high_on_texture_and_low_on_a_blank_facing():
    be = ReferenceBackend(CFG)
    rng = np.random.default_rng(0)
    img = blank(120)
    a1 = CFG.rois[0]                                    # 320x240 coords -> x2 for full res
    img[a1["y"] * 2:(a1["y"] + a1["h"]) * 2, a1["x"] * 2:(a1["x"] + a1["w"]) * 2] = \
        rng.integers(0, 255, (a1["h"] * 2, a1["w"] * 2, 3), dtype=np.uint8)
    r = be.process(img, stream_id=1, frame_id=0)
    assert int(r["roi_fill"][0]) > 100, "textured facing should read as stocked"
    assert int(r["roi_fill"][1]) < 40, "blank facing should read as empty"


def test_unused_roi_and_lane_slots_are_zero():
    be = ReferenceBackend(CFG)
    r = be.process(blank(), stream_id=1, frame_id=0)
    assert not r["roi_fill"][len(CFG.rois):].any()
    assert not r["lane_occupancy"][len(CFG.lanes):].any()


def test_lane_occupancy_is_a_percentage_and_tracks_foreground():
    be = warm(ReferenceBackend(CFG))
    empty = be.process(blank(), 0, 100)
    assert int(empty["lane_occupancy"][0]) < 10

    l0 = CFG.lanes[0]
    img = with_box(blank(), l0["x"] * 2, l0["y"] * 2, l0["w"] * 2, l0["h"] * 2)
    full = be.process(img, 0, 101)
    assert 80 <= int(full["lane_occupancy"][0]) <= 100


def test_deterministic_for_the_same_input():
    a = warm(ReferenceBackend(CFG)).process(with_box(blank(), 100, 100, 60, 60), 0, 100)
    b = warm(ReferenceBackend(CFG)).process(with_box(blank(), 100, 100, 60, 60), 0, 100)
    assert a.tobytes() == b.tobytes()


def test_letterboxes_a_non_640x480_input():
    be = ReferenceBackend(CFG)
    r = be.process(np.full((720, 1280, 3), 30, np.uint8), 0, 0)
    assert r.dtype == FRAME_RESULT_DT


def test_stage_outputs_written_for_khushwant(tmp_path):
    be = ReferenceBackend(CFG, stage_dir=tmp_path)
    be.process(with_box(blank(), 100, 100, 60, 60), 0, 42)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"000042_0_gray.png", "000042_1_mask.png", "000042_2_morph.png"} <= names
    import cv2
    assert cv2.imread(str(tmp_path / "000042_0_gray.png"), 0).shape == (240, 320)


def test_bg_model_resets_when_the_stream_changes():
    """One background model in BRAM (plan 01, phase 2): it re-warms on switch."""
    be = warm(ReferenceBackend(CFG, warmup_frames=10))
    assert int(be.process(blank(), 0, 30)["flags"]) & FLAG_BG_WARM
    assert not int(be.process(blank(), 1, 31)["flags"]) & FLAG_BG_WARM


def test_unknown_bg_model_rejected():
    with pytest.raises(ValueError):
        ReferenceBackend(CFG, bg_model="magic")
