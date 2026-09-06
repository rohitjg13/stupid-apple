"""pl/yolo.py -- a detector packed into the frozen FrameResult, no torch needed."""
import numpy as np
import pytest

from core.config import load_clipset
from pl.contract import FLAG_BG_WARM, FLAG_TRUNCATED, FRAME_RESULT_DT, MAX_BLOBS
from pl.yolo import YoloBackend, pack_detections

CFG = load_clipset("config/sim")


def blank():
    return np.zeros(1, dtype=FRAME_RESULT_DT)[0]


def frame():
    return np.full((480, 640, 3), 40, np.uint8)


class StubDetector:
    """Stands in for the network; records how often it was asked."""

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def __call__(self, image):
        self.calls += 1
        return self.boxes


# --- packing -------------------------------------------------------------------
def test_boxes_land_in_the_contract_sorted_by_area():
    r = pack_detections(blank(), [(10, 10, 20, 30), (100, 100, 40, 136)])
    assert int(r["num_blobs"]) == 2
    first = r["blobs"][0]
    assert (int(first["x"]), int(first["y"]), int(first["w"]), int(first["h"])) == (100, 100, 40, 136)
    assert int(first["area"]) == 40 * 136


def test_the_warm_flag_is_set_from_the_first_frame():
    """A detector has no background model; nothing to wait for."""
    r = pack_detections(blank(), [])
    assert int(r["flags"]) & FLAG_BG_WARM


def test_boxes_are_clipped_to_the_frame():
    r = pack_detections(blank(), [(-20, -20, 60, 60), (620, 460, 100, 100)])
    for b in r["blobs"][: int(r["num_blobs"])]:
        assert int(b["x"]) >= 0 and int(b["y"]) >= 0
        assert int(b["x"]) + int(b["w"]) <= 640
        assert int(b["y"]) + int(b["h"]) <= 480


def test_slivers_left_after_clipping_are_dropped():
    r = pack_detections(blank(), [(639, 479, 50, 50)])
    assert int(r["num_blobs"]) == 0


def test_more_than_sixty_four_sets_truncated_and_keeps_the_biggest():
    boxes = [(i * 9 % 600, i * 7 % 440, 10 + i % 30, 20 + i % 40) for i in range(70)]
    r = pack_detections(blank(), boxes)
    assert int(r["num_blobs"]) == MAX_BLOBS
    assert int(r["flags"]) & FLAG_TRUNCATED
    areas = [int(b["area"]) for b in r["blobs"]]
    assert areas == sorted(areas, reverse=True)


def test_previous_blobs_are_cleared_not_appended():
    r = pack_detections(blank(), [(0, 0, 40, 40)] * 5)
    r = pack_detections(r, [(0, 0, 40, 40)])
    assert int(r["num_blobs"]) == 1
    assert int(r["blobs"][1]["area"]) == 0


# --- backend -------------------------------------------------------------------
def test_overhead_blobs_come_from_the_detector():
    det = StubDetector([(100, 100, 40, 136)])
    be = YoloBackend(CFG, detector=det)
    r = be.process(frame(), stream_id=0, frame_id=7)
    assert det.calls == 1
    assert int(r["frame_id"]) == 7
    assert int(r["num_blobs"]) == 1
    assert int(r["flags"]) & FLAG_BG_WARM


def test_the_shelf_stream_never_runs_the_detector():
    """A person detector has nothing to say about a shelf; reference owns it."""
    det = StubDetector([(100, 100, 40, 136)])
    be = YoloBackend(CFG, detector=det)
    be.process(frame(), stream_id=1, frame_id=0)
    assert det.calls == 0


def test_roi_fill_and_lanes_still_come_from_the_reference_chain():
    be = YoloBackend(CFG, detector=StubDetector([]))
    img = frame()
    img[:, 200:400] = 255                                   # texture for the ROIs
    r = be.process(img, stream_id=0, frame_id=0)
    assert r["roi_fill"].shape == (64,) and r["lane_occupancy"].shape == (16,)


def test_output_is_the_frozen_856_byte_contract():
    r = YoloBackend(CFG, detector=StubDetector([])).process(frame())
    assert r.dtype == FRAME_RESULT_DT and r.dtype.itemsize == 856


def test_available_is_a_bool_and_true_with_an_injected_detector():
    assert YoloBackend(CFG, detector=StubDetector([])).available() is True
    from pl.yolo import available
    assert isinstance(available(), bool)


def test_letterboxed_input_of_any_size_is_accepted():
    det = StubDetector([(0, 0, 40, 40)])
    be = YoloBackend(CFG, detector=det)
    r = be.process(np.zeros((1080, 1920, 3), np.uint8))
    assert int(r["num_blobs"]) == 1
