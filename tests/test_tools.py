"""tools/compare_backends.py and tools/perf.py — the W4 integration numbers."""
import numpy as np
import pytest

from core.config import load_clipset
from pl.contract import FRAME_RESULT_DT
from pl.reference import ReferenceBackend
from tools.compare_backends import compare, iou, match_blobs
from tools.perf import Stats, benchmark, format_table

CFG = load_clipset("config/sim")


def result(boxes):
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    r["num_blobs"] = len(boxes)
    for i, (x, y, w, h) in enumerate(boxes):
        r["blobs"][i] = (x, y, w, h, w * h)
    return r


# ---- IoU and matching ----------------------------------------------------

def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_of_half_overlap():
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_touching_boxes_do_not_overlap():
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0


def test_match_blobs_pairs_the_obvious_ones():
    a = result([(0, 0, 20, 20), (100, 100, 30, 30)])
    b = result([(102, 101, 30, 30), (1, 1, 20, 20)])
    pairs, unmatched_a, unmatched_b = match_blobs(a, b, thresh=0.5)
    assert len(pairs) == 2 and not unmatched_a and not unmatched_b
    assert min(p[2] for p in pairs) > 0.8


def test_match_blobs_reports_what_it_could_not_pair():
    a = result([(0, 0, 20, 20), (300, 300, 20, 20)])
    b = result([(0, 0, 20, 20)])
    pairs, ua, ub = match_blobs(a, b, thresh=0.5)
    assert len(pairs) == 1 and ua == [1] and ub == []


def test_each_blob_is_matched_at_most_once():
    a = result([(0, 0, 20, 20)])
    b = result([(0, 0, 20, 20), (1, 1, 20, 20)])
    pairs, ua, ub = match_blobs(a, b, thresh=0.5)
    assert len(pairs) == 1 and len(ub) == 1


# ---- compare -------------------------------------------------------------

def test_identical_streams_compare_perfectly():
    rs = [result([(10, 10, 20, 20), (50, 50, 30, 30)]) for _ in range(10)]
    s = compare(rs, list(rs))
    assert s["count_agreement"] == 1.0
    assert s["mean_iou"] == pytest.approx(1.0)
    assert s["frames"] == 10


def test_compare_detects_disagreement():
    a = [result([(10, 10, 20, 20), (50, 50, 30, 30)]) for _ in range(10)]
    b = [result([(10, 10, 20, 20)]) for _ in range(10)]
    s = compare(a, b)
    assert s["count_agreement"] == 0.0
    assert s["mean_blob_delta"] == pytest.approx(-1.0)


def test_compare_applies_the_w4_acceptance_thresholds():
    """docs/plans/02-rohit.md W4: >= 0.7 IoU and >= 90 % count agreement."""
    rs = [result([(10, 10, 20, 20)]) for _ in range(10)]
    assert compare(rs, list(rs))["passes_w4"] is True
    bad = [result([(10, 10, 20, 20), (99, 99, 5, 5)]) for _ in range(10)]
    assert compare(rs, bad)["passes_w4"] is False


def test_compare_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compare([result([])], [result([]), result([])])


def test_compare_handles_frames_with_no_blobs():
    rs = [result([]) for _ in range(5)]
    s = compare(rs, list(rs))
    assert s["count_agreement"] == 1.0
    assert s["mean_iou"] == 0.0        # nothing to match, not a failure
    assert s["matched"] == 0


def test_roi_fill_agreement_is_reported():
    a, b = result([]), result([])
    a["roi_fill"][:3] = [200, 100, 50]
    b["roi_fill"][:3] = [205, 95, 50]
    s = compare([a], [b], n_rois=3)
    assert s["roi_fill_mae"] == pytest.approx(10 / 3)


# ---- perf ----------------------------------------------------------------

def test_benchmark_returns_usable_numbers():
    be = ReferenceBackend(CFG)
    img = np.full((480, 640, 3), 30, np.uint8)
    s = benchmark(be, [img] * 20, label="reference")
    assert s.frames == 20
    assert s.fps > 0 and s.mean_ms > 0
    assert s.p95_ms >= s.mean_ms * 0.5
    assert s.latency_cycles is None      # no LATENCY register off the board


def test_benchmark_reads_the_latency_register_when_present():
    class FakeDriver:
        def process(self, img, stream_id=0, frame_id=0):
            return np.zeros(1, dtype=FRAME_RESULT_DT)[0]

        def read_latency_cycles(self):
            return 250_000

    s = benchmark(FakeDriver(), [np.zeros((480, 640, 3), np.uint8)] * 5, label="pl")
    assert s.latency_cycles == 250_000
    assert s.pl_ms_at_100mhz == pytest.approx(2.5)


def test_format_table_lines_up():
    rows = [Stats("reference", 100, 12.0, 15.0, 83.0, None), Stats("pl", 100, 3.0, 4.0, 333.0, 250_000)]
    out = format_table(rows)
    assert "reference" in out and "pl" in out
    widths = {len(l) for l in out.splitlines()}
    assert len(widths) == 1, "columns are not aligned"
