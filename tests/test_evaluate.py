"""tools/evaluate.py — accuracy vs footage/groundtruth/*.json (W5)."""
import json

import pytest

from core.events import Event
from tools.evaluate import evaluate, load_groundtruth

STORE = "demo-01"


def ev(t, event_type, payload, stream="overhead", zone_id=None):
    return Event(t, STORE, stream, zone_id, event_type, payload)


GT = {
    "clip": "overhead_01",
    "entries": [{"t": 10.0, "dir": "in"}, {"t": 20.0, "dir": "in"}, {"t": 30.0, "dir": "out"}],
    "occupancy": [[10.0, 1], [20.0, 2], [30.0, 1]],
    "stockouts": [{"facing": "A1", "t_start": 100.0, "t_end": 160.0}],
}


def test_perfect_run_scores_perfectly():
    events = [ev(e["t"], "tripwire", {"dir": e["dir"], "tripwire_id": "door"}, zone_id="door")
              for e in GT["entries"]]
    events += [ev(t, "occupancy", {"count": c}) for t, c in GT["occupancy"]]
    events += [ev(100.0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"),
               ev(160.0, "stockout_end", {"sku": "X", "facing": "A1"}, "shelf", "A1")]
    m = evaluate(events, GT)
    assert m["entry_count_error"] == 0.0
    assert m["occupancy_mae"] == 0.0
    assert m["stockout_recall"] == 1.0
    assert m["stockout_false_alarms"] == 0
    assert m["passes"] is True


def test_missed_entries_show_up_as_count_error():
    events = [ev(10.0, "tripwire", {"dir": "in", "tripwire_id": "door"}, zone_id="door")]
    events += [ev(t, "occupancy", {"count": c}) for t, c in GT["occupancy"]]
    m = evaluate(events, GT)
    assert m["entry_count_error"] == pytest.approx(-0.5)   # 1 of 2 "in" crossings
    assert m["passes"] is False


def test_occupancy_mae_uses_the_nearest_reported_count():
    events = [ev(t + 0.2, "occupancy", {"count": c + 1}) for t, c in GT["occupancy"]]
    m = evaluate(events, GT)
    assert m["occupancy_mae"] == pytest.approx(1.0)


def test_occupancy_with_no_events_is_reported_not_crashed():
    m = evaluate([], GT)
    assert m["occupancy_mae"] is None
    assert m["passes"] is False


def test_a_stockout_that_was_never_detected_lowers_recall():
    events = [ev(t, "occupancy", {"count": c}) for t, c in GT["occupancy"]]
    m = evaluate(events, GT)
    assert m["stockout_recall"] == 0.0


def test_a_stockout_nobody_asked_for_is_a_false_alarm():
    events = [ev(t, "occupancy", {"count": c}) for t, c in GT["occupancy"]]
    events += [ev(500.0, "stockout_start", {"sku": "X", "facing": "B2"}, "shelf", "B2")]
    m = evaluate(events, GT)
    assert m["stockout_false_alarms"] == 1


def test_detection_within_tolerance_still_counts():
    events = [ev(t, "occupancy", {"count": c}) for t, c in GT["occupancy"]]
    events += [ev(104.0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1")]
    m = evaluate(events, GT, stockout_tolerance_s=10.0)
    assert m["stockout_recall"] == 1.0
    assert m["stockout_false_alarms"] == 0


def test_load_groundtruth_rejects_a_file_missing_required_keys(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"clip": "x"}))
    with pytest.raises(ValueError, match="occupancy"):
        load_groundtruth(p)


def test_thresholds_match_the_test_protocol():
    from tools.evaluate import TARGETS
    assert TARGETS["entry_count_error"] == 0.10
    assert TARGETS["occupancy_mae"] == 1.5
    assert TARGETS["stockout_recall"] == 0.95
