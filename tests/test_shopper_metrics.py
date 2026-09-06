"""shopper/metrics.py -- scoring against either ground-truth schema."""
import pytest

from core.events import Event
from shopper.metrics import (dwell_and_switches, entry_exit_error, evaluate_shopper,
                             normalise, occupancy_mae, passes)

STORE = "demo-01"


def ev(t, event_type, payload, zone_id=None):
    return Event(t, STORE, "overhead", zone_id, event_type, payload)


def tripwire(t, direction):
    return ev(t, "tripwire", {"dir": direction, "tripwire_id": "door"}, zone_id="door")


def visit(track_id, zone, t0, t1):
    return ev(t1, "visit", {"track_id": track_id, "t_enter": t0, "t_exit": t1},
              zone_id=zone)


# The shape tools/evaluate.py and tests/test_evaluate.py already use.
GT_LEGACY = {
    "entries": [{"t": 10.0, "dir": "in"}, {"t": 20.0, "dir": "in"},
                {"t": 30.0, "dir": "out"}],
    "occupancy": [[10.0, 1], [20.0, 2]],
}
# The shape plan 06's annotation tool will write.
GT_PLAN06 = {
    "clip": "overhead_01", "fps": 15, "duration_s": 120,
    "entries": [{"t": 10.0}, {"t": 20.0}],
    "exits": [{"t": 30.0}],
    "occupancy": [{"t": 10.0, "count": 1}, {"t": 20.0, "count": 2}],
    "people": [{"gt_id": 1, "zones": [{"zone": "aisle_a", "t_enter": 20.0,
                                       "t_exit": 50.0}]}],
}


# --- normalisation --------------------------------------------------------
def test_both_schemas_normalise_to_the_same_entries_and_exits():
    a, b = normalise(GT_LEGACY), normalise(GT_PLAN06)
    assert a["entries"] == b["entries"] == [10.0, 20.0]
    assert a["exits"] == b["exits"] == [30.0]


def test_both_occupancy_shapes_normalise():
    assert normalise(GT_LEGACY)["occupancy"] == normalise(GT_PLAN06)["occupancy"]


def test_people_zones_become_labelled_visits():
    v = normalise(GT_PLAN06)["visits"]
    assert v == [{"gt_id": 1, "zone": "aisle_a", "t_enter": 20.0, "t_exit": 50.0}]


def test_ground_truth_without_people_has_no_labelled_visits():
    assert normalise(GT_LEGACY)["visits"] == []


# --- counts ---------------------------------------------------------------
def test_a_perfect_run_has_zero_count_error():
    events = [tripwire(10.0, "in"), tripwire(20.0, "in"), tripwire(30.0, "out")]
    assert entry_exit_error(events, GT_PLAN06) == (0.0, 0.0)


def test_a_missed_exit_shows_as_a_negative_error():
    events = [tripwire(10.0, "in"), tripwire(20.0, "in")]
    _, exit_err = entry_exit_error(events, GT_PLAN06)
    assert exit_err == pytest.approx(-1.0)


def test_a_double_counted_exit_shows_as_a_positive_error():
    events = [tripwire(30.0, "out"), tripwire(35.0, "out")]
    _, exit_err = entry_exit_error(events, GT_PLAN06)
    assert exit_err == pytest.approx(1.0)


def test_count_error_is_none_when_nothing_is_labelled():
    assert entry_exit_error([], {"entries": [], "exits": []}) == (None, None)


# --- occupancy ------------------------------------------------------------
def test_occupancy_mae_of_a_perfect_run_is_zero():
    events = [ev(10.0, "occupancy", {"count": 1}), ev(20.0, "occupancy", {"count": 2})]
    assert occupancy_mae(events, GT_PLAN06) == 0.0


def test_occupancy_mae_uses_the_nearest_report():
    events = [ev(10.2, "occupancy", {"count": 2}), ev(20.2, "occupancy", {"count": 3})]
    assert occupancy_mae(events, GT_PLAN06) == pytest.approx(1.0)


def test_occupancy_mae_is_none_with_no_reports():
    assert occupancy_mae([], GT_PLAN06) is None


# --- dwell and fragmentation ---------------------------------------------
def test_a_perfectly_matched_visit_has_zero_dwell_error():
    m = dwell_and_switches([visit(0, "aisle_a", 20.0, 50.0)], GT_PLAN06)
    assert m["dwell_mae"] == 0.0
    assert m["visits_matched"] == 1


def test_dwell_error_is_the_absolute_difference():
    m = dwell_and_switches([visit(0, "aisle_a", 20.0, 44.0)], GT_PLAN06)
    assert m["dwell_mae"] == pytest.approx(6.0)


def test_a_visit_in_the_wrong_zone_does_not_match():
    m = dwell_and_switches([visit(0, "aisle_b", 20.0, 50.0)], GT_PLAN06)
    assert m["dwell_mae"] is None and m["visits_matched"] == 0


def test_a_barely_overlapping_visit_does_not_match():
    m = dwell_and_switches([visit(0, "aisle_a", 48.0, 52.0)], GT_PLAN06)
    assert m["visits_matched"] == 0


def test_a_visit_split_across_two_tracks_counts_as_an_id_switch():
    events = [visit(0, "aisle_a", 20.0, 38.0), visit(1, "aisle_a", 32.0, 50.0)]
    assert dwell_and_switches(events, GT_PLAN06)["id_switches"] == 1


def test_a_cleanly_tracked_visit_has_no_id_switch():
    assert dwell_and_switches([visit(0, "aisle_a", 20.0, 50.0)],
                              GT_PLAN06)["id_switches"] == 0


def test_dwell_metrics_are_none_when_no_people_are_labelled():
    m = dwell_and_switches([visit(0, "aisle_a", 1.0, 9.0)], GT_LEGACY)
    assert m["dwell_mae"] is None and m["id_switches"] is None


# --- the aggregate --------------------------------------------------------
def test_evaluate_shopper_reports_the_keys_evaluate_py_merges():
    m = evaluate_shopper([visit(0, "aisle_a", 20.0, 50.0), tripwire(30.0, "out")],
                         GT_PLAN06)
    assert set(m) >= {"exit_count_error", "dwell_mae", "id_switches",
                      "visits_detected", "visits_labelled"}


def test_a_good_run_passes():
    m = evaluate_shopper([visit(0, "aisle_a", 20.0, 50.0), tripwire(30.0, "out")],
                         GT_PLAN06)
    assert passes(m) is True


def test_a_bad_dwell_fails():
    # 20 s measured against 30 s labelled: matches on overlap, misses on dwell.
    m = evaluate_shopper([visit(0, "aisle_a", 20.0, 40.0), tripwire(30.0, "out")],
                         GT_PLAN06)
    assert m["dwell_mae"] == pytest.approx(10.0)
    assert passes(m) is False


def test_a_visit_too_fragmented_to_match_is_a_miss_not_a_dwell_error():
    """visits_matched is what reports it; dwell stays unknown rather than wrong."""
    m = evaluate_shopper([visit(0, "aisle_a", 20.0, 26.0)], GT_PLAN06)
    assert m["visits_matched"] == 0 and m["dwell_mae"] is None


def test_metrics_that_could_not_be_computed_do_not_fail_the_run():
    """Missing labels mean unknown, not failed."""
    assert passes({"exit_count_error": None, "dwell_mae": None}) is True


# --- results files for the accuracy slide ---------------------------------
def test_a_result_file_is_written_per_clip(tmp_path):
    from shopper.metrics import write_result
    import json
    p = write_result({"occupancy_mae": 0.5, "passes": True}, GT_PLAN06, tmp_path)
    assert p.name == "overhead_01.json"
    assert json.loads(p.read_text())["clip"] == "overhead_01"


def test_summary_averages_across_clips(tmp_path):
    from shopper.metrics import summarise, write_result
    write_result({"occupancy_mae": 1.0, "passes": True},
                 dict(GT_PLAN06, clip="a"), tmp_path)
    write_result({"occupancy_mae": 2.0, "passes": True},
                 dict(GT_PLAN06, clip="b"), tmp_path)
    s = summarise(tmp_path)
    assert s["n_clips"] == 2
    assert s["occupancy_mae"] == pytest.approx(1.5)
    assert s["all_passed"] is True


def test_summary_reports_a_failing_clip(tmp_path):
    from shopper.metrics import summarise, write_result
    write_result({"passes": True}, dict(GT_PLAN06, clip="a"), tmp_path)
    write_result({"passes": False}, dict(GT_PLAN06, clip="b"), tmp_path)
    assert summarise(tmp_path)["all_passed"] is False


def test_a_metric_no_clip_could_measure_stays_none(tmp_path):
    from shopper.metrics import summarise, write_result
    write_result({"dwell_mae": None}, dict(GT_PLAN06, clip="a"), tmp_path)
    assert summarise(tmp_path)["dwell_mae"] is None


def test_summary_excludes_itself_when_rerun(tmp_path):
    from shopper.metrics import summarise, write_result
    write_result({"occupancy_mae": 1.0, "passes": True},
                 dict(GT_PLAN06, clip="a"), tmp_path)
    summarise(tmp_path)
    assert summarise(tmp_path)["n_clips"] == 1, "summary.json counted as a clip"
