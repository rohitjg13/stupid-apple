"""tools/backfill_history.py, tools/reset_run.py, and the queue half of evaluate."""
import pytest

from backend.db import DB
from core.config import load_clipset
from tools.backfill_history import backfill
from tools.reset_run import reset
from tools.evaluate import evaluate_queue

CFG = load_clipset("config/sim")


def test_reset_run_starts_a_run(tmp_path):
    db = DB(tmp_path / "r.db")
    reset(db, "seg-1", "demo-01", "sim", t=123.0)
    assert db.query_one("SELECT run_id, clipset FROM run WHERE run_id='seg-1'") == \
        {"run_id": "seg-1", "clipset": "sim"}
    db.close()


def test_backfill_writes_events_under_run_id(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.05)
    n = backfill(db, CFG, "demo-01", "history", days=1.0, seed=0)
    assert n > 0
    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy WHERE run_id='history'")["n"] > 0
    assert db.query_one("SELECT store_id FROM run WHERE run_id='history'")["store_id"] == "demo-01"
    db.close()


def test_evaluate_queue_matches_nearest_prediction():
    from core.events import Event

    def qe(t, lane, count, wait):
        return Event(t, "demo-01", "overhead", None, "queue_estimate",
                     {"lane": lane, "count": count, "pred_wait_s": wait})

    events = [qe(10.0, 1, 5.0, 120.0), qe(12.0, 1, 6.0, 130.0)]
    truth = [{"t": 11.0, "lane": 1, "count": 5, "actual_wait_s": 110.0}]
    m = evaluate_queue(events, truth)
    assert m["count_mae"] == pytest.approx(0.0)
    assert m["wait_mae"] == pytest.approx(10.0)
    assert m["count_pass"] is True


def test_evaluate_queue_ignores_far_predictions():
    from core.events import Event

    def qe(t, lane, count, wait):
        return Event(t, "demo-01", "overhead", None, "queue_estimate",
                     {"lane": lane, "count": count, "pred_wait_s": wait})

    events = [qe(1000.0, 1, 5.0, 120.0)]
    truth = [{"t": 11.0, "lane": 1, "count": 5, "actual_wait_s": 120.0}]
    m = evaluate_queue(events, truth)
    assert m["n"] == 0 and m["count_mae"] is None


def test_evaluate_queue_empty():
    assert evaluate_queue([], []) == {"count_mae": None, "wait_mae": None, "n": 0}


# ---- backfill pre-aggregates, or the history charts are empty at boot -------

def test_backfill_writes_kpi_buckets(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.05)
    try:
        backfill(db, CFG, "demo-01", "history", days=1.0, seed=0)
        rows = db.query("SELECT DISTINCT t_bucket FROM kpi_15m WHERE run_id='history'")
        assert len(rows) > 24, "a day of history should be many 15-minute buckets"
        keys = {r["key"] for r in db.query("SELECT DISTINCT key FROM kpi_15m")}
        assert {"footfall_in", "occupancy_mean", "pos_amount"} <= keys
    finally:
        db.close()


def test_backfill_rollup_can_be_skipped(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.05)
    try:
        backfill(db, CFG, "demo-01", "h", days=0.5, seed=0, rollup=False)
        assert db.query_one("SELECT COUNT(*) AS n FROM kpi_15m")["n"] == 0
    finally:
        db.close()


def test_backfill_twice_does_not_trip_the_run_primary_key(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.05)
    try:
        backfill(db, CFG, "demo-01", "history", days=0.5, seed=0)
        backfill(db, CFG, "demo-01", "history", days=0.5, seed=0)
        assert db.query_one("SELECT COUNT(*) AS n FROM run")["n"] == 1
        assert db.dropped == 0
    finally:
        db.close()


def test_backfill_gives_the_multi_store_view_its_stores(tmp_path):
    from backend.aggregates import Aggregates
    db = DB(tmp_path / "r.db", batch_interval_s=0.05)
    try:
        for store, run_id, seed in [("demo-01", "history", 0),
                                    ("demo-02", "history-demo02", 1),
                                    ("demo-03", "history-demo03", 2)]:
            backfill(db, CFG, store, run_id, days=0.25, seed=seed, rollup=False)
        assert Aggregates(db).stores() == ["demo-01", "demo-02", "demo-03"]
    finally:
        db.close()


# ---- queue ground truth ----------------------------------------------------

QUEUE_TRUTH = {
    "queue": [{"t": 10.0, "lane": 1, "count": 2, "actual_wait_s": 60.0},
              {"t": 20.0, "lane": 1, "count": 7, "actual_wait_s": 240.0}],
    "peak_t": 20.0,
}


def _write(tmp_path, body):
    import json
    p = tmp_path / "q.json"
    p.write_text(json.dumps(body))
    return p


def test_load_queue_truth(tmp_path):
    from tools.evaluate import load_queue_truth
    rows, peak = load_queue_truth(_write(tmp_path, QUEUE_TRUTH))
    assert len(rows) == 2 and peak == 20.0


def test_load_queue_truth_rejects_a_non_queue_file(tmp_path):
    from tools.evaluate import load_queue_truth
    with pytest.raises(ValueError, match="not a queue clip"):
        load_queue_truth(_write(tmp_path, {"entries": []}))


def test_load_queue_truth_names_the_missing_field(tmp_path):
    from tools.evaluate import load_queue_truth
    # `count` is required; `actual_wait_s` is not (see the count-only tests).
    with pytest.raises(ValueError, match="count"):
        load_queue_truth(_write(tmp_path, {"queue": [{"t": 1, "lane": 1}]}))


def test_store_queue_truth_feeds_wait_pred_vs_actual(tmp_path):
    from backend.aggregates import Aggregates
    from tools.evaluate import store_queue_truth

    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    try:
        assert store_queue_truth(db, QUEUE_TRUTH["queue"], "queue_01") == 2
        db.insert("queue_estimate", {"run_id": "r1", "t": 20.0, "lane": 1,
                                     "count": 6.0, "pred_wait_s": 200.0}).drain()
        out = Aggregates(db, run_id="r1").wait_pred_vs_actual(0, 100)
        assert out["n"] == 2
        assert out["rows"][1]["actual_wait_s"] == 240.0
    finally:
        db.close()


def test_store_queue_truth_replaces_a_previous_load(tmp_path):
    from tools.evaluate import store_queue_truth
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    try:
        store_queue_truth(db, QUEUE_TRUTH["queue"], "queue_01")
        store_queue_truth(db, QUEUE_TRUTH["queue"], "queue_01")
        assert db.query_one("SELECT COUNT(*) AS n FROM queue_truth")["n"] == 2
    finally:
        db.close()


# ---- staffing alert lead time (plan W5 §2) ---------------------------------

def _alert(t, rule="staffing"):
    from core.events import Event
    return Event(t, "demo-01", "-", None, "alert",
                 {"severity": "warning", "rule": rule, "message": "open counter 2"})


def test_alert_lead_time_is_measured_back_from_the_peak():
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(-160.0)], QUEUE_TRUTH["queue"], peak_t=20.0)
    assert m["peak_t"] == 20.0
    assert m["lead_time_s"] == pytest.approx(180.0)
    assert m["passes"] is True          # target is 2 minutes


def test_alert_lead_time_fails_when_the_warning_is_too_late():
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(-40.0)], QUEUE_TRUTH["queue"], peak_t=20.0)
    assert m["lead_time_s"] == pytest.approx(60.0)
    assert m["passes"] is False


def test_alert_after_the_peak_does_not_count():
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(30.0)], QUEUE_TRUTH["queue"], peak_t=20.0)
    assert m["alert_t"] is None and m["passes"] is False


def test_an_alert_from_an_unrelated_blip_does_not_count():
    """Half an hour early is not a prediction of this peak."""
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(-2000.0)], QUEUE_TRUTH["queue"], peak_t=20.0)
    assert m["alert_t"] is None


def test_only_staffing_alerts_count_towards_lead_time():
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(-160.0, rule="stockout")], QUEUE_TRUTH["queue"],
                        peak_t=20.0)
    assert m["alert_t"] is None


def test_the_earliest_qualifying_alert_wins():
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(-100.0), _alert(-300.0)], QUEUE_TRUTH["queue"],
                        peak_t=20.0)
    assert m["alert_t"] == -300.0


def test_peak_falls_back_to_the_busiest_labelled_sample():
    from tools.evaluate import alert_lead_time, peak_time
    assert peak_time(QUEUE_TRUTH["queue"]) == 20.0     # the count-7 row
    m = alert_lead_time([_alert(-160.0)], QUEUE_TRUTH["queue"])
    assert m["lead_time_s"] == pytest.approx(180.0)


def test_alert_lead_time_with_no_truth_at_all():
    from tools.evaluate import alert_lead_time
    m = alert_lead_time([_alert(0.0)], [])
    assert m == {"peak_t": None, "alert_t": None, "lead_time_s": None,
                 "passes": False}


def test_queue_count_target_comes_from_the_shared_table():
    from tools.evaluate import TARGETS, evaluate_queue
    from core.events import Event

    def qe(t, count):
        return Event(t, "demo-01", "overhead", None, "queue_estimate",
                     {"lane": 1, "count": count, "pred_wait_s": 100.0})

    truth = [{"t": 10.0, "lane": 1, "count": 5, "actual_wait_s": 100.0}]
    assert TARGETS["queue_count_mae"] == 1.0
    assert evaluate_queue([qe(10.0, 5.5)], truth)["count_pass"] is True
    assert evaluate_queue([qe(10.0, 7.0)], truth)["count_pass"] is False


# ---- count-only ground truth (a clip too short to time a wait) -------------

COUNT_ONLY = {"queue": [{"t": 10.0, "lane": 1, "count": 4},
                        {"t": 20.0, "lane": 1, "count": 3}]}


def test_load_queue_truth_accepts_counts_without_wait_times(tmp_path):
    from tools.evaluate import load_queue_truth
    rows, peak = load_queue_truth(_write(tmp_path, COUNT_ONLY))
    assert len(rows) == 2 and peak is None


def test_evaluate_queue_reports_count_mae_without_wait_labels():
    """A missing wait label means 'not measured', not an error of zero."""
    from core.events import Event
    from tools.evaluate import evaluate_queue

    def qe(t, count):
        return Event(t, "demo-01", "overhead", None, "queue_estimate",
                     {"lane": 1, "count": count, "pred_wait_s": 999.0})

    m = evaluate_queue([qe(10.0, 3.0), qe(20.0, 3.0)], COUNT_ONLY["queue"])
    assert m["n"] == 2
    assert m["count_mae"] == pytest.approx(0.5)     # |3-4| and |3-3|
    assert m["wait_mae"] is None                     # not fabricated as 0


def test_evaluate_queue_still_scores_waits_when_they_are_labelled():
    from core.events import Event
    from tools.evaluate import evaluate_queue

    truth = [{"t": 10.0, "lane": 1, "count": 4, "actual_wait_s": 100.0}]
    e = Event(10.0, "demo-01", "overhead", None, "queue_estimate",
              {"lane": 1, "count": 4.0, "pred_wait_s": 130.0})
    assert evaluate_queue([e], truth)["wait_mae"] == pytest.approx(30.0)


def test_the_real_clip_labels_load(tmp_path):
    """footage/groundtruth/queue_01.json is committed; the clip is not."""
    import os
    from tools.evaluate import load_queue_truth
    path = "footage/groundtruth/queue_01.json"
    if not os.path.exists(path):
        pytest.skip("queue_01 labels not present")
    rows, peak = load_queue_truth(path)
    assert len(rows) == 31
    assert {r["lane"] for r in rows} == {1}
    assert all(r["count"] >= 0 for r in rows)
