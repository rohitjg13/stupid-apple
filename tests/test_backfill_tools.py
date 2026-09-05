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
