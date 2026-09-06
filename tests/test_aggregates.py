"""backend/aggregates.py — every dashboard query."""
import time

import pytest

from backend.aggregates import Aggregates
from backend.db import DB


@pytest.fixture
def agg(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    db.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0, "clipset": "sim"})
    a = Aggregates(db, run_id="r1")
    yield a
    db.close()


def test_live_summary_uses_latest_occupancy(agg):
    agg.db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 3})
    agg.db.insert("occupancy", {"run_id": "r1", "t": 2.0, "count": 7})
    agg.db.drain()
    assert agg.live_summary()["occupancy"] == 7


def test_live_summary_empty(agg):
    assert agg.live_summary()["occupancy"] == 0


def test_footfall_by_hour(agg):
    # Anchor to today's local midnight so _day_bounds matches the insertions.
    import datetime
    lt = time.localtime()
    midnight = time.mktime(datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday).timetuple())
    agg.db.insert("tripwire", {"run_id": "r1", "t": midnight + 100.0, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": midnight + 200.0, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": midnight + 3600.0, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": midnight + 500.0, "dir": "out"})
    agg.db.drain()
    rows = {r["hour"]: r["entries"] for r in agg.footfall_by_hour(time.time())}
    assert sum(rows.values()) == 3


def test_footfall_by_day(agg):
    now = time.time()
    agg.db.insert("tripwire", {"run_id": "r1", "t": now, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": now - 86400, "dir": "in"})
    agg.db.drain()
    rows = agg.footfall_by_day(2)
    assert sum(r["entries"] for r in rows) == 2


def test_dwell_by_zone_computes_median_and_p90(agg):
    agg.db.insert("visit", {"run_id": "r1", "track_id": 1, "zone": "aisle_a",
                            "t_enter": 0.0, "t_exit": 10.0})
    agg.db.insert("visit", {"run_id": "r1", "track_id": 2, "zone": "aisle_a",
                            "t_enter": 0.0, "t_exit": 20.0})
    agg.db.insert("visit", {"run_id": "r1", "track_id": 3, "zone": "aisle_a",
                            "t_enter": 0.0, "t_exit": 30.0})
    agg.db.drain()
    z = agg.dwell_by_zone(0, 100)["aisle_a"]
    assert z["visits"] == 3
    assert z["median_dwell_s"] == 20.0
    assert z["p90_dwell_s"] == 30.0


def test_heatmap_sums_counts(agg):
    agg.db.insert("heatmap", {"run_id": "r1", "t_bucket": 5, "gx": 1, "gy": 2, "count": 3})
    agg.db.insert("heatmap", {"run_id": "r1", "t_bucket": 5, "gx": 1, "gy": 3, "count": 4})
    agg.db.drain()
    rows = agg.heatmap(0, 100)
    assert sum(r["count"] for r in rows) == 7


def test_stockout_durations(agg):
    agg.db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "X",
                               "t_start": 10.0, "t_end": 70.0})
    agg.db.drain()
    rows = agg.stockout_durations(0, 100)
    assert rows[0]["duration_s"] == 60.0


def test_replenishment_list_sorted_by_revenue_rate(agg):
    agg.db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "LOW",
                               "t_start": 1.0, "t_end": None})
    agg.db.insert("stockout", {"run_id": "r1", "roi": "B1", "sku": "HIGH",
                               "t_start": 2.0, "t_end": None})
    agg.db.drain()
    planogram = [
        {"sku": "LOW", "expected_sales_per_hour": 1, "unit_price": 10},
        {"sku": "HIGH", "expected_sales_per_hour": 10, "unit_price": 10},
    ]
    rows = agg.replenishment_list(planogram=planogram)
    assert rows[0]["sku"] == "HIGH"


def test_queue_status_latest_per_lane(agg):
    agg.db.insert("queue_estimate", {"run_id": "r1", "t": 1.0, "lane": 1,
                                     "count": 2.0, "pred_wait_s": 10.0})
    agg.db.insert("queue_estimate", {"run_id": "r1", "t": 2.0, "lane": 1,
                                     "count": 5.0, "pred_wait_s": 30.0})
    agg.db.drain()
    rows = agg.queue_status()
    assert rows[0]["count"] == 5.0


def test_wait_pred_vs_actual_mae(agg):
    agg.db.insert("queue_estimate", {"run_id": "r1", "t": 10.0, "lane": 1,
                                     "count": 2.0, "pred_wait_s": 100.0})
    agg.db.insert("queue_truth", {"clip": "q", "t": 10.0, "lane": 1,
                                  "count": 2, "actual_wait_s": 120.0})
    agg.db.drain()
    out = agg.wait_pred_vs_actual(0, 100)
    assert out["n"] == 1
    assert out["mae_s"] == pytest.approx(20.0)


def test_alerts_open_only(agg):
    agg.db.insert("alert", {"run_id": "r1", "t": 1.0, "severity": "warning",
                            "rule": "r", "message": "m", "acked": 0})
    agg.db.insert("alert", {"run_id": "r1", "t": 2.0, "severity": "info",
                            "rule": "r2", "message": "m2", "acked": 1})
    agg.db.drain()
    assert len(agg.alerts(open_only=True)) == 1
    assert len(agg.alerts(open_only=False)) == 2


def test_stores_returns_distinct_store_ids(agg):
    agg.db.insert("run", {"run_id": "r2", "store_id": "demo-02", "started": 0.0, "clipset": "sim"})
    agg.db.drain()
    assert set(agg.stores()) == {"demo-01", "demo-02"}
