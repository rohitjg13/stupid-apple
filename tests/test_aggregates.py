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


# ---- Little's law (plan W2 §1, and an acceptance criterion) -----------------

def test_littles_law_on_a_stream_with_a_known_answer(agg):
    """L = 10 people, lambda = 5 entries / 100 s -> W = L/lambda = 200 s."""
    for i in range(10):
        agg.db.insert("occupancy", {"run_id": "r1", "t": float(i * 10), "count": 10})
    for i in range(5):
        agg.db.insert("tripwire", {"run_id": "r1", "t": float(i * 10), "dir": "in"})
    agg.db.drain()

    out = agg.dwell_littles_law(0.0, 100.0)
    assert out["mean_occupancy"] == pytest.approx(10.0)
    assert out["entries"] == 5
    assert out["lambda_per_s"] == pytest.approx(0.05)
    assert out["mean_dwell_s"] == pytest.approx(200.0)


def test_littles_law_varying_occupancy(agg):
    """L is the mean of the samples: (2+4+6+8)/4 = 5; lambda = 2/40 -> W = 100."""
    for i, c in enumerate([2, 4, 6, 8]):
        agg.db.insert("occupancy", {"run_id": "r1", "t": float(i * 10), "count": c})
    agg.db.insert("tripwire", {"run_id": "r1", "t": 1.0, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": 2.0, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": 3.0, "dir": "out"})   # ignored
    agg.db.drain()

    out = agg.dwell_littles_law(0.0, 40.0)
    assert out["mean_occupancy"] == pytest.approx(5.0)
    assert out["mean_dwell_s"] == pytest.approx(100.0)


def test_littles_law_without_entries_is_none_not_infinity(agg):
    agg.db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 4})
    agg.db.drain()
    out = agg.dwell_littles_law(0.0, 100.0)
    assert out["mean_dwell_s"] is None and out["entries"] == 0


def test_littles_law_on_an_empty_window(agg):
    out = agg.dwell_littles_law(0.0, 100.0)
    assert out["mean_occupancy"] is None and out["mean_dwell_s"] is None


# ---- stock-out economics ---------------------------------------------------

PLANOGRAM = [{"sku": "MAGGI-70G", "name": "Maggi", "unit_price": 14,
              "expected_sales_per_hour": 6}]


def test_lost_revenue_is_hours_times_rate_times_price(tmp_path):
    from backend.aggregates import Aggregates
    from backend.db import DB
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "MAGGI-70G",
                           "t_start": 0.0, "t_end": 3600.0}).drain()
    a = Aggregates(db, run_id="r1", planogram=PLANOGRAM)
    # one hour empty x 6 sales/h x Rs 14 = Rs 84
    assert a.stockout_durations(0, 7200)[0]["lost_revenue"] == pytest.approx(84.0)
    db.close()


def test_lost_revenue_is_zero_for_an_unpriced_sku(agg):
    agg.db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "UNKNOWN",
                               "t_start": 0.0, "t_end": 3600.0}).drain()
    assert agg.stockout_durations(0, 7200)[0]["lost_revenue"] == 0.0


def test_restock_stats(tmp_path):
    from backend.aggregates import Aggregates
    from backend.db import DB
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    for start, end in [(0.0, 60.0), (100.0, 220.0), (300.0, 600.0)]:
        db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "MAGGI-70G",
                               "t_start": start, "t_end": end})
    db.drain()
    a = Aggregates(db, run_id="r1", planogram=PLANOGRAM)
    s = a.restock_stats(0, 1000)
    assert s["n"] == 3
    assert s["median_s"] == pytest.approx(120.0)   # durations 60, 120, 300
    assert s["p90_s"] == pytest.approx(300.0)
    assert s["mean_s"] == pytest.approx(160.0)
    # 480 s empty in total = 0.1333 h x 84 /h
    assert s["total_lost_revenue"] == pytest.approx(11.2, abs=0.05)
    db.close()


def test_restock_stats_empty(agg):
    assert agg.restock_stats(0, 100)["n"] == 0


def test_replenishment_reports_time_empty_and_cost_so_far(tmp_path):
    from backend.aggregates import Aggregates
    from backend.db import DB
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "MAGGI-70G",
                           "t_start": 100.0, "t_end": None}).drain()
    a = Aggregates(db, run_id="r1", planogram=PLANOGRAM)
    row = a.replenishment_list(now=3700.0)[0]
    assert row["empty_s"] == pytest.approx(3600.0)
    assert row["revenue_rate"] == pytest.approx(84.0)
    assert row["lost_revenue_so_far"] == pytest.approx(84.0)
    assert row["name"] == "Maggi"
    db.close()


def test_replenishment_uses_clip_time_not_wall_clock(agg):
    """A replayed clip lives in its own time base; wall-clock 'now' would give
    a 55-year-old stock-out."""
    agg.db.insert("occupancy", {"run_id": "r1", "t": 500.0, "count": 1})
    agg.db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "X",
                               "t_start": 200.0, "t_end": None})
    agg.db.drain()
    assert agg.replenishment_list()[0]["empty_s"] == pytest.approx(300.0)


# ---- footfall by local day -------------------------------------------------

def test_footfall_by_day_zero_fills_and_is_oldest_first(agg):
    now = time.time()
    agg.db.insert("tripwire", {"run_id": "r1", "t": now, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": now - 2 * 86400, "dir": "in"})
    agg.db.drain()
    rows = agg.footfall_by_day(3, now=now)
    assert len(rows) == 3
    assert [r["entries"] for r in rows] == [1, 0, 1]
    assert rows[0]["day_start"] < rows[-1]["day_start"]
    assert all(len(r["day"]) == 10 for r in rows)     # ISO YYYY-MM-DD labels


def test_footfall_by_day_excludes_exits(agg):
    now = time.time()
    agg.db.insert("tripwire", {"run_id": "r1", "t": now, "dir": "out"})
    agg.db.drain()
    assert agg.footfall_by_day(1, now=now)[0]["entries"] == 0


# ---- queue truth join ------------------------------------------------------

def test_wait_pred_vs_actual_matches_within_a_lane(agg):
    """A lane-2 truth sample must not be scored against a lane-1 prediction."""
    agg.db.insert("queue_estimate", {"run_id": "r1", "t": 10.0, "lane": 1,
                                     "count": 2.0, "pred_wait_s": 100.0})
    agg.db.insert("queue_truth", {"clip": "q", "t": 10.0, "lane": 2,
                                  "count": 9, "actual_wait_s": 999.0})
    agg.db.drain()
    assert agg.wait_pred_vs_actual(0, 100)["n"] == 0


def test_wait_pred_vs_actual_picks_the_nearest_in_lane(agg):
    for t, wait in [(10.0, 100.0), (20.0, 200.0)]:
        agg.db.insert("queue_estimate", {"run_id": "r1", "t": t, "lane": 1,
                                         "count": 2.0, "pred_wait_s": wait})
    agg.db.insert("queue_truth", {"clip": "q", "t": 19.0, "lane": 1,
                                  "count": 3, "actual_wait_s": 190.0})
    agg.db.drain()
    out = agg.wait_pred_vs_actual(0, 100)
    assert out["n"] == 1
    assert out["mae_s"] == pytest.approx(10.0)       # matched the t=20 prediction
    assert out["count_mae"] == pytest.approx(1.0)


def test_wait_pred_vs_actual_honours_a_tolerance(agg):
    agg.db.insert("queue_estimate", {"run_id": "r1", "t": 10.0, "lane": 1,
                                     "count": 2.0, "pred_wait_s": 100.0})
    agg.db.insert("queue_truth", {"clip": "q", "t": 90.0, "lane": 1,
                                  "count": 2, "actual_wait_s": 100.0})
    agg.db.drain()
    assert agg.wait_pred_vs_actual(0, 100, tolerance_s=2.0)["n"] == 0


# ---- alerts and acking -----------------------------------------------------

def test_ack_alert_closes_it(agg):
    agg.db.insert("alert", {"run_id": "r1", "t": 1.0, "severity": "warning",
                            "rule": "stockout", "message": "A1 empty", "acked": 0})
    agg.db.drain()
    alert_id = agg.alerts()[0]["id"]
    row = agg.ack_alert(alert_id)
    assert row["acked"] == 1
    assert agg.alerts(open_only=True) == []
    assert len(agg.alerts(open_only=False)) == 1


def test_ack_alert_is_idempotent_and_404s_on_a_bad_id(agg):
    agg.db.insert("alert", {"run_id": "r1", "t": 1.0, "severity": "info",
                            "rule": "r", "message": "m", "acked": 0})
    agg.db.drain()
    alert_id = agg.alerts()[0]["id"]
    assert agg.ack_alert(alert_id)["acked"] == 1
    assert agg.ack_alert(alert_id)["acked"] == 1
    assert agg.ack_alert(999999) is None


def test_ack_alert_will_not_cross_runs(agg):
    agg.db.insert("alert", {"run_id": "other", "t": 1.0, "severity": "info",
                            "rule": "r", "message": "m", "acked": 0})
    agg.db.drain()
    other_id = agg.db.query_one("SELECT rowid AS id FROM alert")["id"]
    assert agg.ack_alert(other_id) is None


def test_ack_all(agg):
    for i in range(3):
        agg.db.insert("alert", {"run_id": "r1", "t": float(i), "severity": "info",
                                "rule": "r", "message": "m", "acked": 0})
    agg.db.drain()
    assert agg.ack_all()["n"] == 0
    assert agg.alerts() == []


# ---- live summary ----------------------------------------------------------

def test_live_summary_gathers_the_dashboard_tiles(agg):
    now = time.time()
    agg.db.insert("occupancy", {"run_id": "r1", "t": now, "count": 12})
    agg.db.insert("tripwire", {"run_id": "r1", "t": now, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": now, "dir": "out"})
    agg.db.insert("alert", {"run_id": "r1", "t": now, "severity": "warning",
                            "rule": "r", "message": "m", "acked": 0})
    agg.db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "X",
                               "t_start": now, "t_end": None})
    agg.db.insert("pos_txn", {"run_id": "r1", "t": now, "lane": 1, "items": 2,
                              "amount": 40.0, "zone_hint": "aisle_a"})
    agg.db.insert("queue_estimate", {"run_id": "r1", "t": now, "lane": 1,
                                     "count": 3.0, "pred_wait_s": 90.0})
    agg.db.drain()
    s = agg.live_summary()
    assert s["occupancy"] == 12
    assert s["entries_today"] == 1 and s["exits_today"] == 1
    assert s["open_alerts"] == 1 and s["open_stockouts"] == 1
    assert s["txns_today"] == 1 and s["sales_today"] == pytest.approx(40.0)
    assert s["queue_total"] == pytest.approx(3.0)
    assert s["max_pred_wait_s"] == pytest.approx(90.0)
    assert s["run_id"] == "r1"


def test_live_summary_on_an_empty_db_is_all_zeroes(agg):
    s = agg.live_summary()
    assert s["occupancy"] == 0 and s["open_alerts"] == 0
    assert s["lanes"] == [] and s["max_pred_wait_s"] == 0.0


# ---- 15-minute pre-aggregation --------------------------------------------

def test_kpi_rollup_writes_one_row_per_key(agg):
    from backend.aggregates import kpi_bucket
    t0 = 1_700_000_000.0
    b = kpi_bucket(t0)
    for i in range(4):
        agg.db.insert("occupancy", {"run_id": "r1", "t": b + i, "count": (i + 1) * 2})
    for i in range(3):
        agg.db.insert("tripwire", {"run_id": "r1", "t": b + i, "dir": "in"})
    agg.db.insert("tripwire", {"run_id": "r1", "t": b + 5, "dir": "out"})
    agg.db.insert("pos_txn", {"run_id": "r1", "t": b + 6, "lane": 1, "items": 4,
                              "amount": 100.0, "zone_hint": "aisle_a"})
    agg.db.drain()

    assert agg.kpi_rollup(b, b + 900) > 0
    kpi = {r["key"]: r["value"] for r in agg.kpi_series(b, b + 900)}
    assert kpi["occupancy_mean"] == pytest.approx(5.0)     # (2+4+6+8)/4
    assert kpi["occupancy_peak"] == pytest.approx(8.0)
    assert kpi["footfall_in"] == 3 and kpi["footfall_out"] == 1
    assert kpi["pos_txns"] == 1 and kpi["pos_amount"] == pytest.approx(100.0)
    assert kpi["pos_items"] == 4


def test_kpi_rollup_is_idempotent(agg):
    from backend.aggregates import kpi_bucket
    b = kpi_bucket(1_700_000_000.0)
    agg.db.insert("occupancy", {"run_id": "r1", "t": b + 1, "count": 5}).drain()
    agg.kpi_rollup(b, b + 900)
    agg.kpi_rollup(b, b + 900)
    rows = agg.kpi_series(b, b + 900, key="occupancy_mean")
    assert len(rows) == 1 and rows[0]["value"] == pytest.approx(5.0)


def test_kpi_rollup_spans_several_buckets(agg):
    from backend.aggregates import kpi_bucket
    b = kpi_bucket(1_700_000_000.0)
    agg.db.insert("occupancy", {"run_id": "r1", "t": b + 1, "count": 1})
    agg.db.insert("occupancy", {"run_id": "r1", "t": b + 901, "count": 9})
    agg.db.drain()
    agg.kpi_rollup(b, b + 1800)
    rows = agg.kpi_series(b, b + 1800, key="occupancy_mean")
    assert [r["value"] for r in rows] == [pytest.approx(1.0), pytest.approx(9.0)]
    assert [r["t_bucket"] for r in rows] == [b, b + 900]


def test_kpi_bucket_floors_to_15_minutes():
    from backend.aggregates import kpi_bucket
    assert kpi_bucket(0) == 0
    assert kpi_bucket(899.9) == 0
    assert kpi_bucket(900) == 900
    assert kpi_bucket(1801) == 1800


# ---- multi-store -----------------------------------------------------------

def test_store_details_groups_runs_under_stores(agg):
    agg.db.insert("run", {"run_id": "history", "store_id": "demo-02",
                          "started": 5.0, "clipset": "sim"})
    agg.db.insert("run", {"run_id": "history-2", "store_id": "demo-02",
                          "started": 9.0, "clipset": "sim"})
    agg.db.drain()
    details = {d["store_id"]: d for d in agg.store_details()}
    assert set(details) == {"demo-01", "demo-02"}
    assert [r["run_id"] for r in details["demo-02"]["runs"]] == ["history", "history-2"]


def test_runs_are_newest_first(agg):
    agg.db.insert("run", {"run_id": "r2", "store_id": "demo-01", "started": 99.0,
                          "clipset": "sim"}).drain()
    assert agg.runs()[0]["run_id"] == "r2"


# ---- conversion through the database ---------------------------------------

def test_conversion_by_zone_joins_visits_to_transactions(agg):
    agg.db.insert("visit", {"run_id": "r1", "track_id": 1, "zone": "aisle_a",
                            "t_enter": 0.0, "t_exit": 30.0})
    agg.db.insert("visit", {"run_id": "r1", "track_id": 2, "zone": "aisle_b",
                            "t_enter": 0.0, "t_exit": 30.0})
    agg.db.insert("pos_txn", {"run_id": "r1", "t": 40.0, "lane": 1, "items": 1,
                              "amount": 10.0, "zone_hint": "aisle_a"})
    agg.db.drain()
    out = agg.conversion_by_zone(0, 100)
    assert out["aisle_a"]["conversions"] == 1 and out["aisle_a"]["rate"] == 1.0
    assert out["aisle_b"]["conversions"] == 0 and out["aisle_b"]["rate"] == 0.0
    assert out["aisle_b"]["label"] == "ignored"


def test_conversion_by_zone_uses_shelf_drops_too(agg):
    agg.db.insert("visit", {"run_id": "r1", "track_id": 1, "zone": "A1",
                            "t_enter": 0.0, "t_exit": 30.0})
    agg.db.insert("shelf_fill", {"run_id": "r1", "t": 10.0, "roi": "A1", "fill": 200})
    agg.db.insert("shelf_fill", {"run_id": "r1", "t": 20.0, "roi": "A1", "fill": 20})
    agg.db.drain()
    assert agg.conversion_by_zone(0, 100)["A1"]["rate"] == 1.0


def test_conversion_by_zone_maps_hints_through_zone_of(agg):
    agg.db.insert("visit", {"run_id": "r1", "track_id": 1, "zone": "aisle_a",
                            "t_enter": 0.0, "t_exit": 30.0})
    agg.db.insert("pos_txn", {"run_id": "r1", "t": 40.0, "lane": 1, "items": 1,
                              "amount": 10.0, "zone_hint": "A1"})
    agg.db.drain()
    out = agg.conversion_by_zone(0, 100, zone_of=lambda h: {"A1": "aisle_a"}.get(h))
    assert out["aisle_a"]["rate"] == 1.0
