"""Every dashboard query must ride an index, not scan the table.

Timing assertions would be flaky and would say nothing about the board, which
is roughly an order of magnitude slower than a laptop. `EXPLAIN QUERY PLAN` is
the durable check: it is the same plan on both machines, and a full SCAN of
`occupancy` is what turns a 3 ms tile into a 300 ms one once a demo day and the
7-day backfill share a file.

The plan's acceptance criterion is "all aggregates < 200 ms on the board".
"""
import pytest

from backend.aggregates import Aggregates
from backend.db import DB

# Tables that grow with time. A scan of any of these is the bug this catches;
# small lookup tables (run, stockout, sync_state) are fine to scan.
BIG_TABLES = {"occupancy", "tripwire", "visit", "heatmap", "shelf_fill",
              "lane_occ", "queue_estimate", "pos_txn", "kpi_15m"}


@pytest.fixture
def db(tmp_path):
    d = DB(tmp_path / "plans.db", batch_interval_s=0.01)
    d.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0,
                     "clipset": "sim"})
    d.drain()
    yield d
    d.close()


def plan(db, sql, params=()):
    return [r["detail"] for r in
            db.query("EXPLAIN QUERY PLAN " + sql, params)]


def assert_no_big_scan(steps, sql):
    bad = [s for s in steps
           if s.startswith("SCAN") and any(t in s for t in BIG_TABLES)]
    assert not bad, f"full scan in {sql!r}: {steps}"


# The queries as `aggregates.py` issues them, in the same shape.
QUERIES = {
    "live occupancy": (
        "SELECT t, count FROM occupancy WHERE run_id = ? ORDER BY t DESC LIMIT 1",
        ("r1",)),
    "live entries": (
        "SELECT COUNT(*) AS n FROM tripwire WHERE run_id = ? AND dir = 'in' "
        "AND t >= ?", ("r1", 0.0)),
    "live alerts": (
        "SELECT COUNT(*) AS n FROM alert WHERE run_id = ? AND acked = 0", ("r1",)),
    "live sales": (
        "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS amount FROM pos_txn "
        "WHERE run_id = ? AND t >= ?", ("r1", 0.0)),
    "footfall_by_hour": (
        "SELECT CAST((t - ?) / 3600 AS INTEGER) AS h, COUNT(*) AS n FROM tripwire "
        "WHERE run_id = ? AND dir = 'in' AND t >= ? AND t < ? GROUP BY h ORDER BY h",
        (0.0, "r1", 0.0, 86400.0)),
    "footfall_by_day": (
        "SELECT COUNT(*) AS n FROM tripwire WHERE run_id = ? AND dir = 'in' "
        "AND t >= ? AND t < ?", ("r1", 0.0, 86400.0)),
    "dwell_by_zone": (
        "SELECT zone, dwell_s FROM visit WHERE run_id = ? AND t_enter >= ? "
        "AND t_enter < ? ORDER BY zone", ("r1", 0.0, 86400.0)),
    "littles law occupancy": (
        "SELECT AVG(count) AS l, COUNT(*) AS n FROM occupancy WHERE run_id = ? "
        "AND t >= ? AND t < ?", ("r1", 0.0, 86400.0)),
    "heatmap": (
        "SELECT gx, gy, SUM(count) AS count FROM heatmap WHERE run_id = ? "
        "AND t_bucket >= ? AND t_bucket < ? GROUP BY gx, gy", ("r1", 0, 86400)),
    "queue lanes": (
        "SELECT DISTINCT lane FROM queue_estimate WHERE run_id = ? ORDER BY lane",
        ("r1",)),
    "queue latest per lane": (
        "SELECT lane, t, count, pred_wait_s FROM queue_estimate WHERE run_id = ? "
        "AND lane = ? ORDER BY t DESC LIMIT 1", ("r1", 1)),
    "queue predictions": (
        "SELECT t, lane, count, pred_wait_s FROM queue_estimate WHERE run_id = ? "
        "AND t >= ? AND t < ? ORDER BY t", ("r1", 0.0, 86400.0)),
    "conversion txns": (
        "SELECT t, zone_hint FROM pos_txn WHERE run_id = ? AND t >= ? AND t < ?",
        ("r1", 0.0, 86400.0)),
    "conversion fills": (
        "SELECT t, roi, fill FROM shelf_fill WHERE run_id = ? AND t >= ? "
        "AND t < ? ORDER BY t", ("r1", 0.0, 86400.0)),
    "kpi series": (
        "SELECT t_bucket, key, value FROM kpi_15m WHERE run_id = ? "
        "AND t_bucket >= ? AND t_bucket < ? ORDER BY t_bucket", ("r1", 0, 86400)),
    "retention shelf_fill": ("DELETE FROM shelf_fill WHERE t < ?", (0.0,)),
    "retention lane_occ": ("DELETE FROM lane_occ WHERE t < ?", (0.0,)),
}


@pytest.mark.parametrize("name", sorted(QUERIES))
def test_query_uses_an_index(db, name):
    sql, params = QUERIES[name]
    assert_no_big_scan(plan(db, sql, params), name)


def test_every_aggregate_runs_without_a_scan(db, tmp_path):
    """Belt and braces: run the real functions and assert they return, so a
    query that changed shape cannot drift away from the list above unnoticed."""
    agg = Aggregates(db, run_id="r1", planogram=[])
    agg.live_summary()
    agg.footfall_by_hour(0.0)
    agg.footfall_by_day(7, now=86400.0)
    agg.dwell_by_zone(0, 86400)
    agg.dwell_littles_law(0, 86400)
    agg.heatmap(0, 86400)
    agg.stockout_durations(0, 86400)
    agg.restock_stats(0, 86400)
    agg.replenishment_list()
    agg.queue_status()
    agg.wait_pred_vs_actual(0, 86400)
    agg.conversion_by_zone(0, 86400)
    agg.alerts()
    agg.store_details()
    agg.kpi_series(0, 86400)


def test_the_composite_indexes_are_actually_created(db):
    names = {r["name"] for r in
             db.query("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"idx_occupancy_run_t", "idx_tripwire_run_t", "idx_visit_run_enter",
            "idx_queue_est_run", "idx_pos_txn_run_t", "idx_alert_run",
            "idx_shelf_fill_run_t", "idx_lane_occ_run_t"} <= names


def test_superseded_single_column_indexes_are_gone(db):
    """Every redundant index is write amplification on an SD card."""
    names = {r["name"] for r in
             db.query("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert not ({"idx_occupancy_t", "idx_tripwire_t", "idx_visit_enter",
                 "idx_queue_estimate_t", "idx_pos_txn_t", "idx_alert_t"} & names)


def test_reapplying_the_schema_is_a_no_op(tmp_path):
    """schema.sql runs on every open, including files written by older code."""
    from backend.db import SCHEMA
    path = tmp_path / "again.db"
    first = DB(path)
    first.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 3}).drain()
    first.close()
    second = DB(path, schema=SCHEMA)
    try:
        assert second.query_one("SELECT count FROM occupancy")["count"] == 3
        assert second.integrity_check() == "ok"
    finally:
        second.close()
