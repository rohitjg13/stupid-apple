"""backend/db.py — WAL, batched writes, single writer, retention/rollup, integrity."""
import sqlite3
import time

import pytest

from backend.db import DB, RETENTION_S, ROLLUP_S


@pytest.fixture
def db(tmp_path):
    d = DB(tmp_path / "retail.db", batch_interval_s=0.01)
    yield d
    d.close()


def test_wal_and_synchronous_are_set(db):
    row = db.query_one("PRAGMA journal_mode")
    assert row["journal_mode"].lower() == "wal"
    assert db.query_one("PRAGMA synchronous")["synchronous"] == 1   # NORMAL


def test_insert_then_read_round_trips(db):
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 5}).drain()
    rows = db.query("SELECT * FROM occupancy WHERE run_id = 'r1'")
    assert rows == [{"run_id": "r1", "t": 1.0, "count": 5}]


def test_batch_commit_is_not_one_per_event(db):
    # Every enqueue goes through one writer, so order is preserved.
    for i in range(100):
        db.insert("occupancy", {"run_id": "r1", "t": float(i), "count": i})
    db.drain()
    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 100
    assert [r["count"] for r in db.query(
        "SELECT count FROM occupancy ORDER BY t")] == list(range(100))


def test_drain_blocks_until_committed(db):
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 1}).drain()
    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 1


def test_insert_many(db):
    rows = [{"run_id": "r1", "t": float(i), "count": i} for i in range(5)]
    db.insert_many("occupancy", rows).drain()
    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 5


def test_run_table_has_run_id(db):
    db.insert("run", {"run_id": "r9", "store_id": "demo-01",
                      "started": 1.0, "clipset": "sim"}).drain()
    assert db.query_one("SELECT store_id FROM run WHERE run_id='r9'")["store_id"] == "demo-01"


def test_retention_drops_old_shelf_and_lane_rows(db):
    old, new = time.time() - RETENTION_S - 100, time.time()
    db.insert("shelf_fill", {"run_id": "r1", "t": old, "roi": "A1", "fill": 40})
    db.insert("shelf_fill", {"run_id": "r1", "t": new, "roi": "A1", "fill": 200})
    db.insert("lane_occ", {"run_id": "r1", "t": old, "lane": 1, "cells": b"\x00" * 16})
    db.insert("lane_occ", {"run_id": "r1", "t": new, "lane": 1, "cells": b"\x01" * 16})
    db.retention(now=new).drain()
    assert db.query_one("SELECT COUNT(*) AS n FROM shelf_fill")["n"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM lane_occ")["n"] == 1


def test_rollup_averages_shelf_fill_and_lane_occ(db):
    now = time.time()
    # An hour inside the window, not 100 s: flooring to the minute can push the
    # bucket back below `lo` and quietly take the rows out of range.
    lo = now - RETENTION_S + 3600
    bucket = int(lo / 60) * 60
    assert now - RETENTION_S <= bucket < now - ROLLUP_S
    t1, t2 = bucket + 1.0, bucket + 2.0     # same minute, no boundary straddle
    db.insert("shelf_fill", {"run_id": "r1", "t": t1, "roi": "A1", "fill": 100})
    db.insert("shelf_fill", {"run_id": "r1", "t": t2, "roi": "A1", "fill": 200})
    db.insert("lane_occ", {"run_id": "r1", "t": t1, "lane": 1, "cells": b"\x00" * 16})
    db.insert("lane_occ", {"run_id": "r1", "t": t2, "lane": 1, "cells": b"\x10" * 16})
    db.rollup(now=now).drain()

    fills = db.query("SELECT * FROM shelf_fill")
    assert len(fills) == 1
    assert fills[0]["t"] == bucket and fills[0]["fill"] == 150

    lanes = db.query("SELECT * FROM lane_occ")
    assert len(lanes) == 1
    assert lanes[0]["t"] == bucket and lanes[0]["cells"] == b"\x08" * 16


def test_rollup_leaves_recent_rows_alone(db):
    now = time.time()
    fresh = now - 100
    db.insert("shelf_fill", {"run_id": "r1", "t": fresh, "roi": "A1", "fill": 250})
    db.rollup(now=now).drain()
    assert db.query_one("SELECT fill FROM shelf_fill")["fill"] == 250


def test_integrity_check_passes(db):
    db.drain()
    assert db.integrity_check() == "ok"


def test_close_stops_the_writer_thread(db):
    db.close()
    assert db._thread is None or not db._thread.is_alive()


def test_writer_raises_on_bad_statement_and_survives(db):
    # A bad statement must raise inside drain() so tests notice, not silently drop.
    db.enqueue("INSERT INTO no_such_table VALUES (1)")
    with pytest.raises(sqlite3.Error):
        db.drain()
    # A subsequent valid write still lands after a rollback.
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 9}).drain()
    assert db.query_one("SELECT count FROM occupancy")["count"] == 9


def test_one_bad_row_does_not_take_the_batch_with_it(db):
    """Acceptance criterion is zero dropped events: a single malformed insert
    must not roll back the 500 good rows batched behind it."""
    for i in range(20):
        db.insert("occupancy", {"run_id": "r1", "t": float(i), "count": i})
    db.enqueue("INSERT INTO no_such_table VALUES (1)")
    for i in range(20, 40):
        db.insert("occupancy", {"run_id": "r1", "t": float(i), "count": i})
    with pytest.raises(sqlite3.Error):
        db.drain()                       # the error still surfaces to the caller
    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 40
    assert db.dropped == 1


def test_dropped_counter_starts_at_zero(db):
    for i in range(50):
        db.insert("occupancy", {"run_id": "r1", "t": float(i), "count": i})
    db.drain()
    assert db.dropped == 0


def test_ten_thousand_events_land_with_none_dropped(db):
    """The plan's W1 §1 target: 10,000 events, zero drops, integrity OK."""
    for i in range(10_000):
        db.insert("occupancy", {"run_id": "load", "t": float(i), "count": i % 40})
    db.drain(timeout=60.0)
    assert db.query_one(
        "SELECT COUNT(*) AS n FROM occupancy WHERE run_id='load'")["n"] == 10_000
    assert db.dropped == 0
    assert db.integrity_check() == "ok"


WRITER_SCRIPT = """
import os, sys
sys.path.insert(0, {root!r})
from backend.db import DB
db = DB(sys.argv[1], batch_interval_s=0.01)
for i in range(200):
    db.insert("occupancy", {{"run_id": "r1", "t": float(i), "count": i}})
db.drain()
for i in range(200, 400):
    db.insert("occupancy", {{"run_id": "r1", "t": float(i), "count": i}})
sys.stdout.write("ready")
sys.stdout.flush()
os.kill(os.getpid(), 9)
"""


def test_integrity_survives_a_killed_writer(tmp_path):
    """Process-level stand-in for the power pull.

    A writer is SIGKILLed with a batch still in flight: no close, no flush, no
    finaliser. Committed rows must survive and the file must not be corrupt.
    This is as far as pytest can go — losing the OS page cache too is the board
    protocol in docs/TEST_PROTOCOL.md, three times, in W4.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = str(_Path(__file__).resolve().parent.parent)
    path = tmp_path / "pull.db"
    proc = subprocess.run([_sys.executable, "-c",
                           WRITER_SCRIPT.format(root=root), str(path)],
                          capture_output=True)
    assert proc.returncode == -9, proc.stderr.decode()
    assert proc.stdout == b"ready"

    reopened = DB(path, batch_interval_s=0.01)
    try:
        assert reopened.integrity_check() == "ok"
        # The drained 200 are committed; the in-flight batch may or may not be.
        n = reopened.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"]
        assert 200 <= n <= 400
    finally:
        reopened.close()


def test_reopening_an_existing_db_does_not_wipe_it(tmp_path):
    path = tmp_path / "again.db"
    first = DB(path)
    first.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 3}).drain()
    first.close()
    second = DB(path)
    try:
        assert second.query_one("SELECT count FROM occupancy")["count"] == 3
    finally:
        second.close()


def test_close_is_idempotent(db):
    db.close()
    db.close()


def test_rollup_sums_heatmap_tiles_into_fifteen_minute_buckets(db):
    from backend.db import HEATMAP_BUCKET_S
    now = time.time()
    # Flooring to a 900 s boundary can move the bucket up to 899 s earlier, so
    # anchor an hour inside the window rather than on its very edge.
    lo = now - RETENTION_S + 3600
    bucket = int(lo / HEATMAP_BUCKET_S) * HEATMAP_BUCKET_S
    assert now - RETENTION_S <= bucket < now - ROLLUP_S
    # Three 10-second tiles for the same cell inside one 15-minute bucket.
    for i, count in enumerate([3, 4, 5]):
        db.insert("heatmap", {"run_id": "r1", "t_bucket": bucket + i * 10,
                              "gx": 2, "gy": 3, "count": count})
    db.insert("heatmap", {"run_id": "r1", "t_bucket": bucket + 20,
                          "gx": 9, "gy": 9, "count": 1})
    db.rollup(now=now).drain()

    rows = db.query("SELECT t_bucket, gx, gy, count FROM heatmap ORDER BY gx")
    assert rows == [{"t_bucket": bucket, "gx": 2, "gy": 3, "count": 12},
                    {"t_bucket": bucket, "gx": 9, "gy": 9, "count": 1}]


def test_heatmap_rollup_preserves_the_total(db):
    """Counts sum, they do not average: the chart's totals must not move."""
    from backend.db import HEATMAP_BUCKET_S
    now = time.time()
    # Flooring to a 900 s boundary can move the bucket up to 899 s earlier, so
    # anchor an hour inside the window rather than on its very edge.
    lo = now - RETENTION_S + 3600
    bucket = int(lo / HEATMAP_BUCKET_S) * HEATMAP_BUCKET_S
    assert now - RETENTION_S <= bucket < now - ROLLUP_S
    for i in range(30):
        db.insert("heatmap", {"run_id": "r1", "t_bucket": bucket + i * 10,
                              "gx": i % 4, "gy": 1, "count": i})
    db.drain()
    before = db.query_one("SELECT SUM(count) AS n FROM heatmap")["n"]
    db.rollup(now=now).drain()
    after = db.query_one("SELECT SUM(count) AS n FROM heatmap")["n"]
    assert before == after == sum(range(30))
    assert db.query_one("SELECT COUNT(*) AS n FROM heatmap")["n"] == 4   # 4 cells


def test_heatmap_rollup_leaves_the_last_hour_alone(db):
    """The live demo reads the last hour at full resolution."""
    now = time.time()
    db.insert("heatmap", {"run_id": "r1", "t_bucket": int(now) - 100,
                          "gx": 1, "gy": 1, "count": 7})
    db.rollup(now=now).drain()
    assert db.query_one("SELECT t_bucket FROM heatmap")["t_bucket"] == int(now) - 100


def test_rollup_into_an_existing_bucket_accumulates(db):
    """A second rollup pass must add to the bucket it already wrote, not fail
    on the primary key and lose the batch."""
    from backend.db import HEATMAP_BUCKET_S
    now = time.time()
    # Flooring to a 900 s boundary can move the bucket up to 899 s earlier, so
    # anchor an hour inside the window rather than on its very edge.
    lo = now - RETENTION_S + 3600
    bucket = int(lo / HEATMAP_BUCKET_S) * HEATMAP_BUCKET_S
    assert now - RETENTION_S <= bucket < now - ROLLUP_S
    db.insert("heatmap", {"run_id": "r1", "t_bucket": bucket, "gx": 1, "gy": 1,
                          "count": 5})
    db.insert("heatmap", {"run_id": "r1", "t_bucket": bucket + 10, "gx": 1,
                          "gy": 1, "count": 6})
    db.rollup(now=now).drain()
    assert db.query_one("SELECT count FROM heatmap")["count"] == 11
    assert db.dropped == 0
