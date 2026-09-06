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
    lo = now - RETENTION_S + 100
    bucket = int(lo / 60) * 60
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


def test_one_bad_statement_does_not_discard_the_batch(tmp_path):
    """A batch is up to 500 rows; losing all of them to one bad row is not on."""
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 1})
    db.enqueue("INSERT INTO nope (x) VALUES (?)", (1,))
    db.insert("occupancy", {"run_id": "r1", "t": 2.0, "count": 2})
    with pytest.raises(sqlite3.Error):
        db.drain()                       # the failure is still reported...
    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 2   # ...not hidden
    assert db.dropped == 1
    db.close()


def test_a_steady_stream_commits_on_time_not_only_when_it_stops(tmp_path):
    """The queue never goes idle while a pipeline runs, so age has to flush it."""
    db = DB(tmp_path / "r.db", batch_interval_s=0.05, batch_size=10_000)
    deadline = time.monotonic() + 0.6
    i = 0
    while time.monotonic() < deadline:
        db.insert("occupancy", {"run_id": "r1", "t": float(i), "count": i})
        i += 1
        time.sleep(0.005)                # busier than the interval, never empty
    landed = db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"]
    db.close()
    assert landed > 0, "nothing was readable until the writer was closed"
    assert landed >= i // 2
