"""backend/sink.py — every event maps to exactly one insert, none dropped."""
import time

import pytest

from backend.db import DB
from backend.sink import Sink, pack_cells
from core.events import Event


def ev(t, event_type, payload, stream="-", zone_id=None):
    return Event(t, "demo-01", stream, zone_id, event_type, payload)


@pytest.fixture
def sink(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    s = Sink(db, run_id="run-1")
    yield s
    db.close()


def test_pack_cells_is_16_bytes_and_clamped():
    assert pack_cells([0] * 16) == b"\x00" * 16
    assert pack_cells([100] * 16) == bytes([100] * 16)
    assert pack_cells([255, 256, -1] + [0] * 13) == bytes([255, 0, 255] + [0] * 13)
    with pytest.raises(ValueError):
        pack_cells([0] * 15)


def test_tripwire(sink):
    sink.on_event(ev(1.0, "tripwire", {"dir": "in", "tripwire_id": "door"}, "overhead", "door"))
    sink.drain()
    assert sink.db.query("SELECT t, dir FROM tripwire") == [{"t": 1.0, "dir": "in"}]


def test_occupancy(sink):
    sink.on_event(ev(2.0, "occupancy", {"count": 7}, "overhead"))
    sink.drain()
    assert sink.db.query_one("SELECT count FROM occupancy")["count"] == 7


def test_visit_stores_dwell_derived_column(sink):
    sink.on_event(ev(0, "visit", {"track_id": 4, "t_enter": 10.0, "t_exit": 25.0},
                     "overhead", "aisle_a"))
    sink.drain()
    row = sink.db.query_one("SELECT * FROM visit")
    assert row["track_id"] == 4 and row["zone"] == "aisle_a"
    assert row["dwell_s"] == pytest.approx(15.0)


def test_heatmap_expands_tiles(sink):
    sink.on_event(ev(0, "heatmap", {"t_bucket": 5, "tiles": [[1, 2, 3], [4, 5, 6]]}, "overhead"))
    sink.drain()
    assert sink.db.query_one("SELECT COUNT(*) AS n FROM heatmap")["n"] == 2


def test_shelf_fill(sink):
    sink.on_event(ev(3.0, "shelf_fill", {"fill": 120}, "shelf", "A1"))
    sink.drain()
    assert sink.db.query_one("SELECT roi, fill FROM shelf_fill") == {"roi": "A1", "fill": 120}


def test_stockout_start_then_end_closes_one_row(sink):
    sink.on_event(ev(10.0, "stockout_start", {"sku": "MAGGI-70G", "facing": "A1"}, "shelf", "A1"))
    sink.on_event(ev(130.0, "stockout_end", {"sku": "MAGGI-70G", "facing": "A1"}, "shelf", "A1"))
    sink.drain()
    rows = sink.db.query("SELECT * FROM stockout")
    assert len(rows) == 1
    assert rows[0]["t_start"] == 10.0 and rows[0]["t_end"] == 130.0


def test_planogram_violation(sink):
    sink.on_event(ev(4.0, "planogram_violation",
                     {"expected_sku": "MAGGI-70G", "observed": "empty"}, "shelf", "A1"))
    sink.drain()
    assert sink.db.query_one("SELECT roi, observed FROM planogram_violation") == \
        {"roi": "A1", "observed": "empty"}


def test_lane_occ_packs_blob(sink):
    sink.on_event(ev(5.0, "lane_occ", {"lane": 2, "cells": list(range(16))}, "overhead"))
    sink.drain()
    row = sink.db.query_one("SELECT lane, cells FROM lane_occ")
    assert row["lane"] == 2 and row["cells"] == bytes(range(16))


def test_queue_estimate(sink):
    sink.on_event(ev(6.0, "queue_estimate", {"lane": 1, "count": 3.0, "pred_wait_s": 90.0}, "overhead"))
    sink.drain()
    assert sink.db.query_one("SELECT count, pred_wait_s FROM queue_estimate") == \
        {"count": 3.0, "pred_wait_s": 90.0}


def test_pos_txn_allows_null_zone_hint(sink):
    sink.on_event(ev(7.0, "pos_txn", {"lane": 1, "items": 3, "amount": 42.0, "zone_hint": None}))
    sink.drain()
    assert sink.db.query_one("SELECT items, zone_hint FROM pos_txn")["items"] == 3


def test_alert(sink):
    sink.on_event(ev(8.0, "alert", {"severity": "warning", "rule": "stockout",
                                    "message": "A1 empty"}), )
    sink.drain()
    assert sink.db.query_one("SELECT severity, rule FROM alert") == \
        {"severity": "warning", "rule": "stockout"}


def test_all_types_round_trip_through_the_bus(tmp_path):
    from core.bus import Bus

    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    sink = Sink(db, "run-1")
    bus = Bus()
    sink.subscribe(bus)

    events = [
        ev(1.0, "occupancy", {"count": 1}, "overhead"),
        ev(1.0, "shelf_fill", {"fill": 100}, "shelf", "A1"),
        ev(1.0, "queue_estimate", {"lane": 1, "count": 2.0, "pred_wait_s": 5.0}, "overhead"),
        ev(1.0, "alert", {"severity": "info", "rule": "r", "message": "m"}),
    ]
    for e in events:
        bus.publish(e)
    bus.drain()
    sink.drain()

    assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM shelf_fill")["n"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM queue_estimate")["n"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM alert")["n"] == 1
    db.close()


def test_no_drops_under_load(sink):
    for i in range(500):
        sink.on_event(ev(float(i), "occupancy", {"count": i}, "overhead"))
    sink.drain()
    assert sink.db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 500


def test_repeated_heatmap_buckets_replace_rather_than_collide(sink):
    """shopper/heatmap.py publishes the running total every 10 s but buckets by
    the minute, so the same primary key arrives six times a bucket."""
    sink.on_event(ev(0, "heatmap", {"t_bucket": 5, "tiles": [[1, 2, 3]]}, "overhead"))
    sink.on_event(ev(10, "heatmap", {"t_bucket": 5, "tiles": [[1, 2, 9]]}, "overhead"))
    sink.db.drain()
    rows = sink.db.query("SELECT gx, gy, count FROM heatmap")
    assert rows == [{"gx": 1, "gy": 2, "count": 9}]


def test_a_colliding_heatmap_row_does_not_lose_the_events_around_it(sink):
    """One integrity error used to roll back the whole 500-row batch."""
    sink.on_event(ev(1, "tripwire", {"dir": "in", "tripwire_id": "door"}, "overhead"))
    sink.on_event(ev(2, "heatmap", {"t_bucket": 1, "tiles": [[0, 0, 1]]}, "overhead"))
    sink.on_event(ev(3, "heatmap", {"t_bucket": 1, "tiles": [[0, 0, 4]]}, "overhead"))
    sink.on_event(ev(4, "occupancy", {"count": 7}, "overhead"))
    sink.db.drain()
    assert sink.db.query_one("SELECT COUNT(*) AS n FROM tripwire")["n"] == 1
    assert sink.db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 1
