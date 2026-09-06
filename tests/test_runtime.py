"""backend/runtime.py — the backend wired to the bus as one running thing.

These are the tests that would have caught the state this branch was in before:
every backend module existed and was unit-tested, and nothing was ever
instantiated, so `python main.py --source sim` persisted nothing at all.
"""
import numpy as np
import pytest

from backend.db import DB
from backend.runtime import BackendRuntime
from core.bus import Bus
from core.config import load_clipset
from core.events import Event
from pl.contract import FRAME_RESULT_DT, Frame

CFG = load_clipset("config/sim")


def frame(frame_id=0, t_ns=0, stream="overhead", lane_occupancy=None, num_blobs=0):
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    r["frame_id"] = frame_id
    r["num_blobs"] = num_blobs
    if lane_occupancy is not None:
        r["lane_occupancy"] = lane_occupancy
    return Frame(stream, frame_id, t_ns, r)


def ev(t, event_type, payload, stream="-", zone_id=None):
    return Event(t, "demo-01", stream, zone_id, event_type, payload)


@pytest.fixture
def rt(tmp_path):
    bus = Bus(maxsize=4096)
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    runtime = BackendRuntime(CFG, bus, db=db, run_id="test-run", started=1000.0)
    runtime.start(started=1000.0)
    yield runtime
    runtime.close()
    db.close()


def settle(rt):
    """Let the bus subscriber threads catch up, then flush the writer."""
    rt.bus.drain()
    rt.db.drain()


# ---- the run row -----------------------------------------------------------

def test_start_records_the_run(rt):
    settle(rt)
    row = rt.db.query_one("SELECT * FROM run WHERE run_id='test-run'")
    assert row["store_id"] == "demo-01"
    assert row["started"] == 1000.0
    assert row["clipset"] == "sim"


def test_start_is_idempotent(rt):
    before = len(rt.bus._subs)
    rt.start()
    assert len(rt.bus._subs) == before


def test_runtime_needs_a_db_or_a_path():
    with pytest.raises(ValueError):
        BackendRuntime(CFG, Bus())


# ---- lane_occ: the one event derived from raw FrameResult ------------------

def test_on_frame_publishes_lane_occ_per_lane(rt):
    published = rt.on_frame(frame(lane_occupancy=list(range(16))), 1000.0)
    assert [e.payload["lane"] for e in published] == [1, 2]
    assert published[0].payload["cells"] == [0, 1, 2, 3] + [0] * 12
    assert published[1].payload["cells"] == [4, 5, 6, 7] + [0] * 12
    assert all(e.event_type == "lane_occ" for e in published)


def test_lane_occ_reaches_the_database_as_a_blob(rt):
    rt.on_frame(frame(lane_occupancy=[90] * 16), 1000.0)
    settle(rt)
    rows = rt.db.query("SELECT lane, cells FROM lane_occ ORDER BY lane")
    assert len(rows) == 2
    assert rows[0]["cells"] == bytes([90, 90, 90, 90] + [0] * 12)


def test_lane_occ_is_rate_limited_to_two_hertz(rt):
    """The frozen contract says lane_occ is 2 Hz; frames arrive at 15."""
    assert len(rt.on_frame(frame(lane_occupancy=[50] * 16), 1000.0)) == 2
    assert rt.on_frame(frame(lane_occupancy=[50] * 16), 1000.1) == []
    assert len(rt.on_frame(frame(lane_occupancy=[50] * 16), 1000.6)) == 2


def test_shelf_frames_produce_no_lane_occ(rt):
    assert rt.on_frame(frame(stream="shelf", lane_occupancy=[90] * 16), 1000.0) == []


def test_lane_occ_payload_is_a_valid_event(rt):
    from core.events import EVENT_TYPES
    e = rt.on_frame(frame(lane_occupancy=[10] * 16), 1000.0)[0]
    assert set(e.payload) == EVENT_TYPES["lane_occ"]
    assert len(e.payload["cells"]) == 16


# ---- queue estimates -------------------------------------------------------

def test_tick_publishes_queue_estimates_every_two_seconds(rt):
    def estimates(events):
        return [e for e in events if e.event_type == "queue_estimate"]

    rt.on_frame(frame(lane_occupancy=[90] * 16), 1000.0)
    first = estimates(rt.tick(1000.0))
    assert [e.payload["lane"] for e in first] == [1, 2]
    assert estimates(rt.tick(1001.0)) == []         # inside the 2 s interval
    assert [e.payload["lane"] for e in estimates(rt.tick(1002.0))] == [1, 2]


def test_queue_estimates_are_persisted(rt):
    rt.on_frame(frame(lane_occupancy=[90] * 16), 1000.0)
    rt.tick(1000.0)
    settle(rt)
    rows = rt.db.query("SELECT lane, count, pred_wait_s FROM queue_estimate ORDER BY lane")
    assert [r["lane"] for r in rows] == [1, 2]
    assert all(r["count"] > 0 and r["pred_wait_s"] >= 0 for r in rows)


def test_no_lanes_seen_yet_means_no_estimates(rt):
    assert [e for e in rt.tick(1000.0) if e.event_type == "queue_estimate"] == []


# ---- POS -------------------------------------------------------------------

def test_a_shortening_lane_produces_a_transaction(rt):
    rt.on_frame(frame(lane_occupancy=[95] * 16), 1000.0)
    rt.on_frame(frame(lane_occupancy=[0] * 16), 1000.6)      # queue empties
    settle(rt)
    rows = rt.db.query("SELECT lane, items, amount, zone_hint FROM pos_txn")
    assert rows, "a lane dropping by a customer should book a transaction"
    assert all(r["items"] >= 1 and r["amount"] > 0 for r in rows)


def test_idle_lanes_still_transact_so_mu_is_estimable(rt):
    txns = [e for e in rt.tick(1000.0) if e.event_type == "pos_txn"]
    assert [e.payload["lane"] for e in txns] == [1, 2]


def test_transactions_feed_the_service_rate(rt):
    for i, t in enumerate([1000.0, 1020.0, 1040.0]):
        rt.pos.txn(t, 1)
        rt.queue.on_departure(t, 1)
    assert rt.queue.mu(1) == pytest.approx(1 / 20.0)


def test_visits_from_the_bus_steer_the_zone_hint(rt):
    rt.bus.publish(ev(1000.0, "visit", {"track_id": 1, "t_enter": 900.0,
                                        "t_exit": 1000.0}, "overhead", "aisle_b"))
    rt.bus.drain()
    assert rt.pos.txn(1001.0, 1).payload["zone_hint"] == "aisle_b"


# ---- arrivals --------------------------------------------------------------

def test_tripwire_out_counts_as_an_arrival_to_checkout(rt):
    rt.bus.publish(ev(1000.0, "tripwire", {"dir": "out", "tripwire_id": "door"},
                      "overhead", "door"))
    rt.bus.drain()
    assert len(rt.queue._arrivals) == 1


def test_tripwire_in_is_not_an_arrival_to_checkout(rt):
    rt.bus.publish(ev(1000.0, "tripwire", {"dir": "in", "tripwire_id": "door"},
                      "overhead", "door"))
    rt.bus.drain()
    assert rt.queue._arrivals == []


# ---- alerts ----------------------------------------------------------------

def test_stockout_on_the_bus_raises_and_stores_an_alert(rt):
    rt.bus.publish(ev(1000.0, "stockout_start", {"sku": "MAGGI-70G", "facing": "A1"},
                      "shelf", "A1"))
    rt.bus.drain()
    rt.bus.drain()          # the alert the rule published is itself an event
    settle(rt)
    rows = rt.db.query("SELECT severity, rule, message FROM alert")
    assert rows and rows[0]["rule"] == "stockout"
    assert rows[0]["severity"] == "warning"


def test_an_alert_does_not_feed_back_into_the_rules(rt):
    """The rules subscribe to '*' and publish onto the same bus: an alert must
    not be able to trigger another alert."""
    rt.bus.publish(ev(1000.0, "alert", {"severity": "info", "rule": "r",
                                        "message": "m"}))
    rt.bus.drain()
    settle(rt)
    assert rt.db.query_one("SELECT COUNT(*) AS n FROM alert")["n"] == 1


def test_unsynced_clock_raises_an_alert_at_start(tmp_path):
    class FakeClock:
        synced = False

        def now(self):
            return 1000.0

    bus = Bus(maxsize=1024)
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    rt = BackendRuntime(CFG, bus, db=db, run_id="unsynced", clock=FakeClock())
    rt.start(started=1000.0)
    bus.drain()
    db.drain()
    try:
        assert rt.db.query_one("SELECT rule FROM alert")["rule"] == "clock_unsynced"
    finally:
        rt.close()
        db.close()


def test_open_stockout_escalates_on_a_later_tick(rt):
    rt.bus.publish(ev(1000.0, "stockout_start", {"sku": "MAGGI-70G", "facing": "A1"},
                      "shelf", "A1"))
    rt.bus.drain()
    rt.tick(1400.0)                     # more than five minutes later
    rt.bus.drain()
    settle(rt)
    rules = {r["rule"] for r in rt.db.query("SELECT rule FROM alert")}
    assert "stockout_long" in rules


# ---- maintenance -----------------------------------------------------------

def test_kpi_rollup_runs_on_the_maintenance_cadence(rt):
    rt.bus.publish(ev(1000.0, "occupancy", {"count": 5}, "overhead"))
    rt.bus.drain()
    rt.db.drain()
    rt.tick(1000.0)
    assert rt.db.query_one("SELECT COUNT(*) AS n FROM kpi_15m")["n"] > 0


def test_kpi_rollup_does_not_run_on_every_tick(tmp_path):
    bus = Bus(maxsize=1024)
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    rt = BackendRuntime(CFG, bus, db=db, run_id="r", kpi_interval_s=1e9)
    rt.start(started=0.0)
    try:
        rt.tick(0.0)
        n = db.query_one("SELECT COUNT(*) AS n FROM kpi_15m")["n"]
        rt.tick(1.0)
        assert db.query_one("SELECT COUNT(*) AS n FROM kpi_15m")["n"] == n
    finally:
        rt.close()
        db.close()


def test_cloud_sync_is_off_unless_configured(rt):
    assert rt.cloud is None              # config/sim has cloud.enabled false


def test_cloud_sync_uploads_kpi_rows_when_a_url_is_given(tmp_path):
    bus = Bus(maxsize=1024)
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    rt = BackendRuntime(CFG, bus, db=db, run_id="r", cloud_url="http://example",
                        sync_interval_s=0.0)
    sent = []
    rt.cloud._sender = lambda url, data: sent.append(data)
    rt.start(started=0.0)
    try:
        bus.publish(ev(1000.0, "occupancy", {"count": 5}, "overhead"))
        bus.drain()
        db.drain()
        rt.tick(1000.0)
        assert sent, "kpi rows should have been pushed"
        import json
        body = json.loads(sent[0])
        assert body["store_id"] == "demo-01"
        assert {"run_id", "t_bucket", "key", "value"} == set(body["kpi"][0])
    finally:
        rt.close()
        db.close()


def test_cloud_sync_never_uploads_raw_events(tmp_path):
    """The DPDP promise: 15-minute scalars leave the store, nothing else."""
    bus = Bus(maxsize=1024)
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    rt = BackendRuntime(CFG, bus, db=db, run_id="r", cloud_url="http://example",
                        sync_interval_s=0.0)
    sent = []
    rt.cloud._sender = lambda url, data: sent.append(data)
    rt.start(started=0.0)
    try:
        bus.publish(ev(1000.0, "visit", {"track_id": 7, "t_enter": 900.0,
                                         "t_exit": 1000.0}, "overhead", "aisle_a"))
        bus.drain()
        db.drain()
        rt.tick(1000.0)
        blob = b"".join(sent)
        assert sent
        assert b"track_id" not in blob
        assert b"t_enter" not in blob and b"t_exit" not in blob
        assert b"zone_hint" not in blob and b"aisle_a" not in blob

        # Positively: every uploaded row is a scalar from the known KPI set.
        import json
        body = json.loads(blob)
        keys = {r["key"] for r in body["kpi"]}
        assert keys <= {
            "occupancy_mean", "occupancy_peak", "footfall_in", "footfall_out",
            "visits", "dwell_mean_s", "dwell_littles_law_s", "pos_txns",
            "pos_items", "pos_amount", "queue_count_mean", "queue_wait_mean_s",
            "alerts",
        }
        assert all(isinstance(r["value"], (int, float)) for r in body["kpi"])
    finally:
        rt.close()
        db.close()


def test_retention_and_rollup_run_on_their_own_cadence(tmp_path):
    from backend.db import RETENTION_S
    bus = Bus(maxsize=1024)
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    rt = BackendRuntime(CFG, bus, db=db, run_id="r", retention_interval_s=0.0)
    rt.start(started=0.0)
    try:
        now = 10 * 86400.0
        db.insert("shelf_fill", {"run_id": "r", "t": now - RETENTION_S - 10,
                                 "roi": "A1", "fill": 100})
        db.insert("shelf_fill", {"run_id": "r", "t": now, "roi": "A1", "fill": 200})
        db.drain()
        rt.tick(now)
        db.drain()
        assert db.query_one("SELECT COUNT(*) AS n FROM shelf_fill")["n"] == 1
    finally:
        rt.close()
        db.close()


# ---- end to end ------------------------------------------------------------

def test_a_run_of_sim_frames_fills_every_table_the_backend_owns(tmp_path):
    """The integration this branch was missing: frames in, dashboard rows out."""
    from sources.sim import SimSource

    bus = Bus(maxsize=8192)
    db = DB(tmp_path / "e2e.db", batch_interval_s=0.01)
    rt = BackendRuntime(CFG, bus, db=db, run_id="e2e", started=1000.0)
    rt.start(started=1000.0)
    src = SimSource(CFG, stream="overhead", seed=3)
    try:
        t = 1000.0
        for i, f in enumerate(src.frames()):
            t = 1000.0 + i / 15.0
            rt.on_frame(f, t)
            rt.tick(t)
            if i % 15 == 0:
                bus.publish(ev(t, "occupancy",
                               {"count": int(f.result["num_blobs"])}, "overhead"))
            if i >= 450:                 # 30 s of clip time
                break
        bus.drain()
        db.drain()

        for table in ("lane_occ", "queue_estimate", "pos_txn", "occupancy"):
            n = db.query_one(f"SELECT COUNT(*) AS n FROM {table} "
                             "WHERE run_id='e2e'")["n"]
            assert n > 0, f"{table} is empty after a 30 s sim run"
        assert db.dropped == 0
        assert db.integrity_check() == "ok"

        summary = rt.aggregates.live_summary()
        assert summary["run_id"] == "e2e"
        assert len(summary["lanes"]) == 2
        assert summary["txns_today"] > 0
    finally:
        src.close()
        rt.close()
        db.close()


def test_close_drains_everything_still_queued(tmp_path):
    bus = Bus(maxsize=4096)
    db = DB(tmp_path / "close.db", batch_interval_s=5.0)   # slow flush on purpose
    rt = BackendRuntime(CFG, bus, db=db, run_id="c", started=0.0)
    rt.start(started=0.0)
    for i in range(50):
        bus.publish(ev(float(i), "occupancy", {"count": i}, "overhead"))
    bus.drain()
    rt.close()
    try:
        assert db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"] == 50
    finally:
        db.close()


def test_runtime_owning_its_db_closes_it(tmp_path):
    bus = Bus(maxsize=256)
    rt = BackendRuntime(CFG, bus, db_path=str(tmp_path / "owned.db"), run_id="o")
    rt.start(started=0.0)
    rt.close()
    assert rt.db._thread is None or not rt.db._thread.is_alive()
