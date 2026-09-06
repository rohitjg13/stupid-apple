"""main.py wiring: capture -> analytics -> bus, and a clean exit."""
import subprocess
import sys

import pytest

import numpy as np

from core.bus import Bus
from core.config import load_clipset
from main import build_backend, build_source, run


def test_frame_budget_is_respected():
    bus = Bus()
    seen = []
    bus.subscribe("*", seen.append)
    n = run(config="config/sim", source="sim", backend="sim", frames=60, realtime=False, bus=bus)
    bus.drain()
    assert n == 60
    assert seen, "no events reached the bus"


def test_occupancy_is_published_about_once_a_second():
    bus = Bus()
    occ = []
    bus.subscribe("occupancy", occ.append)
    # overhead alone: 150 frames at 15 fps == 10 s of stream time
    run(config="config/sim", source="sim", backend="sim", frames=150, realtime=False,
        bus=bus, streams=("overhead",))
    bus.drain()
    assert 8 <= len(occ) <= 12, f"expected ~10 occupancy events, got {len(occ)}"
    assert all(e.payload["count"] >= 0 for e in occ)


def test_shelf_fill_published_per_roi():
    bus = Bus()
    fills = []
    bus.subscribe("shelf_fill", fills.append)
    run(config="config/sim", source="sim", backend="sim", frames=50, realtime=False,
        bus=bus, streams=("shelf",))
    bus.drain()
    assert {e.zone_id for e in fills} == {"A1", "A2", "A3", "B1", "B2", "B3"}


def test_events_carry_the_store_id_from_store_yaml():
    bus = Bus()
    seen = []
    bus.subscribe("*", seen.append)
    run(config="config/sim", source="sim", backend="sim", frames=30, realtime=False, bus=bus)
    bus.drain()
    assert {e.store_id for e in seen} == {"demo-01"}


def test_frames_budget_counts_both_streams():
    """--frames N is N frames total, not N per stream. CI depends on it terminating."""
    bus = Bus()
    n = run(config="config/sim", source="sim", backend="sim", frames=100, realtime=False, bus=bus)
    assert n == 100


def test_event_time_follows_stream_time_not_the_wall_clock():
    """--fast must emit the same events as --realtime, just sooner."""
    bus = Bus()
    occ = []
    bus.subscribe("occupancy", occ.append)
    run(config="config/sim", source="sim", backend="sim", frames=150, realtime=False,
        bus=bus, streams=("overhead",))
    bus.drain()
    gaps = [b.t - a.t for a, b in zip(occ, occ[1:])]
    assert all(0.9 <= g <= 1.15 for g in gaps), gaps


def test_health_file_is_written(tmp_path):
    import json
    hp = tmp_path / "health.json"
    run(config="config/sim", source="sim", backend="sim", frames=40, realtime=False,
        health_file=hp, clock_state=tmp_path / "clock.json")
    h = json.loads(hp.read_text())
    assert h["store_id"] == "demo-01" and h["frames"] == 40
    assert h["clock_unsynced"] is False and h["running"] is False


def test_run_never_writes_outside_the_given_state_path(tmp_path):
    """Offline-first: no writes to /var on a laptop, and no crash either."""
    run(config="config/sim", source="sim", backend="sim", frames=20, realtime=False,
        clock_state=tmp_path / "sub" / "clock.json")
    assert (tmp_path / "sub" / "clock.json").exists()


def test_pl_backend_refuses_to_start_without_a_board():
    """Never silently run reference while the judges are told it is the FPGA."""
    with pytest.raises(SystemExit):
        run(config="config/sim", source="sim", backend="pl", frames=1, realtime=False)


def test_file_source_through_the_reference_backend(tmp_path):
    """The W2 checkpoint: real footage, no board, blobs from pl/reference.py."""
    import cv2
    from tests.test_sources import make_clip

    clip = make_clip(tmp_path / "c.avi", n=40)
    cfg_dir = tmp_path / "cs"
    import shutil
    shutil.copytree("config/sim", cfg_dir)
    store = (cfg_dir / "store.yaml").read_text().replace(
        "overhead: {source: sim, fps: 15, loop: true}",
        f"overhead: {{source: file, path: {clip}, fps: 15, loop: true}}")
    (cfg_dir / "store.yaml").write_text(store)

    bus = Bus()
    occ = []
    bus.subscribe("occupancy", occ.append)
    n = run(config=cfg_dir, source="file", backend="reference", frames=200,
            realtime=False, bus=bus, streams=("overhead",))
    bus.drain()
    assert n == 200
    assert occ, "no occupancy events from footage"
    assert max(e.payload["count"] for e in occ) >= 1, "the moving box was never detected"


def test_backend_fills_in_the_frame_result():
    cfg = load_clipset("config/sim")
    be = build_backend("reference", cfg)
    img = np.full((480, 640, 3), 30, np.uint8)
    r = be.process(img, 0, 5)
    assert int(r["frame_id"]) == 5


def test_sim_backend_is_a_passthrough():
    assert build_backend("sim", load_clipset("config/sim")) is None


def test_unknown_source_fails_loudly():
    with pytest.raises(SystemExit):
        build_source("teleport", None, "overhead")


def test_ci_smoke_command_exits_zero():
    """The exact command CI runs; if this breaks, main is broken."""
    r = subprocess.run([sys.executable, "main.py", "--source", "sim", "--frames", "300",
                        "--headless", "--config", "config/sim"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "occupancy" in r.stderr + r.stdout


# ---- the backend, wired in (opt-in, so the CI command above is unchanged) ---

def test_backend_is_off_unless_asked_for(tmp_path):
    """`python main.py --source sim` must not need a writable database."""
    run(config="config/sim", source="sim", backend="sim", frames=20, realtime=False,
        clock_state=tmp_path / "clock.json")
    assert not (tmp_path / "retail.db").exists()
    assert not (tmp_path.parent / "retail.db").exists()


def test_db_flag_persists_every_table_the_backend_owns(tmp_path):
    from backend.aggregates import Aggregates
    from backend.db import DB

    db_path = tmp_path / "retail.db"
    n = run(config="config/sim", source="sim", backend="sim", frames=600,
            realtime=False, db=str(db_path), run_id="seg-1",
            clock_state=tmp_path / "clock.json")
    assert n == 600

    db = DB(db_path)
    try:
        for table in ("occupancy", "shelf_fill", "lane_occ", "queue_estimate",
                      "pos_txn"):
            count = db.query_one(
                f"SELECT COUNT(*) AS n FROM {table} WHERE run_id='seg-1'")["n"]
            assert count > 0, f"{table} is empty after a 600-frame run"
        assert db.query_one("SELECT store_id FROM run WHERE run_id='seg-1'")[
            "store_id"] == "demo-01"
        assert db.integrity_check() == "ok"
        assert db.dropped == 0

        summary = Aggregates(db, run_id="seg-1").live_summary()
        assert summary["run_id"] == "seg-1"
        assert len(summary["lanes"]) == 2
    finally:
        db.close()


def test_run_id_defaults_to_the_start_time(tmp_path):
    from backend.db import DB
    db_path = tmp_path / "r.db"
    run(config="config/sim", source="sim", backend="sim", frames=20, realtime=False,
        db=str(db_path), clock_state=tmp_path / "clock.json")
    db = DB(db_path)
    try:
        rows = db.query("SELECT run_id FROM run")
        assert len(rows) == 1 and rows[0]["run_id"].startswith("run-")
    finally:
        db.close()


def test_health_file_names_the_run(tmp_path):
    import json
    hp = tmp_path / "health.json"
    run(config="config/sim", source="sim", backend="sim", frames=30, realtime=False,
        db=str(tmp_path / "r.db"), run_id="seg-9", health_file=hp,
        clock_state=tmp_path / "clock.json")
    h = json.loads(hp.read_text())
    assert h["run_id"] == "seg-9"
    assert h["bus_dropped"] == 0
    assert h["running"] is False


def test_two_segments_keep_their_numbers_apart(tmp_path):
    """tools/reset_run.py's whole purpose: each demo segment reads clean."""
    from backend.aggregates import Aggregates
    from backend.db import DB

    db_path = tmp_path / "r.db"
    for run_id in ("seg-1", "seg-2"):
        run(config="config/sim", source="sim", backend="sim", frames=200,
            realtime=False, db=str(db_path), run_id=run_id,
            clock_state=tmp_path / "clock.json")

    db = DB(db_path)
    try:
        assert {r["run_id"] for r in db.query("SELECT run_id FROM run")} == \
            {"seg-1", "seg-2"}
        one = db.query_one("SELECT COUNT(*) AS n FROM occupancy WHERE run_id='seg-1'")["n"]
        both = db.query_one("SELECT COUNT(*) AS n FROM occupancy")["n"]
        assert 0 < one < both
        assert Aggregates(db, run_id="seg-1").live_summary()["run_id"] == "seg-1"
    finally:
        db.close()


def test_the_whole_pipeline_logs_cleanly_at_info(tmp_path, caplog):
    """logging reserves names like `message` on LogRecord and raises if an
    `extra` collides. Tests default to WARNING, so an INFO-only crash walks
    straight past them and onto the stage."""
    import logging
    caplog.set_level(logging.INFO)
    run(config="config/sim", source="sim", backend="sim", frames=400, realtime=False,
        db=str(tmp_path / "r.db"), run_id="loud", clock_state=tmp_path / "clock.json")
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_serve_flag_smoke(tmp_path):
    """--serve must bring up a real socket the frontend can reach."""
    import json
    import threading
    import urllib.request

    from backend.aggregates import Aggregates
    from backend.db import DB
    from backend.server import create_app, serve_in_thread

    db = DB(tmp_path / "s.db", batch_interval_s=0.01)
    db.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0,
                      "clipset": "sim"})
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 4})
    db.drain()
    app = create_app(Aggregates(db, run_id="r1"), health=lambda: {"ok": True})
    thread, server = serve_in_thread(app, host="127.0.0.1", port=8129)
    try:
        deadline = threading.Event()
        body = None
        for _ in range(100):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8129/api/live",
                                            timeout=1.0) as r:
                    body = json.loads(r.read())
                break
            except Exception:
                deadline.wait(0.1)
        assert body is not None, "the API never came up"
        assert body["occupancy"] == 4
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        db.close()


def test_streams_flag_runs_one_stream_only(tmp_path):
    """A queue clip has no shelf. One ReferenceBackend is shared by both
    streams and re-warms its background model on every switch, so running both
    against a single-stream clip destroys the model rather than just wasting
    CPU."""
    bus = Bus()
    seen = []
    bus.subscribe("*", seen.append)
    run(config="config/sim", source="sim", backend="sim", frames=60, realtime=False,
        bus=bus, streams=("overhead",))
    bus.drain()
    assert {e.stream for e in seen} == {"overhead"}


def test_unknown_stream_name_fails_loudly():
    from main import main as cli
    with pytest.raises(SystemExit):
        cli(["--source", "sim", "--frames", "1", "--headless", "--streams", "roof"])
