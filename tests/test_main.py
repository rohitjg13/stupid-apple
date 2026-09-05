"""main.py wiring: capture -> analytics -> bus, and a clean exit."""
import subprocess
import sys

import pytest

from core.bus import Bus
from main import build_source, run


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


def test_pl_backend_refuses_to_start_without_a_board():
    """Never silently run reference while the judges are told it is the FPGA."""
    with pytest.raises(SystemExit):
        run(config="config/sim", source="sim", backend="pl", frames=1, realtime=False)


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
