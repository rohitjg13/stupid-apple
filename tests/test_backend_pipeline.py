"""backend/pipeline.py: raw lane cells -> the four backend event types."""
from dataclasses import replace

import numpy as np
import pytest

from backend.pipeline import BackendPipeline
from core.bus import Bus
from core.config import load_clipset
from core.events import Event
from pl.contract import FRAME_RESULT_DT

CFG = load_clipset("config/sim")


class Collector:
    def __init__(self, bus):
        self.events = []
        bus.subscribe("*", self.events.append)

    def of(self, event_type):
        return [e for e in self.events if e.event_type == event_type]


def result(lane_occupancy=()):
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    for i, v in enumerate(lane_occupancy):
        r["lane_occupancy"][i] = v
    return r


def drive(bus, pipe, cells, t0=1000.0, n=10, dt=1.0):
    for i in range(n):
        pipe.on_frame(t0 + i * dt, result(cells))
    bus.drain()


def test_publishes_lane_occ_per_lane_with_sixteen_cells():
    bus = Bus()
    got = Collector(bus)
    drive(bus, BackendPipeline(CFG, bus), [90] * 8, n=1)

    occ = got.of("lane_occ")
    assert {e.payload["lane"] for e in occ} == {1, 2}
    for e in occ:
        assert len(e.payload["cells"]) == 16          # sink.pack_cells demands 16


def test_queue_estimate_follows_the_lane_filling_up():
    bus = Bus()
    got = Collector(bus)
    pipe = BackendPipeline(CFG, bus)
    drive(bus, pipe, [0] * 8, n=2)
    drive(bus, pipe, [95] * 8, t0=1010.0, n=20)

    lane1 = [e.payload for e in got.of("queue_estimate") if e.payload["lane"] == 1]
    assert lane1[0]["count"] == 0.0
    assert lane1[-1]["count"] > 2.0                   # the EMA has caught up
    assert lane1[-1]["pred_wait_s"] == pytest.approx(lane1[-1]["count"] * 45.0, abs=0.1)


def test_empty_lanes_predict_no_wait():
    bus = Bus()
    got = Collector(bus)
    drive(bus, BackendPipeline(CFG, bus), [0] * 8, n=5)

    assert all(e.payload["pred_wait_s"] == 0.0 for e in got.of("queue_estimate"))


def test_rate_limited_to_the_period():
    bus = Bus()
    got = Collector(bus)
    pipe = BackendPipeline(CFG, bus, period_s=0.5)
    for i in range(10):
        pipe.on_frame(1000.0 + i * 0.1, result([50] * 8))     # 1 s of 10 fps
    bus.drain()

    assert len(got.of("lane_occ")) == 2 * 2           # two lanes, twice, not ten


def test_a_long_queue_raises_the_wait_alert():
    # Four cells x 0.8 caps the estimate at 3.2 people, so config/sim's 180 s
    # target is only reachable with a slower till. A real clipset gets more
    # cells (tools/autoconfig.py lays out eight) and does not need this.
    cfg = replace(CFG, checkout=dict(CFG.checkout, service_s=90.0))
    bus = Bus()
    got = Collector(bus)
    drive(bus, BackendPipeline(cfg, bus), [100] * 8, n=60)

    rules = {e.payload["rule"] for e in got.of("alert")}
    assert "wait_over" in rules                        # target_wait_s is 180 in config/sim


def test_busy_door_recommends_opening_a_counter():
    bus = Bus()
    got = Collector(bus)
    pipe = BackendPipeline(CFG, bus, advice_period_s=1.0)
    for i in range(120):                               # 2 entries a second, sustained
        bus.publish(Event(1000.0 + i * 0.5, CFG.store_id, "overhead", "door",
                          "tripwire", {"dir": "in", "tripwire_id": "door"}))
    bus.drain()
    drive(bus, pipe, [0] * 8, t0=1060.0, n=4)

    assert any(e.payload["rule"] == "open_counter" for e in got.of("alert"))


def test_an_empty_store_sells_nothing():
    """Phantom sales put Purchased above Footfall and conversion over 100%."""
    bus = Bus()
    got = Collector(bus)
    drive(bus, BackendPipeline(CFG, bus), [0] * 8, n=5)

    assert not got.of("pos_txn")


def test_a_quiet_lane_still_sells_once_somebody_is_in_the_store():
    bus = Bus()
    got = Collector(bus)
    pipe = BackendPipeline(CFG, bus)
    bus.publish(Event(1000.0, CFG.store_id, "overhead", None, "occupancy", {"count": 3}))
    bus.drain()
    drive(bus, pipe, [0] * 8, n=5)

    assert got.of("pos_txn"), "conversion needs transactions from a quiet store too"
