"""backend/alerts.py — rules -> alert events, with cooldowns."""
import pytest

from backend.alerts import Alerts
from core.config import load_clipset
from core.events import Event

CFG = load_clipset("config/sim")


def ev(t, et, payload, stream="overhead", zone_id=None):
    return Event(t, "demo-01", stream, zone_id, et, payload)


def test_stockout_start_warns_once_then_cooldown():
    a = Alerts(CFG)
    e1 = a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    assert e1 is not None and e1.payload["rule"] == "stockout"
    # within cooldown -> suppressed
    assert a.on_event(ev(1, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1")) is None


def test_stockout_long_after_five_minutes():
    a = Alerts(CFG)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    e = a.on_event(ev(301, "stockout_end", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    assert e is not None and e.payload["rule"] == "stockout_long"


def test_stockout_short_is_not_critical():
    a = Alerts(CFG)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    assert a.on_event(ev(10, "stockout_end", {"sku": "X", "facing": "A1"}, "shelf", "A1")) is None


def test_wait_over_target_warns():
    a = Alerts(CFG)
    e = a.on_event(ev(0, "queue_estimate", {"lane": 1, "count": 10, "pred_wait_s": 300}, "overhead"))
    assert e is not None and e.payload["rule"] == "wait_over"
    assert a.on_event(ev(0, "queue_estimate", {"lane": 1, "count": 1, "pred_wait_s": 30}, "overhead")) is None


def test_occupancy_over_capacity_informs():
    a = Alerts(CFG)
    e = a.on_event(ev(0, "occupancy", {"count": 999}, "overhead"))
    assert e is not None and e.payload["rule"] == "occupancy_over"


def test_clock_unsynced_info():
    a = Alerts(CFG)
    e = a.clock_unsynced(0)
    assert e is not None and e.payload["severity"] == "info"


def test_alerts_publish_to_bus():
    from core.bus import Bus
    bus, seen = Bus(), []
    bus.subscribe("alert", seen.append)
    a = Alerts(CFG, bus=bus)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    bus.drain()
    assert len(seen) == 1
