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


# ---- cooldowns are per subject, not per rule -------------------------------

def test_two_lanes_backing_up_both_alert():
    """One lane must not silence the other: the cooldown is keyed per subject."""
    a = Alerts(CFG)
    e1 = a.on_event(ev(0, "queue_estimate", {"lane": 1, "count": 9, "pred_wait_s": 300}))
    e2 = a.on_event(ev(0, "queue_estimate", {"lane": 2, "count": 9, "pred_wait_s": 300}))
    assert e1 is not None and e2 is not None
    # ...but the same lane twice inside the cooldown is suppressed.
    assert a.on_event(ev(1, "queue_estimate", {"lane": 1, "count": 9,
                                               "pred_wait_s": 300})) is None


def test_two_facings_empty_both_alert():
    a = Alerts(CFG)
    assert a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"},
                         "shelf", "A1")) is not None
    assert a.on_event(ev(0, "stockout_start", {"sku": "Y", "facing": "B1"},
                         "shelf", "B1")) is not None


def test_cooldown_expires():
    a = Alerts(CFG)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    a.on_event(ev(0, "stockout_end", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    e = a.on_event(ev(61, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    assert e is not None


# ---- critical stock-out fires while it is still open -----------------------

def test_open_stockout_escalates_without_waiting_for_the_end():
    """Waiting for stockout_end would only warn once the shelf is refilled."""
    a = Alerts(CFG)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    assert a.check_stockout_duration(100) == []
    out = a.check_stockout_duration(301)
    assert len(out) == 1
    assert out[0].payload["rule"] == "stockout_long"
    assert out[0].payload["severity"] == "critical"


def test_open_stockout_escalation_respects_its_cooldown():
    a = Alerts(CFG)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    assert len(a.check_stockout_duration(301)) == 1
    assert a.check_stockout_duration(302) == []


def test_stockout_end_after_escalation_does_not_double_fire():
    a = Alerts(CFG)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    a.check_stockout_duration(301)
    assert a.on_event(ev(302, "stockout_end", {"sku": "X", "facing": "A1"},
                         "shelf", "A1")) is None


# ---- staffing (the alert that has to fire early) ---------------------------

def test_staffing_alert_message_carries_the_numbers():
    a = Alerts(CFG)
    e = a.staffing(0, {"open": 2, "lambda_per_min": 0.9, "mu_per_min": 0.7,
                       "projected_wait_s": 390.0})
    assert e is not None and e.payload["rule"] == "staffing"
    assert e.payload["severity"] == "warning"
    assert "open counter 2" in e.payload["message"]
    assert "lambda=0.9/min" in e.payload["message"]
    assert "6.5 min" in e.payload["message"]


def test_staffing_advice_of_none_is_not_an_alert():
    assert Alerts(CFG).staffing(0, None) is None


def test_staffing_has_a_sixty_second_cooldown():
    from backend.alerts import DEFAULT_RULES
    assert DEFAULT_RULES["staffing"]["cooldown_s"] == 60.0
    a = Alerts(CFG)
    advice = {"open": 2, "lambda_per_min": 1.0, "mu_per_min": 1.0,
              "projected_wait_s": 60.0}
    assert a.staffing(0, advice) is not None
    assert a.staffing(30, advice) is None
    assert a.staffing(61, advice) is not None


# ---- capacity --------------------------------------------------------------

def test_capacity_is_configurable_without_touching_store_yaml():
    a = Alerts(CFG, capacity=5)
    assert a.on_event(ev(0, "occupancy", {"count": 6})) is not None
    assert Alerts(CFG, capacity=5).on_event(ev(0, "occupancy", {"count": 5})) is None


def test_default_capacity_when_the_clipset_does_not_say():
    from backend.alerts import DEFAULT_CAPACITY
    assert Alerts(CFG).capacity == DEFAULT_CAPACITY


# ---- every alert is a valid Event on the frozen contract -------------------

def test_every_rule_produces_a_valid_alert_event():
    from core.events import EVENT_TYPES
    a = Alerts(CFG, capacity=1)
    events = [
        a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1")),
        a.on_event(ev(0, "queue_estimate", {"lane": 1, "count": 9, "pred_wait_s": 999})),
        a.on_event(ev(0, "occupancy", {"count": 99})),
        a.clock_unsynced(0),
        a.staffing(0, {"open": 3, "lambda_per_min": 1.0, "mu_per_min": 1.0,
                       "projected_wait_s": 10.0}),
    ]
    assert all(e is not None for e in events)
    rules = {e.payload["rule"] for e in events}
    assert rules == {"stockout", "wait_over", "occupancy_over", "clock_unsynced",
                     "staffing"}
    for e in events:
        assert e.event_type == "alert"
        assert set(e.payload) == EVENT_TYPES["alert"]
        assert e.payload["severity"] in {"info", "warning", "critical"}


def test_every_rule_logs_cleanly_at_info(caplog):
    """`logging` reserves names like `message` on LogRecord and raises when an
    `extra` collides with one. pytest defaults to WARNING and a 400-frame sim
    run fires no alerts at all, so without this the alert path is never logged
    under test and the crash only shows up on stage.
    """
    import logging
    caplog.set_level(logging.INFO, logger="backend.alerts")
    a = Alerts(CFG, capacity=1)
    a.on_event(ev(0, "stockout_start", {"sku": "X", "facing": "A1"}, "shelf", "A1"))
    a.on_event(ev(0, "queue_estimate", {"lane": 1, "count": 9, "pred_wait_s": 999}))
    a.on_event(ev(0, "occupancy", {"count": 99}))
    a.clock_unsynced(0)
    a.staffing(0, {"open": 3, "lambda_per_min": 1.0, "mu_per_min": 1.0,
                   "projected_wait_s": 10.0})
    a.check_stockout_duration(400)

    logged = [r for r in caplog.records if r.name == "backend.alerts"]
    assert len(logged) == 6
    assert {r.rule for r in logged} == {
        "stockout", "wait_over", "occupancy_over", "clock_unsynced", "staffing",
        "stockout_long"}


def test_alert_records_survive_the_json_formatter(caplog):
    """main.py renders every record as a JSON line; an alert must not break it."""
    import json
    import logging

    from main import JsonLines

    caplog.set_level(logging.INFO, logger="backend.alerts")
    Alerts(CFG).clock_unsynced(0)
    record = [r for r in caplog.records if r.name == "backend.alerts"][0]
    line = json.loads(JsonLines().format(record))
    assert line["msg"] == "alert"
    assert line["rule"] == "clock_unsynced"
    assert line["alert_message"] == "system clock is unsynced"
