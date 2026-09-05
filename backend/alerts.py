"""Alert rules -> `alert` events, with per-rule cooldowns.

The plan (05-backend-queue.md W2 §3) specifies these rules:

- stockout_start -> warning
- stock-out > 5 min -> critical
- queue predicted wait > target -> warning
- occupancy > capacity -> info
- clock_unsynced -> info

Each rule has its own cooldown so a sustained condition does not flood the bus.
The rules engine consumes events and publishes `alert` events back onto the bus.
"""
from __future__ import annotations

import logging

from core.events import Event

log = logging.getLogger(__name__)

DEFAULT_RULES = {
    "stockout":        {"severity": "warning",  "cooldown_s": 60.0},
    "stockout_long":   {"severity": "critical", "cooldown_s": 300.0},
    "wait_over":       {"severity": "warning",  "cooldown_s": 60.0},
    "occupancy_over":  {"severity": "info",     "cooldown_s": 60.0},
    "clock_unsynced":  {"severity": "info",     "cooldown_s": 3600.0},
}


class Alerts:
    def __init__(self, cfg, bus=None, store_id=None, rules=None, _now=None):
        self.cfg = cfg
        self.bus = bus
        self.store_id = store_id or cfg.store_id
        self.rules = rules or DEFAULT_RULES
        self._now = _now
        self._last = {}                  # rule -> last emit time
        self._open_stockouts = {}        # facing -> start time
        self._open_alerts = {}           # alert id -> event (for ack)

    def _publish(self, rule, message, t, severity=None):
        spec = self.rules[rule]
        sev = severity or spec["severity"]
        if self._now is None and t - self._last.get(rule, -1e18) < spec["cooldown_s"]:
            return None
        self._last[rule] = t
        ev = Event(t, self.store_id, "-", None, "alert",
                   {"severity": sev, "rule": rule, "message": message})
        if self.bus is not None:
            self.bus.publish(ev)
        return ev

    def on_event(self, event):
        if event.event_type == "stockout_start":
            self._open_stockouts[event.zone_id] = event.t
            return self._publish("stockout", f"{event.zone_id} empty",
                                 event.t)
        if event.event_type == "stockout_end":
            started = self._open_stockouts.pop(event.zone_id, None)
            if started is not None and event.t - started > 300.0:
                return self._publish("stockout_long",
                                     f"{event.zone_id} empty for {int(event.t - started)}s",
                                     event.t, severity="critical")
        if event.event_type == "queue_estimate":
            target = self.cfg.checkout.get("target_wait_s", 180.0)
            if event.payload["pred_wait_s"] > target:
                return self._publish("wait_over",
                                     f"lane {event.payload['lane']} predicted wait "
                                     f"{event.payload['pred_wait_s']:.0f}s > {target}s",
                                     event.t)
        if event.event_type == "occupancy":
            capacity = self.cfg.checkout.get("capacity", 50)
            if event.payload["count"] > capacity:
                return self._publish("occupancy_over",
                                     f"occupancy {event.payload['count']} > {capacity}",
                                     event.t)
        return None

    def clock_unsynced(self, t):
        return self._publish("clock_unsynced", "system clock is unsynced", t)

    def ack(self, alert_id):
        return self._open_alerts.pop(alert_id, None)
