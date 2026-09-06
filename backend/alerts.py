"""Alert rules -> `alert` events, with per-rule cooldowns.

The plan (05-backend-queue.md W2 §3) specifies these rules:

- stockout_start -> warning
- stock-out > 5 min -> critical
- queue predicted wait > target -> warning
- occupancy > capacity -> info
- clock_unsynced -> info

plus the W3 §3 staffing advice ("open counter 2"), which is the one alert that
has to fire *before* the thing it warns about.

Each rule has its own cooldown so a sustained condition does not flood the bus;
cooldowns are keyed per rule *and* per subject, so two lanes backing up at once
produce two alerts rather than one lane silencing the other.
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
    "staffing":        {"severity": "warning",  "cooldown_s": 60.0},
}

DEFAULT_CAPACITY = 50
STOCKOUT_CRITICAL_S = 300.0


class Alerts:
    def __init__(self, cfg, bus=None, store_id=None, rules=None, capacity=None):
        self.cfg = cfg
        self.bus = bus
        self.store_id = store_id or cfg.store_id
        self.rules = dict(rules or DEFAULT_RULES)
        # `capacity` is not part of the frozen store.yaml schema, so it is a
        # constructor argument; an optional checkout.capacity key overrides it
        # for a clipset that wants to tune the number without a code change.
        self.capacity = (capacity if capacity is not None
                         else cfg.checkout.get("capacity", DEFAULT_CAPACITY))
        self._last = {}                  # (rule, subject) -> last emit time
        self._open_stockouts = {}        # facing -> start time

    # ---- emission ---------------------------------------------------------
    def _publish(self, rule, message, t, severity=None, subject=None):
        spec = self.rules[rule]
        key = (rule, subject)
        if t - self._last.get(key, float("-inf")) < spec["cooldown_s"]:
            return None
        self._last[key] = t
        ev = Event(t, self.store_id, "-", None, "alert",
                   {"severity": severity or spec["severity"], "rule": rule,
                    "message": message})
        if self.bus is not None:
            self.bus.publish(ev)
        # Not `message`: logging reserves it on LogRecord and raises if an
        # `extra` tries to overwrite it.
        log.info("alert", extra={"rule": rule, "alert_message": message})
        return ev

    # ---- rules ------------------------------------------------------------
    def on_event(self, event):
        et = event.event_type
        if et == "stockout_start":
            self._open_stockouts[event.zone_id] = event.t
            return self._publish("stockout", f"{event.zone_id} empty", event.t,
                                 subject=event.zone_id)
        if et == "stockout_end":
            started = self._open_stockouts.pop(event.zone_id, None)
            if started is not None and event.t - started > STOCKOUT_CRITICAL_S:
                return self._publish(
                    "stockout_long",
                    f"{event.zone_id} empty for {int(event.t - started)}s",
                    event.t, subject=event.zone_id)
            return None
        if et == "queue_estimate":
            target = self.cfg.checkout.get("target_wait_s", 180.0)
            if event.payload["pred_wait_s"] > target:
                lane = event.payload["lane"]
                return self._publish(
                    "wait_over",
                    f"lane {lane} predicted wait "
                    f"{event.payload['pred_wait_s']:.0f}s > {target}s",
                    event.t, subject=lane)
            return None
        if et == "occupancy":
            if event.payload["count"] > self.capacity:
                return self._publish(
                    "occupancy_over",
                    f"occupancy {event.payload['count']} > {self.capacity}",
                    event.t, subject=None)
            return None
        return None

    def check_stockout_duration(self, t):
        """Escalate a stock-out that is still open past 5 minutes.

        Waiting for `stockout_end` would mean the critical alert only fires
        once the problem is already fixed, which is the wrong way round.
        """
        out = []
        for facing, started in self._open_stockouts.items():
            if t - started > STOCKOUT_CRITICAL_S:
                ev = self._publish("stockout_long",
                                   f"{facing} empty for {int(t - started)}s",
                                   t, subject=facing)
                if ev is not None:
                    out.append(ev)
        return out

    def staffing(self, t, advice):
        """"open counter 2: lambda=0.9/min, mu=0.7/min, projected wait 6.5 min"."""
        if not advice:
            return None
        return self._publish(
            "staffing",
            f"open counter {advice['open']}: "
            f"lambda={advice['lambda_per_min']}/min, mu={advice['mu_per_min']}/min, "
            f"projected wait {advice['projected_wait_s'] / 60.0:.1f} min",
            t, subject=advice["open"])

    def clock_unsynced(self, t):
        return self._publish("clock_unsynced", "system clock is unsynced", t)
