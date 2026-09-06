"""Raw `lane_occupancy` -> `lane_occ`, `queue_estimate`, `pos_txn`, `alert`.

docs/SHARED.md §5 assigns these four event types to the backend, "from raw",
and nothing else in the tree turns a FrameResult's 16 queue cells into events.
main.py calls `on_frame()` for every overhead frame; this is the only place
that reads `result["lane_occupancy"]`.

Wait is `count * service_s` -- one server per lane, Little's law -- not Erlang.
Erlang C answers a different question (how many counters should be open), and
that is what `staffing.staffing_advice` is used for below, once every
`advice_period_s`, off the arrival rate measured from door tripwires.

The POS is simulated and the deck has to say so. What it is *not* allowed to be
is untethered: a transaction is a shopper leaving the checkout with something,
so a simulated one follows an observed departure -- a lane's count dropping
where the lanes are calibrated, otherwise a shopper crossing the door outbound,
with probability `pos.conversion`. Sales on a timer sold to an empty shop and
printed conversion rates over 100%.
"""
from __future__ import annotations

import logging

from backend.alerts import Alerts
from backend.pos_stub import PosStub
from backend.queue_model import ema, naive_count
from backend.staffing import staffing_advice
from core.events import Event

log = logging.getLogger(__name__)

CELLS = 16                      # lane_occ payload width, frozen by the contract
LAMBDA_WINDOW_S = 300.0         # arrivals are averaged over the last 5 minutes


class BackendPipeline:
    def __init__(self, cfg, bus, period_s=0.5, advice_period_s=30.0, seed=0):
        self.cfg, self.bus = cfg, bus
        self.store_id = cfg.store_id
        self.period_s = float(period_s)
        self.advice_period_s = float(advice_period_s)

        self.lanes = {}                                 # lane number -> cell indices
        for i, lane in enumerate(cfg.lanes):
            self.lanes.setdefault(int(lane["lane"]), []).append(i)

        self.counters = int(cfg.checkout.get("counters", max(len(self.lanes), 1)))
        self.target_wait_s = float(cfg.checkout.get("target_wait_s", 180.0))
        # ponytail: a fixed mean service time until a real POS gives us one.
        # store.yaml can override it per clipset; the POS stub's timestamps are
        # the upgrade path.
        self.service_s = float(cfg.checkout.get("service_s", 45.0))

        self.pos = PosStub(cfg, seed=seed) if cfg.planogram else None
        self.buy_rate = float(cfg.pos.get("conversion", 0.35))
        self._lanes_live = False        # true once a lane cell ever reads busy
        self.alerts = Alerts(cfg, bus)
        bus.subscribe("*", self.alerts.on_event)
        bus.subscribe("tripwire", self._on_tripwire)

        self._count = {}                                # lane -> smoothed headcount
        self._arrivals = []                             # entry times, trimmed
        self._last = -1e9
        self._last_advice = -1e9

    # ---- inputs -----------------------------------------------------------
    def _on_tripwire(self, event):
        if event.payload["dir"] == "in":
            self._arrivals.append(event.t)
        elif (event.payload["dir"] == "out" and self.pos is not None
              and not self._lanes_live and self.pos.rng.random() < self.buy_rate):
            # Whoever just left either bought something or did not. With real
            # lane cells the drop in `on_count` is the better signal and this
            # stands down, so a departure is never counted twice.
            self.bus.publish(self.pos.txn(event.t, lane=1))

    def _lambda(self, t):
        """Entries per second over the last LAMBDA_WINDOW_S."""
        self._arrivals = [a for a in self._arrivals if a > t - LAMBDA_WINDOW_S]
        if not self._arrivals:
            return 0.0
        span = max(t - self._arrivals[0], 1.0)
        return len(self._arrivals) / span

    def _publish(self, t, event_type, payload, zone_id=None):
        self.bus.publish(Event(t, self.store_id, "-", zone_id, event_type, payload))

    # ---- per frame --------------------------------------------------------
    def on_frame(self, t, result):
        """Overhead frames only. Rate-limited to `period_s` (2 Hz per §5)."""
        if t - self._last < self.period_s:
            return
        self._last = t

        for lane, idxs in sorted(self.lanes.items()):
            cells = [int(result["lane_occupancy"][i]) for i in idxs][:CELLS]
            self._publish(t, "lane_occ", {"lane": lane, "cells": cells + [0] * (CELLS - len(cells))})

            count = ema(self._count.get(lane), naive_count(cells))
            self._count[lane] = count
            self._lanes_live = self._lanes_live or count > 0.5
            self._publish(t, "queue_estimate",
                          {"lane": lane, "count": round(count, 2),
                           "pred_wait_s": round(count * self.service_s, 1)})

            if self.pos is not None:
                txn = self.pos.on_count(t, lane, count)
                if txn is not None:
                    self.bus.publish(txn)

        self._advise(t)

    def _advise(self, t):
        if t - self._last_advice < self.advice_period_s:
            return
        self._last_advice = t
        advice = staffing_advice(self.counters, self._lambda(t),
                                 1.0 / self.service_s, self.target_wait_s)
        if advice is None:
            return
        self._publish(t, "alert",
                      {"severity": "warning", "rule": "open_counter",
                       "message": f"congestion predicted: open {advice['open']} counters "
                                  f"(now {self.counters}) — {advice['lambda_per_min']}/min "
                                  f"arriving, projected wait "
                                  f"{advice['projected_wait_s']:.0f}s"})
