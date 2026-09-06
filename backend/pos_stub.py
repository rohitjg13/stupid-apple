"""Synthetic POS/ERP adapter with a REST-shaped interface.

A real POS would expose `GET /pos/transactions?since=<t>`; we back the same
shape with a generator so the backend can develop conversion and queue-mu
estimation before any real integration exists. **The deck must say POS is
simulated.**

Rules (from the plan):
- a transaction is emitted when a lane's estimated count drops by >= 1 (else on
  a timer when idle)
- items ~ Poisson(3), amount from planogram prices, zone_hint sampled from the
  last 10 minutes of zone dwell so conversion correlates with dwell
- service rate mu is derived from stub timestamps exactly as a real POS would

`zone_hint` is a **zone id**, not a shelf facing: on a real POS the line items
give SKUs, which map through planogram.json to the zone the SKU sits in, and
that zone is what the conversion join needs. The stub short-circuits the SKU
step but emits the same field a real adapter would.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

from core.events import Event

log = logging.getLogger(__name__)

DWELL_WINDOW_S = 600.0          # "the last 10 min zone-dwell distribution"
IDLE_TXN_AFTER_S = 120.0


class PosStub:
    def __init__(self, cfg, store_id=None, seed=0, rng=None):
        self.cfg = cfg
        self.store_id = store_id or cfg.store_id
        self.rng = rng or np.random.default_rng(seed)

        self.skus = [(p["sku"], p["unit_price"], p["facing"]) for p in cfg.planogram]
        if not self.skus:
            raise ValueError("planogram is empty; the POS stub needs SKUs to price")
        weights = np.array([max(0.0, float(p["expected_sales_per_hour"]))
                            for p in cfg.planogram])
        if weights.sum() <= 0:
            weights = np.ones(len(self.skus))       # no sales rates: sample evenly
        self.weights = weights / weights.sum()

        self.zones = [z["id"] for z in cfg.zones] or ["-"]
        self.lanes = sorted({int(l["lane"]) for l in cfg.lanes}) or [1]

        self._last_count = {}       # lane -> most recent estimated headcount
        self._last_txn_t = {}       # lane -> time of that lane's last txn
        self._dwell = deque()       # (t, zone, dwell_s) for the last 10 minutes
        self._idle_after_s = float(cfg.checkout.get("idle_txn_after_s",
                                                    IDLE_TXN_AFTER_S))
        self._buffer = []

    # ---- zone_hint sampling ------------------------------------------------
    def observe_visit(self, t, zone, dwell_s):
        """Feed a zone visit so zone_hint tracks where people actually spend time."""
        self._dwell.append((float(t), zone, max(0.0, float(dwell_s))))
        self._expire(t)

    def _expire(self, t):
        while self._dwell and self._dwell[0][0] < t - DWELL_WINDOW_S:
            self._dwell.popleft()

    def _zone_weights(self, t):
        self._expire(t)
        totals = {}
        for _, zone, dwell in self._dwell:
            if zone in self.zones:
                totals[zone] = totals.get(zone, 0.0) + dwell
        w = np.array([totals.get(z, 0.0) for z in self.zones])
        if w.sum() <= 0:
            # Nothing observed yet: uniform, so a cold start is not a silent
            # bias towards whichever zone happens to be first in zones.json.
            w = np.ones(len(self.zones))
        return w / w.sum()

    # ---- generation --------------------------------------------------------
    def _txn(self, t, lane):
        idx = int(self.rng.choice(len(self.skus), p=self.weights))
        sku, price, _facing = self.skus[idx]
        items = max(1, int(self.rng.poisson(3.0)))
        amount = round(items * price * self.rng.uniform(0.8, 1.4), 2)
        zone = str(self.rng.choice(self.zones, p=self._zone_weights(t)))
        event = Event(t, self.store_id, "-", None, "pos_txn", {
            "lane": int(lane), "items": items, "amount": amount,
            "zone_hint": zone,
        })
        self._buffer.append(event)
        self._last_txn_t[int(lane)] = float(t)
        return event

    def on_count(self, t, lane, count):
        """Feed the current estimated headcount; emits a txn when it drops.

        The plan says "drops by >= 1", and it means it: the headcount is a
        smoothed regression output that jitters by tenths of a person every
        sample. Comparing raw floats booked a transaction on every downward
        wobble — 59 of them in a 45-second clip — which then fed a service rate
        of nearly a customer a second into the wait model.

        A drop of two or more in one sample is noise, not two people leaving at
        once, so it still books a single transaction.
        """
        lane = int(lane)
        prev = self._last_count.get(lane)
        level = int(round(float(count)))
        self._last_count[lane] = level
        if prev is None or level >= prev:
            return None
        return self._txn(t, lane)

    def idle_tick(self, t, lanes=None):
        """Emit a txn on an idle timer so mu is estimable even when quiet."""
        out = []
        for lane in (self.lanes if lanes is None else lanes):
            last = self._last_txn_t.get(int(lane))
            if last is None or t - last >= self._idle_after_s:
                out.append(self._txn(t, lane))
        return out

    def txn(self, t, lane):
        return self._txn(t, lane)

    # ---- the REST-shaped adapter a real POS would expose --------------------
    def transactions(self, since=0.0):
        """Yield transactions since `since`, from the in-memory buffer.

        The stub keeps a short history so `since` can be honoured the same way
        a real ERP export would. The generator above drives this buffer.
        """
        return (e for e in self._buffer if e.t >= since)

    def feed(self, event):
        """Record a transaction event (so a real adapter and the stub are alike)."""
        self._buffer.append(event)
        self._last_txn_t[int(event.payload["lane"])] = float(event.t)


def service_rate(txns, lane=None):
    """mu: per-server service rate from POS timestamps, exactly as a real POS.

    Mean inter-departure time per lane, inverted. Returns None when there are
    fewer than two transactions to difference — an honest "not enough data yet"
    beats a fabricated mu that the wait prediction would inherit.
    """
    ts = sorted(e.t for e in txns
                if lane is None or int(e.payload["lane"]) == int(lane))
    if len(ts) < 2:
        return None
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    if not gaps:
        return None
    mean_gap = sum(gaps) / len(gaps)
    return 1.0 / mean_gap if mean_gap > 0 else None
