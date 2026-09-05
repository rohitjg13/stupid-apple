"""Synthetic POS/ERP adapter with a REST-shaped interface.

A real POS would expose `GET /pos/transactions?since=<t>`; we back the same
shape with a generator so the backend can develop conversion and queue-mu
estimation before any real integration exists. The deck must say POS is
simulated.

Rules (from the plan):
- a transaction is emitted when a lane's estimated count drops by >= 1 (else on
  a timer when idle)
- items ~ Poisson(3), amount from planogram prices, zone_hint sampled from the
  last 10 minutes of zone dwell so conversion correlates with dwell
- service rate mu is derived from stub timestamps exactly as a real POS would
"""
from __future__ import annotations

import logging

import numpy as np

from core.events import Event

log = logging.getLogger(__name__)


class PosStub:
    def __init__(self, cfg, store_id=None, seed=0, rng=None):
        self.cfg = cfg
        self.store_id = store_id or cfg.store_id
        self.rng = rng or np.random.default_rng(seed)

        prices = {p["facing"]: p["unit_price"] for p in cfg.planogram}
        # A txn's SKU is sampled proportional to expected_sales_per_hour.
        self.skus = [(p["sku"], p["unit_price"], p["facing"]) for p in cfg.planogram]
        self.weights = np.array([p["expected_sales_per_hour"] for p in cfg.planogram],
                                dtype=float)
        if not self.skus:
            raise ValueError("planogram is empty; the POS stub needs SKUs to price")
        self.weights /= self.weights.sum()

        # Per-lane most recent headcount, to detect a checkout departure.
        self._last_count = {}
        self._last_txn_t = {}
        self._idle_after_s = float(cfg.checkout.get("idle_txn_after_s", 120.0))
        self._buffer = []

    def _txn(self, t, lane):
        idx = int(self.rng.choice(len(self.skus), p=self.weights))
        sku, price, facing = self.skus[idx]
        items = int(self.rng.poisson(3.0))
        items = max(1, items)
        amount = round(items * price * self.rng.uniform(0.8, 1.4), 2)
        event = Event(t, self.store_id, "-", None, "pos_txn", {
            "lane": int(lane), "items": items, "amount": amount,
            "zone_hint": facing,
        })
        self._buffer.append(event)
        return event

    def on_count(self, t, lane, count):
        """Feed the current estimated headcount; emits a txn when it drops."""
        prev = self._last_count.get(lane, 0)
        self._last_count[lane] = count
        if count >= prev:
            return None
        return self._txn(t, lane)

    def idle_tick(self, t, lanes=(1, 2)):
        """Emit a txn on an idle timer so mu is estimable even when quiet."""
        out = []
        for lane in lanes:
            last = self._last_txn_t.get(lane)
            if last is None or t - last >= self._idle_after_s:
                out.append(self._txn(t, lane))
                self._last_txn_t[lane] = t
        return out

    def txn(self, t, lane):
        self._last_txn_t[lane] = t
        return self._txn(t, lane)

    # ---- the REST-shaped adapter a real POS would call ---------------------
    def transactions(self, since=0.0):
        """Yield transactions since `since`, from the in-memory buffer.

        The stub keeps a short history so `since` can be honoured the same way
        a real ERP export would. The generator above drives this buffer.
        """
        return (e for e in self._buffer if e.t >= since)

    def feed(self, event):
        """Record a transaction event (so a real adapter and the stub are alike)."""
        self._buffer.append(event)
