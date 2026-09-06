"""FastAPI router over the aggregates; the dashboard server mounts it.

Signatures mirror the plan (05-backend-queue.md W1 §3): `/api/...` over the
`Aggregates` functions. No state lives here; the router is built with a bound
`Aggregates` instance, so tests can hit it with a temp DB and no network.

Every window parameter is a unix second, and every `run_id` defaults to the
run the router was built with, so the dashboard can call any endpoint with no
arguments and get the current segment.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.aggregates import Aggregates


def create_router(aggregates: Aggregates, health=None) -> APIRouter:
    router = APIRouter(prefix="/api")
    agg = aggregates

    @router.get("/health")
    def health_check():
        """Offline-first: served from a file main.py writes, never a network call."""
        return health() if callable(health) else (health or {"ok": True})

    @router.get("/live")
    def live_summary(run_id: Optional[str] = None):
        return agg.live_summary(run_id)

    @router.get("/footfall/hour")
    def footfall_by_hour(day: Optional[float] = None, run_id: Optional[str] = None):
        return agg.footfall_by_hour(day if day is not None else time.time(), run_id)

    @router.get("/footfall/day")
    def footfall_by_day(n: int = 7, run_id: Optional[str] = None):
        return agg.footfall_by_day(n, run_id)

    @router.get("/dwell")
    def dwell_by_zone(t0: float, t1: float, run_id: Optional[str] = None):
        return agg.dwell_by_zone(t0, t1, run_id)

    @router.get("/dwell/store")
    def dwell_littles_law(t0: float, t1: float, run_id: Optional[str] = None):
        """Store-wide mean dwell, W = L / lambda."""
        return agg.dwell_littles_law(t0, t1, run_id)

    @router.get("/heatmap")
    def heatmap(t0: float, t1: float, run_id: Optional[str] = None):
        return agg.heatmap(t0, t1, run_id)

    @router.get("/stockouts")
    def stockout_durations(t0: float, t1: float, run_id: Optional[str] = None):
        return agg.stockout_durations(t0, t1, run_id)

    @router.get("/stockouts/stats")
    def restock_stats(t0: float, t1: float, run_id: Optional[str] = None):
        return agg.restock_stats(t0, t1, run_id)

    @router.get("/replenishment")
    def replenishment_list(run_id: Optional[str] = None):
        return agg.replenishment_list(run_id)

    @router.get("/queue")
    def queue_status(run_id: Optional[str] = None):
        return agg.queue_status(run_id)

    @router.get("/queue/wait-pred-vs-actual")
    def wait_pred_vs_actual(t0: float, t1: float, run_id: Optional[str] = None):
        return agg.wait_pred_vs_actual(t0, t1, run_id)

    @router.get("/conversion")
    def conversion_by_zone(t0: float, t1: float, run_id: Optional[str] = None):
        return agg.conversion_by_zone(t0, t1, run_id)

    @router.get("/alerts")
    def alerts(open_only: bool = True, run_id: Optional[str] = None):
        return agg.alerts(open_only, run_id)

    @router.post("/alerts/{alert_id}/ack")
    def ack_alert(alert_id: int, run_id: Optional[str] = None):
        row = agg.ack_alert(alert_id, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no alert {alert_id}")
        return row

    @router.post("/alerts/ack-all")
    def ack_all(run_id: Optional[str] = None):
        return agg.ack_all(run_id)

    @router.get("/kpi")
    def kpi_series(t0: float, t1: float, key: Optional[str] = None,
                   run_id: Optional[str] = None):
        """The pre-aggregated 15-minute buckets the history charts plot."""
        return agg.kpi_series(t0, t1, key, run_id)

    @router.get("/stores")
    def stores():
        return agg.stores()

    @router.get("/stores/details")
    def store_details():
        return agg.store_details()

    @router.get("/runs")
    def runs():
        return agg.runs()

    return router
