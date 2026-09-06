"""FastAPI router over the aggregates; the dashboard server mounts it.

Signatures mirror the plan (05-backend-queue.md W1 §3): `/api/...` over the
`Aggregates` functions. No state lives here; the router is built with a bound
`Aggregates` instance, so tests can hit it with a temp DB and no network.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.aggregates import Aggregates


def create_router(aggregates: Aggregates) -> APIRouter:
    router = APIRouter(prefix="/api")
    agg = aggregates

    @router.get("/live")
    def live_summary():
        return agg.live_summary()

    @router.get("/footfall/hour")
    def footfall_by_hour(day: float, run_id: str = None):
        return agg.footfall_by_hour(day, run_id)

    @router.get("/footfall/day")
    def footfall_by_day(n: int = 7, run_id: str = None):
        return agg.footfall_by_day(n, run_id)

    @router.get("/dwell")
    def dwell_by_zone(t0: float, t1: float, run_id: str = None):
        return agg.dwell_by_zone(t0, t1, run_id)

    @router.get("/heatmap")
    def heatmap(t0: float, t1: float, run_id: str = None):
        return agg.heatmap(t0, t1, run_id)

    @router.get("/stockouts")
    def stockout_durations(t0: float, t1: float, run_id: str = None):
        return agg.stockout_durations(t0, t1, run_id)

    @router.get("/replenishment")
    def replenishment_list(run_id: str = None):
        return agg.replenishment_list(run_id)

    @router.get("/queue")
    def queue_status(run_id: str = None):
        return agg.queue_status(run_id)

    @router.get("/queue/wait-pred-vs-actual")
    def wait_pred_vs_actual(t0: float, t1: float, run_id: str = None):
        return agg.wait_pred_vs_actual(t0, t1, run_id)

    @router.get("/conversion")
    def conversion_by_zone(t0: float, t1: float, run_id: str = None):
        return agg.conversion_by_zone(t0, t1, run_id)

    @router.get("/alerts")
    def alerts(open_only: bool = True, run_id: str = None):
        return agg.alerts(open_only, run_id)

    @router.get("/stores")
    def stores():
        return agg.stores()

    return router
