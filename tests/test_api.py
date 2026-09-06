"""backend/api.py — FastAPI router over the aggregates.

No TestClient here: the newest starlette's test client raises a DeprecationWarning
which this repo escalates to an error, and the board must not carry an HTTP test
dependency. Instead the registered route endpoints are called directly.
"""
import pytest

from backend.aggregates import Aggregates
from backend.api import create_router
from backend.db import DB


@pytest.fixture
def router(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    db.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0, "clipset": "sim"})
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 5})
    db.drain()
    agg = Aggregates(db, run_id="r1")
    yield create_router(agg)
    db.close()


def _endpoint(router, name):
    for route in router.routes:
        if getattr(route, "name", None) == name:
            return route.endpoint
    raise KeyError(name)


def test_live_endpoint(router):
    assert _endpoint(router, "live_summary")()["occupancy"] == 5


def test_stores_endpoint(router):
    assert _endpoint(router, "stores")() == ["demo-01"]


def test_footfall_endpoint(router):
    assert isinstance(_endpoint(router, "footfall_by_day")(n=7), list)


def test_all_planned_routes_are_registered(router):
    names = {getattr(r, "name", None) for r in router.routes}
    assert {
        "live_summary", "footfall_by_hour", "footfall_by_day", "dwell_by_zone",
        "heatmap", "stockout_durations", "replenishment_list", "queue_status",
        "wait_pred_vs_actual", "conversion_by_zone", "alerts", "stores",
    } <= names


# ---- ack (the plan's W2 §3 ack endpoint) -----------------------------------

@pytest.fixture
def alerting_router(tmp_path):
    db = DB(tmp_path / "a.db", batch_interval_s=0.01)
    db.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0,
                      "clipset": "sim"})
    db.insert("alert", {"run_id": "r1", "t": 1.0, "severity": "warning",
                        "rule": "stockout", "message": "A1 empty", "acked": 0})
    db.drain()
    agg = Aggregates(db, run_id="r1")
    yield create_router(agg), agg
    db.close()


def test_ack_endpoint_closes_an_alert(alerting_router):
    router, agg = alerting_router
    alert_id = agg.alerts()[0]["id"]
    assert _endpoint(router, "ack_alert")(alert_id=alert_id)["acked"] == 1
    assert _endpoint(router, "alerts")(open_only=True) == []


def test_ack_endpoint_404s_on_an_unknown_alert(alerting_router):
    from fastapi import HTTPException
    router, _ = alerting_router
    with pytest.raises(HTTPException) as e:
        _endpoint(router, "ack_alert")(alert_id=999999)
    assert e.value.status_code == 404


def test_ack_all_endpoint(alerting_router):
    router, _ = alerting_router
    assert _endpoint(router, "ack_all")()["n"] == 0


# ---- the endpoints added for the dashboard ---------------------------------

def test_every_route_the_dashboard_needs_is_registered(router):
    names = {getattr(r, "name", None) for r in router.routes}
    assert {
        "health_check", "live_summary", "footfall_by_hour", "footfall_by_day",
        "dwell_by_zone", "dwell_littles_law", "heatmap", "stockout_durations",
        "restock_stats", "replenishment_list", "queue_status",
        "wait_pred_vs_actual", "conversion_by_zone", "alerts", "ack_alert",
        "ack_all", "kpi_series", "stores", "store_details", "runs",
    } <= names


def test_footfall_by_hour_defaults_to_today(router):
    # The dashboard calls it with no arguments on first paint.
    assert isinstance(_endpoint(router, "footfall_by_hour")(), list)


def test_littles_law_endpoint(router):
    out = _endpoint(router, "dwell_littles_law")(t0=0.0, t1=100.0)
    assert "mean_dwell_s" in out and "lambda_per_s" in out


def test_kpi_endpoint(router):
    assert _endpoint(router, "kpi_series")(t0=0.0, t1=10_000.0) == []


def test_replenishment_prices_through_the_bound_planogram(tmp_path):
    """The router has no planogram argument, so the Aggregates it is built with
    must carry one — otherwise every revenue rate silently reads zero."""
    db = DB(tmp_path / "p.db", batch_interval_s=0.01)
    db.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0,
                      "clipset": "sim"})
    db.insert("stockout", {"run_id": "r1", "roi": "A1", "sku": "MAGGI-70G",
                           "t_start": 0.0, "t_end": None})
    db.insert("occupancy", {"run_id": "r1", "t": 3600.0, "count": 1})
    db.drain()
    planogram = [{"sku": "MAGGI-70G", "name": "Maggi", "unit_price": 14,
                  "expected_sales_per_hour": 6}]
    r = create_router(Aggregates(db, run_id="r1", planogram=planogram))
    try:
        row = _endpoint(r, "replenishment_list")()[0]
        assert row["revenue_rate"] == pytest.approx(84.0)
        assert row["lost_revenue_so_far"] == pytest.approx(84.0)
    finally:
        db.close()


def test_health_defaults_when_nothing_is_bound(router):
    assert _endpoint(router, "health_check")() == {"ok": True}


def test_run_id_is_optional_on_every_query_endpoint(router):
    """FastAPI must see these as optional query params, not required ones."""
    import inspect
    for route in router.routes:
        sig = inspect.signature(route.endpoint)
        if "run_id" in sig.parameters:
            assert sig.parameters["run_id"].default is None
