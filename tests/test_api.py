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
