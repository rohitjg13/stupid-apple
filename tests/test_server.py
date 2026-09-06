"""backend/server.py — the app a frontend actually talks to.

No TestClient (see tests/test_api.py for why); routes and the SSE generator are
exercised directly, which is also the only way to assert the stream's
drop-rather-than-stall behaviour deterministically.
"""
import json
import threading

import pytest

from backend.aggregates import Aggregates
from backend.db import DB
from backend.server import EventStream, create_app
from core.bus import Bus
from core.events import Event


def ev(t=1.0, event_type="occupancy", payload=None):
    return Event(t, "demo-01", "overhead", None, event_type,
                 payload if payload is not None else {"count": 3})


@pytest.fixture
def app(tmp_path):
    db = DB(tmp_path / "r.db", batch_interval_s=0.01)
    db.insert("run", {"run_id": "r1", "store_id": "demo-01", "started": 0.0,
                      "clipset": "sim"})
    db.insert("occupancy", {"run_id": "r1", "t": 1.0, "count": 5})
    db.drain()
    agg = Aggregates(db, run_id="r1")
    yield create_app(agg, bus=Bus(), health=lambda: {"ok": True, "frames": 7})
    db.close()


def _iter_routes(app):
    """Flatten the app's routes.

    FastAPI 0.141 keeps an included router as a lazy `_IncludedRouter`
    placeholder rather than splicing its routes into `app.routes`, so mounted
    routes have to be walked through `original_router`.
    """
    stack = list(app.routes)
    while stack:
        r = stack.pop()
        inner = getattr(r, "original_router", None)
        if inner is not None:
            stack.extend(inner.routes)
            continue
        yield r


def _paths(app):
    return {getattr(r, "path", None) for r in _iter_routes(app)}


def _route(app, path, method="GET"):
    for r in _iter_routes(app):
        if getattr(r, "path", None) == path and method in getattr(r, "methods", ()):
            return r.endpoint
    raise KeyError(path)


# ---- app wiring ------------------------------------------------------------

def test_api_routes_are_mounted(app):
    paths = _paths(app)
    assert "/api/live" in paths
    assert "/api/queue" in paths
    assert "/api/events" in paths


def test_root_advertises_the_api(app):
    body = _route(app, "/")()
    assert body["run_id"] == "r1"
    assert "/api/events" in body["endpoints"]
    assert body["events"] == "/api/events"


def test_health_is_served_from_the_callable(app):
    assert _route(app, "/api/health")() == {"ok": True, "frames": 7}


def test_health_accepts_a_plain_dict(tmp_path):
    db = DB(tmp_path / "h.db", batch_interval_s=0.01)
    try:
        app = create_app(Aggregates(db, run_id="r1"), health={"ok": False})
        assert _route(app, "/api/health")() == {"ok": False}
    finally:
        db.close()


def test_cors_is_enabled_so_a_dev_frontend_can_call_it(app):
    from fastapi.middleware.cors import CORSMiddleware
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


def test_live_endpoint_reads_through_to_the_database(app):
    assert _route(app, "/api/live")()["occupancy"] == 5


# ---- SSE -------------------------------------------------------------------

def test_stream_yields_one_frame_per_event():
    stream = EventStream()
    q = stream.add()
    stream.on_event(ev(1.0))
    stream.on_event(ev(2.0, "alert", {"severity": "info", "rule": "r",
                                      "message": "m"}))
    assert q.qsize() == 2
    stream.remove(q)


def test_stream_formats_valid_sse_frames():
    stream = EventStream()
    stop = threading.Event()
    gen = stream.stream(stop=stop, heartbeat_s=0.01, limit=1)

    # The generator has to be running before the event is published, so it is
    # started here and the event pushed from this thread once it is registered.
    def publish_when_ready():
        while stream.clients == 0:
            pass
        stream.on_event(ev(1.0))

    t = threading.Thread(target=publish_when_ready, daemon=True)
    t.start()
    frame = next(gen)
    while frame.startswith(":"):            # skip keep-alives
        frame = next(gen)
    t.join(timeout=2.0)

    assert frame.startswith("event: occupancy\n")
    assert frame.endswith("\n\n")
    data = frame.split("data: ", 1)[1].strip()
    assert json.loads(data)["payload"] == {"count": 3}
    gen.close()


def test_stream_emits_heartbeats_when_idle():
    stream = EventStream()
    gen = stream.stream(heartbeat_s=0.01)
    assert next(gen) == ": keep-alive\n\n"
    gen.close()


def test_stream_unregisters_the_client_when_it_closes():
    stream = EventStream()
    gen = stream.stream(heartbeat_s=0.01)
    next(gen)
    assert stream.clients == 1
    gen.close()
    assert stream.clients == 0


def test_a_stalled_client_drops_events_instead_of_stalling_the_bus():
    """The whole point: a browser that stops reading must not back up the
    pipeline behind it."""
    stream = EventStream()
    stream.add(maxsize=2)
    for i in range(10):
        stream.on_event(ev(float(i)))
    assert stream.dropped == 8


def test_stream_fans_out_to_every_client():
    stream = EventStream()
    a, b = stream.add(), stream.add()
    stream.on_event(ev(1.0))
    assert a.qsize() == 1 and b.qsize() == 1


def test_stream_subscribes_to_the_bus():
    bus = Bus()
    stream = EventStream(bus)
    q = stream.add()
    bus.publish(ev(1.0))
    bus.drain()
    assert q.get(timeout=2.0).payload == {"count": 3}


def test_stop_event_ends_the_stream():
    stream = EventStream()
    stop = threading.Event()
    stop.set()
    assert list(stream.stream(stop=stop, heartbeat_s=0.01)) == []
