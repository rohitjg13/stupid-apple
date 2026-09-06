"""Standalone HTTP app so a frontend has something to talk to today.

The dashboard lead owns the real server and mounts `api.create_router` into it
(05-backend-queue.md W1 §3). This module is the same router wrapped in a
minimal app plus two things a browser needs and a router cannot provide:

- CORS, so a frontend dev server on another port can call the API
- `GET /api/events`, a **Server-Sent Events** feed of the live bus

SSE and not a websocket on purpose: the board's dependency list is fixed
(numpy, opencv-python-headless, fastapi, uvicorn, pyyaml, scipy) and plain
uvicorn has no websocket implementation, so a websocket route would import
fine and then fail on the first connection. SSE is ordinary HTTP, works
through `EventSource` in every browser, and needs nothing extra.

The feed is lossy by design: a slow browser drops events rather than stalling
the pipeline behind it, exactly like every other bus subscriber.
"""
from __future__ import annotations

import json
import logging
import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.api import create_router

log = logging.getLogger(__name__)

HEARTBEAT_S = 15.0


class EventStream:
    """Fan the bus out to any number of SSE clients, dropping when they lag."""

    def __init__(self, bus=None, maxsize=256):
        self._clients = []
        self._lock = threading.Lock()
        self._maxsize = int(maxsize)
        self.dropped = 0
        if bus is not None:
            bus.subscribe("*", self.on_event)

    def on_event(self, event):
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                self.dropped += 1       # a stalled browser never stalls the store

    def add(self, maxsize=None):
        q = queue.Queue(maxsize=self._maxsize if maxsize is None else maxsize)
        with self._lock:
            self._clients.append(q)
        return q

    def remove(self, q):
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    @property
    def clients(self):
        with self._lock:
            return len(self._clients)

    def stream(self, stop=None, heartbeat_s=HEARTBEAT_S, limit=None):
        """Yield SSE frames. Heartbeats keep proxies from closing the socket."""
        q = self.add()
        sent = 0
        try:
            while stop is None or not stop.is_set():
                try:
                    event = q.get(timeout=heartbeat_s)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield (f"event: {event.event_type}\n"
                       f"data: {event.to_json()}\n\n")
                sent += 1
                if limit is not None and sent >= limit:
                    return
        finally:
            self.remove(q)


def create_app(aggregates, bus=None, health=None, cors_origins=("*",),
               stop=None) -> FastAPI:
    app = FastAPI(title="Retail analytics", version="0.1.0")
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins),
                           allow_credentials=False, allow_methods=["*"],
                           allow_headers=["*"])
    api = create_router(aggregates, health=health)
    app.include_router(api)
    # Listed from the router directly: FastAPI keeps an included router as a
    # lazy placeholder in `app.routes`, which has no `.path` to read.
    endpoints = sorted(r.path for r in api.routes if hasattr(r, "path"))

    stream = EventStream(bus)
    app.state.aggregates = aggregates
    app.state.stream = stream

    @app.get("/api/events")
    def events():
        return StreamingResponse(
            stream.stream(stop=stop), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    def root():
        return {"api": "/api", "events": "/api/events",
                "run_id": aggregates.run_id,
                "endpoints": endpoints + ["/api/events"]}

    return app


def serve_in_thread(app, host="0.0.0.0", port=8000, log_level="warning"):
    """Run uvicorn on a daemon thread so the pipeline keeps the main thread.

    Returns (thread, server); call `server.should_exit = True` to stop it.
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="backend-api", daemon=True)
    thread.start()
    log.info("api serving", extra={"host": host, "port": port})
    return thread, server


def main(argv=None):
    """Serve an existing database with no pipeline attached:

        python -m backend.server --db retail.db --run-id history
    """
    import argparse

    import uvicorn

    from backend.aggregates import Aggregates
    from backend.db import DB
    from core.config import load_clipset

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="retail.db")
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    db = DB(a.db)
    cfg = load_clipset(a.config)
    run_id = a.run_id
    if run_id is None:
        row = db.query_one("SELECT run_id FROM run ORDER BY started DESC LIMIT 1")
        run_id = row["run_id"] if row else "run-0"
    agg = Aggregates(db, run_id=run_id, planogram=cfg.planogram)
    uvicorn.run(create_app(agg), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
