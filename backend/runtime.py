"""The backend as one running thing: bus in, database and events out.

`main.py` builds one of these and calls two methods — `on_frame` for every
FrameResult and `tick` once a loop. Everything else the backend owns hangs off
it:

    bus --*--> Sink          -> SQLite (single writer, batched commits)
           |-> Alerts        -> `alert` events back onto the bus
           |-> QueueEngine   -> `queue_estimate` every 2 s
           |-> PosStub       -> `pos_txn` when a lane's count drops

    frame ---> lane_occ (2 Hz, derived from FrameResult.lane_occupancy)

The backend is the only module that reads raw `lane_occupancy`, which is why
`lane_occ` is emitted from here rather than by the shopper pipeline (SHARED.md
§5 lists the backend lead as its source). Its own producers feed the queue
engine directly and publish the event for everyone else; going out to the bus
and back would only add a race between the two.

Maintenance — KPI rollup, retention and cloud sync — runs on `tick` at a slow
cadence and never blocks the pipeline: every one of them is a bounded amount of
work on the caller's thread, and cloud sync fails closed when the network or
the clock is not there.
"""
from __future__ import annotations

import logging
import threading

from backend.aggregates import Aggregates
from backend.alerts import Alerts
from backend.cloud_sync import CloudSync
from backend.db import DB
from backend.pos_stub import PosStub
from backend.queue_model import QueueEngine, cells_by_lane
from backend.sink import Sink
from core.events import Event

log = logging.getLogger(__name__)

LANE_OCC_INTERVAL_S = 0.5       # the frozen contract says lane_occ is 2 Hz
KPI_INTERVAL_S = 60.0
SYNC_INTERVAL_S = 900.0
RETENTION_INTERVAL_S = 3600.0


class BackendRuntime:
    def __init__(self, cfg, bus, db_path=None, db=None, run_id=None, store_id=None,
                 clock=None, seed=0, cloud_url=None, pos=None, started=0.0,
                 kpi_interval_s=KPI_INTERVAL_S, sync_interval_s=SYNC_INTERVAL_S,
                 retention_interval_s=RETENTION_INTERVAL_S):
        if db is None and db_path is None:
            raise ValueError("BackendRuntime needs a db or a db_path")
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.store_id = store_id or cfg.store_id
        self.run_id = run_id or f"run-{int(started or 0)}"
        self._owns_db = db is None
        self.db = db if db is not None else DB(db_path)

        self.sink = Sink(self.db, self.run_id)
        self.aggregates = Aggregates(self.db, run_id=self.run_id,
                                     planogram=cfg.planogram)
        self.alerts = Alerts(cfg, bus=bus, store_id=self.store_id)
        self.queue = QueueEngine(
            counters=int(cfg.checkout.get("counters", 2)),
            target_wait_s=float(cfg.checkout.get("target_wait_s", 180.0)))
        self.pos = pos if pos is not None else PosStub(cfg, store_id=self.store_id,
                                                       seed=seed)
        cloud = cfg.cloud or {}
        url = cloud_url or (cloud.get("url") if cloud.get("enabled") else None)
        self.cloud = CloudSync(self.db, url, clock=clock,
                               store_id=self.store_id) if url else None

        self.kpi_interval_s = float(kpi_interval_s)
        self.sync_interval_s = float(sync_interval_s)
        self.retention_interval_s = float(retention_interval_s)

        self._lock = threading.RLock()
        self._last_lane_pub = {}
        self._last_kpi_t = None
        self._last_sync_t = None
        self._last_retention_t = None
        self._started = False
        self.published = 0

    # ---- lifecycle ---------------------------------------------------------
    def start(self, started=None, clipset=None):
        """Record the run and subscribe to the bus. Idempotent."""
        if self._started:
            return self
        self._started = True
        t = started if started is not None else (self.clock.now() if self.clock else 0.0)
        self.sink.start_run(self.store_id, clipset or self.cfg.path.name, t)
        self.sink.subscribe(self.bus)
        self.bus.subscribe("*", self.on_event)
        if self.clock is not None and not self.clock.synced:
            self.alerts.clock_unsynced(t)
        log.info("backend runtime started",
                 extra={"run_id": self.run_id, "store_id": self.store_id})
        return self

    def close(self, timeout=10.0):
        try:
            self.db.drain(timeout)
        except Exception:
            log.exception("backend drain failed on close")
        if self._owns_db:
            self.db.close(timeout)

    # ---- bus input ---------------------------------------------------------
    def on_event(self, event):
        """Everything the *rest* of the pipeline publishes lands here.

        Events this runtime produced itself are fed to the models at the point
        of production, so they are skipped here rather than counted twice.
        """
        et = event.event_type
        with self._lock:
            if et == "tripwire" and event.payload["dir"] == "out":
                # Walking away from the floor is walking towards checkout: the
                # plan's preferred arrival signal when the door count is clean.
                self.queue.on_arrival(event.t)
            elif et == "visit":
                self.pos.observe_visit(event.t, event.zone_id,
                                       event.payload["t_exit"] - event.payload["t_enter"])
            self.alerts.on_event(event)

    # ---- frame input -------------------------------------------------------
    def on_frame(self, frame, t):
        """Derive and publish `lane_occ` from a raw overhead FrameResult."""
        if frame.stream != "overhead" or frame.result is None:
            return []
        published = []
        with self._lock:
            for lane, cells in cells_by_lane(self.cfg.lanes,
                                             frame.result["lane_occupancy"]).items():
                if t - self._last_lane_pub.get(lane, float("-inf")) < LANE_OCC_INTERVAL_S:
                    continue
                self._last_lane_pub[lane] = t
                ev = Event(t, self.store_id, "overhead", None, "lane_occ",
                           {"lane": lane, "cells": cells})
                count = self.queue.on_lane_occ(t, lane, cells)
                self._publish(ev)
                published.append(ev)
                # A lane shortening by one is a customer served: that is what
                # the POS stub turns into a transaction, and what mu is built
                # from. A real POS replaces this and nothing else changes.
                txn = self.pos.on_count(t, lane, count)
                if txn is not None:
                    self.queue.on_departure(txn.t, lane)
                    self._publish(txn)
        return published

    # ---- periodic ----------------------------------------------------------
    def tick(self, t):
        """Call once a loop. Publishes estimates and runs slow maintenance."""
        out = []
        with self._lock:
            if self.queue.due(t):
                for est in self.queue.estimates(t):
                    out.append(self._publish(
                        Event(t, self.store_id, "overhead", None, "queue_estimate",
                              est)))
                self.alerts.staffing(t, self.queue.advice(t))
            for txn in self.pos.idle_tick(t):
                self.queue.on_departure(txn.t, txn.payload["lane"])
                out.append(self._publish(txn))
            self.alerts.check_stockout_duration(t)
        self._maintenance(t)
        return out

    def _publish(self, event):
        self.bus.publish(event)
        self.published += 1
        return event

    def _maintenance(self, t):
        if _due(self._last_kpi_t, t, self.kpi_interval_s):
            self._last_kpi_t = t
            # Two buckets: the one in progress and the one that just closed, so
            # a bucket is never left half-aggregated at a boundary.
            self.aggregates.kpi_rollup(t - 900.0, t)
        if self.cloud is not None and _due(self._last_sync_t, t, self.sync_interval_s):
            self._last_sync_t = t
            n, bucket = self.cloud.sync_once()
            if n:
                log.info("cloud sync", extra={"rows": n, "bucket": bucket})
        if _due(self._last_retention_t, t, self.retention_interval_s):
            self._last_retention_t = t
            self.db.retention(now=t)
            self.db.rollup(now=t)


def _due(last, t, interval):
    return last is None or t - last >= interval
