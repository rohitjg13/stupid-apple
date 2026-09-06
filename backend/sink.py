"""Bus subscriber: every event -> one DB insert, zero drops.

The backend is the sink for the whole event bus. It owns a single DB instance
and maps each `event_type` in docs/SHARED.md §5 to its table, exactly as the
schema lays out. Payload keys are validated by core/events.py before this ever
sees them; here we just translate.

`lane_occ.cells` is a 16-byte blob (16 x uint8), so it is packed from the list.
`pos_txn.zone_hint` may be None, which the schema allows (no NOT NULL).
"""
from __future__ import annotations

import logging

from backend.db import DB

log = logging.getLogger(__name__)


def pack_cells(cells) -> bytes:
    """16 lane cells (0..100) -> 16 bytes. The schema stores a BLOB."""
    if len(cells) != 16:
        raise ValueError(f"lane_occ cells must be 16 elements, got {len(cells)}")
    return bytes(int(c) & 0xFF for c in cells)


class Sink:
    def __init__(self, db: DB, run_id: str):
        self.db = db
        self.run_id = run_id

    def start_run(self, store_id, clipset, t):
        # REPLACE, not INSERT: re-running a demo segment (or the backfill) with
        # the same run_id must not blow up on the primary key and roll back the
        # events batched behind it.
        self.db.enqueue(
            "INSERT OR REPLACE INTO run (run_id, store_id, started, clipset) "
            "VALUES (?, ?, ?, ?)", (self.run_id, store_id, t, clipset))

    def on_event(self, event):
        rid = self.run_id
        t, z, p = event.t, event.zone_id, event.payload
        et = event.event_type

        if et == "tripwire":
            self.db.insert("tripwire", {"run_id": rid, "t": t, "dir": p["dir"]})
        elif et == "occupancy":
            self.db.insert("occupancy", {"run_id": rid, "t": t, "count": p["count"]})
        elif et == "visit":
            self.db.insert("visit", {"run_id": rid, "track_id": p["track_id"],
                                     "zone": z, "t_enter": p["t_enter"],
                                     "t_exit": p["t_exit"]})
        elif et == "heatmap":
            # (run_id, t_bucket, gx, gy) is the primary key, so two heatmap
            # events landing in the same bucket accumulate instead of failing.
            for gx, gy, count in p["tiles"]:
                self.db.enqueue(
                    "INSERT INTO heatmap (run_id, t_bucket, gx, gy, count) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id, t_bucket, gx, gy) "
                    "DO UPDATE SET count = count + excluded.count",
                    (rid, p["t_bucket"], gx, gy, count))
        elif et == "shelf_fill":
            self.db.insert("shelf_fill", {"run_id": rid, "t": t, "roi": z,
                                          "fill": p["fill"]})
        elif et == "stockout_start":
            self.db.insert("stockout", {"run_id": rid, "roi": z, "sku": p["sku"],
                                        "t_start": t, "t_end": None})
        elif et == "stockout_end":
            # Close the most recent still-open stockout for this facing so the
            # stockout table holds one row per incident, start-to-end.
            self.db.enqueue(
                "UPDATE stockout SET t_end = ? WHERE rowid = ("
                "SELECT rowid FROM stockout WHERE run_id = ? AND roi = ? "
                "AND sku = ? AND t_end IS NULL AND t_start <= ? "
                "ORDER BY t_start DESC LIMIT 1)",
                (t, rid, z, p["sku"], t))
        elif et == "planogram_violation":
            self.db.insert("planogram_violation", {"run_id": rid, "t": t, "roi": z,
                                                   "expected": p["expected_sku"],
                                                   "observed": p["observed"]})
        elif et == "lane_occ":
            self.db.insert("lane_occ", {"run_id": rid, "t": t, "lane": p["lane"],
                                        "cells": pack_cells(p["cells"])})
        elif et == "queue_estimate":
            self.db.insert("queue_estimate", {"run_id": rid, "t": t, "lane": p["lane"],
                                              "count": p["count"],
                                              "pred_wait_s": p["pred_wait_s"]})
        elif et == "pos_txn":
            self.db.insert("pos_txn", {"run_id": rid, "t": t, "lane": p["lane"],
                                       "items": p["items"], "amount": p["amount"],
                                       "zone_hint": p["zone_hint"]})
        elif et == "alert":
            self.db.insert("alert", {"run_id": rid, "t": t, "severity": p["severity"],
                                     "rule": p["rule"], "message": p["message"]})
        else:
            # core/events.py rejects unknown types; this is belt-and-braces for
            # the "*" subscriber which also sees events we did not opt into.
            log.warning("sink: unhandled event_type", extra={"event_type": et})

    def subscribe(self, bus):
        bus.subscribe("*", self.on_event)
        return self

    def drain(self, timeout=10.0):
        self.db.drain(timeout)
        return self
