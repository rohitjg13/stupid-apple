#!/usr/bin/env python3
"""Backfill 7 virtual days of history plus 2 fake stores for the dashboard.

Runs the sim in --fast against `sources/sim.py --history-days`, then writes the
resulting events into the backend DB under the given run_id/store_id. Multi-store
view: demo-01 (real) plus demo-02/demo-03 (backfilled).

    python3 tools/backfill_history.py [--db /var/lib/retail/retail.db] [--days 7]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.aggregates import Aggregates
from backend.db import DB
from backend.sink import Sink
from core.config import load_clipset
from sources.sim import SimSource

log = logging.getLogger("backfill")


def backfill(db, cfg, store_id, run_id, days, seed, rollup=True):
    """Write `days` of synthetic history under one run_id, then pre-aggregate it.

    The 15-minute rollup is not optional dressing: `kpi_15m` is what the
    history charts read on the board and the only table cloud sync uploads, so
    a backfill that skipped it would leave both empty at boot.
    """
    src = SimSource(cfg, stream="overhead", seed=seed)
    sink = Sink(db, run_id=run_id)
    sink.start_run(store_id, cfg.path.name if hasattr(cfg, "path") else "sim", 0.0)
    n, t0, t1 = 0, None, None
    try:
        for event in src.history(days=days):
            sink.on_event(event)
            n += 1
            t0 = event.t if t0 is None else min(t0, event.t)
            t1 = event.t if t1 is None else max(t1, event.t)
    finally:
        src.close()
    sink.drain()
    if rollup and t0 is not None:
        Aggregates(db, run_id=run_id, planogram=cfg.planogram).kpi_rollup(t0, t1 + 1)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="/var/lib/retail/retail.db")
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_clipset(a.config)
    db = DB(a.db, batch_interval_s=0.05)

    total = 0
    for store_id, run_id, seed in [("demo-01", "history", a.seed),
                                   ("demo-02", "history-demo02", a.seed + 1),
                                   ("demo-03", "history-demo03", a.seed + 2)]:
        n = backfill(db, cfg, store_id, run_id, a.days, seed)
        total += n
        log.info("backfilled", extra={"store": store_id, "events": n})

    db.close()
    print(f"backfilled {total} events across 3 stores into {a.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
