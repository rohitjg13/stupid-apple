#!/usr/bin/env python3
"""Start a fresh run_id for a demo segment so numbers are clean.

    python3 tools/reset_run.py --run-id segment-1 --store-id demo-01 --clipset sim
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import DB


def reset(db, run_id, store_id, clipset, t=None):
    db.insert("run", {"run_id": run_id, "store_id": store_id,
                      "started": t or time.time(), "clipset": clipset})
    db.drain()
    return run_id


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="/var/lib/retail/retail.db")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--store-id", default="demo-01")
    ap.add_argument("--clipset", default="sim")
    a = ap.parse_args(argv)

    db = DB(a.db)
    reset(db, a.run_id, a.store_id, a.clipset)
    db.close()
    print(f"started run {a.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
