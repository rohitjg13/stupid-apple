"""Read a real POS/ERP CSV export into pos_txn events.

The CSV columns mirror the pos_txn payload so a real integration is a file away:

    t,lane,items,amount,zone_hint

`zone_hint` may be blank (None). Service rate mu is derived from the timestamp
spacing exactly as it would be from a live POS, so the queue model does not
care whether transactions came from a stub or a file.
"""
from __future__ import annotations

import csv

from core.events import Event


def parse_transactions(path, store_id, stream="-"):
    """Yield pos_txn events from a CSV file."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            lane = int(row["lane"])
            items = int(row["items"])
            amount = float(row["amount"])
            zone_hint = row.get("zone_hint") or None
            yield Event(t, store_id, stream, None, "pos_txn", {
                "lane": lane, "items": items, "amount": amount,
                "zone_hint": zone_hint,
            })
