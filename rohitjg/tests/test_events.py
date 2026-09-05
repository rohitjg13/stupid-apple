"""Contract B: the Event envelope. docs/SHARED.md §5."""
import json
import pytest

from core.events import EVENT_TYPES, Event


def test_envelope_fields():
    e = Event(t=1.5, store_id="demo-01", stream="overhead", zone_id="door",
              event_type="tripwire", payload={"dir": "in", "tripwire_id": "door"})
    assert (e.t, e.store_id, e.stream, e.zone_id) == (1.5, "demo-01", "overhead", "door")


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError):
        Event(t=0, store_id="demo-01", stream="-", zone_id=None,
              event_type="shelf_restock", payload={})


def test_every_documented_type_is_registered():
    assert EVENT_TYPES.keys() >= {
        "tripwire", "occupancy", "visit", "heatmap", "shelf_fill",
        "stockout_start", "stockout_end", "planogram_violation",
        "lane_occ", "queue_estimate", "pos_txn", "alert",
    }


@pytest.mark.parametrize("event_type,payload", [
    ("tripwire", {"dir": "in", "tripwire_id": "door"}),
    ("occupancy", {"count": 3}),
    ("visit", {"track_id": 4, "t_enter": 1.0, "t_exit": 9.0}),
    ("heatmap", {"t_bucket": 170, "tiles": [[1, 2, 3]]}),
    ("shelf_fill", {"fill": 120}),
    ("stockout_start", {"sku": "MAGGI-70G", "facing": "A1"}),
    ("stockout_end", {"sku": "MAGGI-70G", "facing": "A1"}),
    ("planogram_violation", {"expected_sku": "MAGGI-70G", "observed": "empty"}),
    ("lane_occ", {"lane": 1, "cells": [0] * 16}),
    ("queue_estimate", {"lane": 1, "count": 4, "pred_wait_s": 120}),
    ("pos_txn", {"lane": 1, "items": 7, "amount": 340.0, "zone_hint": "Z2"}),
    ("alert", {"severity": "high", "rule": "wait_over_target", "message": "open counter 2"}),
])
def test_documented_payloads_accepted(event_type, payload):
    Event(t=0.0, store_id="demo-01", stream="-", zone_id=None,
          event_type=event_type, payload=payload)


def test_missing_payload_key_rejected():
    with pytest.raises(ValueError):
        Event(t=0.0, store_id="demo-01", stream="overhead", zone_id=None,
              event_type="tripwire", payload={"dir": "in"})   # no tripwire_id


def test_extra_payload_key_rejected():
    """Nobody invents a field. docs/SHARED.md, then ask."""
    with pytest.raises(ValueError):
        Event(t=0.0, store_id="demo-01", stream="overhead", zone_id=None,
              event_type="occupancy", payload={"count": 1, "confidence": 0.9})


def test_bad_stream_rejected():
    with pytest.raises(ValueError):
        Event(t=0.0, store_id="demo-01", stream="checkout", zone_id=None,
              event_type="occupancy", payload={"count": 1})


def test_to_json_round_trips():
    e = Event(t=1.5, store_id="demo-01", stream="shelf", zone_id="A1",
              event_type="shelf_fill", payload={"fill": 40})
    d = json.loads(e.to_json())
    assert d == {"t": 1.5, "store_id": "demo-01", "stream": "shelf",
                 "zone_id": "A1", "event_type": "shelf_fill", "payload": {"fill": 40}}
