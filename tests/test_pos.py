"""backend/pos_stub.py and backend/pos_csv.py — synthetic and file POS adapters."""
import csv

import pytest

from backend.pos_csv import parse_transactions
from backend.pos_stub import PosStub
from core.config import load_clipset

CFG = load_clipset("config/sim")


@pytest.fixture
def stub():
    return PosStub(CFG, seed=0)


def test_txn_payload_is_valid(stub):
    e = stub.txn(100.0, 1)
    assert e.event_type == "pos_txn"
    assert e.store_id == "demo-01"
    assert e.payload["lane"] == 1
    assert e.payload["items"] >= 1
    assert e.payload["amount"] > 0
    assert e.payload["zone_hint"] in {p["facing"] for p in CFG.planogram}


def test_on_count_emits_only_on_drop(stub):
    assert stub.on_count(1.0, 1, 2) is None     # rising -> nothing
    assert stub.on_count(2.0, 1, 3) is None
    e = stub.on_count(3.0, 1, 2)                 # dropped -> checkout
    assert e is not None and e.event_type == "pos_txn"


def test_idle_tick_emits_for_quiet_lanes(stub):
    assert len(stub.idle_tick(0.0)) == 2         # both lanes have never seen a txn
    assert len(stub.idle_tick(10.0)) == 0        # within the idle window


def test_transactions_honours_since(stub):
    stub.txn(10.0, 1)
    stub.txn(20.0, 2)
    assert len(list(stub.transactions(since=15.0))) == 1


def test_same_seed_same_transactions():
    a = PosStub(CFG, seed=7)
    b = PosStub(CFG, seed=7)
    for lane in (1, 2):
        assert a.txn(0.0, lane).to_json() == b.txn(0.0, lane).to_json()


def test_pos_csv_round_trips(tmp_path):
    p = tmp_path / "txns.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "lane", "items", "amount", "zone_hint"])
        w.writerow([1.0, 2, 3, 42.0, ""])
        w.writerow([2.0, 1, 1, 14.0, "A1"])
    events = list(parse_transactions(p, "demo-01"))
    assert len(events) == 2
    assert events[0].payload["zone_hint"] is None
    assert events[1].payload["zone_hint"] == "A1"
    assert events[1].payload["amount"] == 14.0
