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
    # zone_hint is a *zone*, not a facing: it is what the conversion join needs,
    # and what a real POS would produce via SKU -> planogram -> zone.
    assert e.payload["zone_hint"] in {z["id"] for z in CFG.zones}


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


# ---- zone_hint follows the last 10 minutes of dwell ------------------------

def test_zone_hint_is_uniform_before_any_visit(stub):
    # Cold start must not silently favour whichever zone is first in zones.json.
    w = stub._zone_weights(0.0)
    assert pytest.approx(w.tolist()) == [1.0 / len(CFG.zones)] * len(CFG.zones)


def test_zone_hint_follows_observed_dwell(stub):
    stub.observe_visit(0.0, "aisle_b", 300.0)
    hints = {stub.txn(1.0, 1).payload["zone_hint"] for _ in range(20)}
    # All the observed dwell is in aisle_b, so that is the only zone sampled.
    assert hints == {"aisle_b"}


def test_dwell_older_than_ten_minutes_is_forgotten(stub):
    from backend.pos_stub import DWELL_WINDOW_S
    stub.observe_visit(0.0, "aisle_b", 300.0)
    w = stub._zone_weights(DWELL_WINDOW_S + 1.0)
    assert pytest.approx(w.tolist()) == [1.0 / len(CFG.zones)] * len(CFG.zones)


def test_lanes_come_from_the_clipset(stub):
    assert stub.lanes == [1, 2]


def test_on_count_resets_the_idle_timer(stub):
    """A drop-triggered txn is still a txn: the idle timer must not double up."""
    assert stub.on_count(0.0, 1, 3) is None
    assert stub.on_count(10.0, 1, 2) is not None      # dropped -> txn on lane 1
    lanes = [e.payload["lane"] for e in stub.idle_tick(20.0)]
    assert lanes == [2]                               # lane 1 just transacted


# ---- mu, derived from POS timestamps exactly as a real POS would -----------

def test_service_rate_is_one_over_mean_gap():
    from backend.pos_stub import service_rate

    class T:
        def __init__(self, t, lane):
            self.t, self.payload = t, {"lane": lane}

    # gaps of 10 s and 30 s -> mean 20 s -> mu = 0.05/s
    assert service_rate([T(0, 1), T(10, 1), T(40, 1)]) == pytest.approx(0.05)


def test_service_rate_filters_by_lane():
    from backend.pos_stub import service_rate

    class T:
        def __init__(self, t, lane):
            self.t, self.payload = t, {"lane": lane}

    txns = [T(0, 1), T(5, 2), T(20, 1), T(500, 2)]
    assert service_rate(txns, lane=1) == pytest.approx(1 / 20.0)


def test_service_rate_needs_two_transactions():
    from backend.pos_stub import service_rate

    class T:
        def __init__(self, t, lane):
            self.t, self.payload = t, {"lane": lane}

    assert service_rate([]) is None
    assert service_rate([T(0, 1)]) is None
