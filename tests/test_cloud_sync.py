"""backend/cloud_sync.py — aggregates-only, retries, waits for a synced clock."""
import pytest

from backend.cloud_sync import CloudSync, kpi_bucket
from backend.db import DB


class FakeClock:
    def __init__(self, synced=True):
        self.synced = synced


@pytest.fixture
def db(tmp_path):
    d = DB(tmp_path / "r.db", batch_interval_s=0.01)
    yield d
    d.close()


def test_kpi_bucket_rounds_to_15m():
    assert kpi_bucket(0) == 0
    assert kpi_bucket(899) == 0
    assert kpi_bucket(900) == 900
    assert kpi_bucket(901) == 900


def test_sync_sends_only_pending_rows(db):
    for bucket, key, value in [(0, "footfall", 10), (900, "footfall", 20),
                               (1800, "footfall", 30)]:
        db.insert("kpi_15m", {"run_id": "r1", "t_bucket": bucket, "key": key,
                              "value": value})
    db.insert("sync_state", {"key": "kpi", "last_bucket": 900})
    db.drain()

    sent = []
    sync = CloudSync(db, "http://example", _sender=lambda url, data: sent.append(data))
    n, latest = sync.sync_once()
    assert n == 1 and latest == 1800
    assert len(sent) == 1
    db.drain()
    assert db.query_one("SELECT last_bucket FROM sync_state")["last_bucket"] == 1800


def test_sync_skips_when_clock_unsynced(db):
    sync = CloudSync(db, "http://example", clock=FakeClock(synced=False))
    assert sync.sync_once() == (0, 0)


def test_sync_retries_then_reports_failure(db):
    db.insert("kpi_15m", {"run_id": "r1", "t_bucket": 900, "key": "footfall", "value": 1})
    db.drain()
    calls = []

    def failing(url, data):
        calls.append(data)
        raise OSError("no route")

    sync = CloudSync(db, "http://example", _sender=failing)
    n, latest = sync.sync_once(retries=2, backoff_s=0.0)
    assert n == 0 and len(calls) == 2


def test_sync_no_pending_rows_is_a_noop(db):
    sync = CloudSync(db, "http://example")
    assert sync.sync_once() == (0, 0)
