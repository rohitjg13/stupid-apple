"""Sync 15-minute KPI aggregates to a receiver when the network is there.

Offline-first: sync never blocks the pipeline, never sends raw events, and
retries with backoff. It only uploads `kpi_15m` rows newer than the stored
`sync_state` cursor, and only when the clock is synced (a RTC-less board that
booted at epoch-1970 must not upload garbage timestamps).
"""
from __future__ import annotations

import json
import logging
import time
from urllib import error, request

log = logging.getLogger(__name__)


class CloudSync:
    def __init__(self, db, url, clock=None, _sender=None):
        self.db = db
        self.url = url
        self.clock = clock
        self._sender = _sender or _http_post

    def _cursor(self):
        row = self.db.query_one("SELECT last_bucket FROM sync_state WHERE key = 'kpi'")
        return int(row["last_bucket"]) if row else 0

    def _save_cursor(self, bucket):
        self.db.enqueue(
            "INSERT INTO sync_state (key, last_bucket) VALUES ('kpi', ?) "
            "ON CONFLICT(key) DO UPDATE SET last_bucket = ?", (bucket, bucket))

    def unsynced(self):
        return self.clock is not None and not self.clock.synced

    def sync_once(self, retries=3, backoff_s=2.0):
        """Try to upload pending KPI rows; return (uploaded, latest_bucket)."""
        if self.unsynced():
            log.warning("clock unsynced; skipping cloud sync")
            return 0, self._cursor()

        cursor = self._cursor()
        rows = self.db.query(
            "SELECT t_bucket, key, value FROM kpi_15m WHERE t_bucket > ? "
            "ORDER BY t_bucket", (cursor,))
        if not rows:
            return 0, cursor

        payload = json.dumps({"kpi": rows}).encode()
        latest = rows[-1]["t_bucket"]
        for attempt in range(retries):
            try:
                self._sender(self.url, payload)
                self._save_cursor(latest)
                return len(rows), latest
            except (error.URLError, OSError, TimeoutError) as e:
                if attempt == retries - 1:
                    log.warning("cloud sync failed", extra={"error": str(e)})
                    return 0, cursor
                time.sleep(backoff_s * (2 ** attempt))
        return 0, cursor


def _http_post(url, data):
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"},
                          method="POST")
    with request.urlopen(req, timeout=5.0) as resp:
        resp.read()


def kpi_bucket(t):
    """15-minute bucket (unix seconds) for a timestamp."""
    return int(t // 900) * 900
