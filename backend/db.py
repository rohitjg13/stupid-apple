"""SQLite store with a single writer thread and batched commits.

The board runs on an SD card: WAL + synchronous=NORMAL + one transaction per
second (not per event) keeps writes fast and survives a power pull without
corrupting the file. Readers use their own short-lived connections so the
analytics pipeline can publish while the dashboard is querying.

`run_id` is set once per demo segment so each clip shows clean numbers
(tools/reset_run.py). Retention drops raw shelf/lane samples after 7 days and
rolls anything older than an hour into 1-minute means so the file stays bounded.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = Path(__file__).with_name("schema.sql").read_text()

RETENTION_S = 7 * 86400
ROLLUP_S = 3600


class DB:
    def __init__(self, path, batch_interval_s=1.0, batch_size=500,
                 schema=SCHEMA, _time=time.time):
        self.path = str(path)
        self.batch_interval_s = float(batch_interval_s)
        self.batch_size = int(batch_size)
        self.schema = schema
        self._time = _time

        self._q = queue.Queue()
        self._thread = None
        self._stop = threading.Event()
        self._conn = None
        self.dropped = 0

        self._init()

    # ---- setup ------------------------------------------------------------
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(self.schema)
            conn.commit()
        finally:
            conn.close()

    # ---- writer -----------------------------------------------------------
    def _start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._writer, name="backend-db",
                                        daemon=True)
        self._thread.start()

    def _writer(self):
        # The connection must be created in this thread: SQLite binds objects
        # to the thread that made them.
        self._conn = self._connect()
        self._error = None
        batch, batch_at = [], time.monotonic()
        while not self._stop.is_set():
            try:
                kind, payload = self._q.get(timeout=self.batch_interval_s)
            except queue.Empty:
                self._commit(batch)      # periodic flush even when idle
                batch, batch_at = [], time.monotonic()
                continue
            try:
                if kind == "sql":
                    batch.append(payload)
                    # Age as well as size. A running pipeline publishes a dozen
                    # events a second, so the queue is never empty for a whole
                    # interval and size alone held a batch open for a minute:
                    # the dashboard showed nothing until the run ended, and a
                    # power pull lost 500 rows instead of one second's worth.
                    if (len(batch) >= self.batch_size
                            or time.monotonic() - batch_at >= self.batch_interval_s):
                        self._commit(batch)
                        batch, batch_at = [], time.monotonic()
                elif kind == "call":
                    self._commit(batch)
                    batch, batch_at = [], time.monotonic()
                    payload(self._conn)     # atomic maintenance op in the writer
                    self._conn.commit()
                else:
                    self._commit(batch)
                    batch, batch_at = [], time.monotonic()
                    if isinstance(payload, threading.Event):
                        payload.set()
            except Exception as e:
                # One bad item must not kill the sink for the whole pipeline.
                self._conn.rollback()
                batch, batch_at = [], time.monotonic()
                if self._error is None:
                    self._error = e
                log.exception("db writer error")
                if kind == "flush" and isinstance(payload, threading.Event):
                    payload.set()
            finally:
                self._q.task_done()
        self._commit(batch)
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _commit(self, batch):
        if not batch:
            return True
        cur = self._conn.cursor()
        try:
            for sql, params in batch:
                cur.execute(sql, params)
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            # Retry one at a time: a single bad row must not take 500 good ones
            # with it. Whatever still fails is dropped, loudly, and re-raised
            # from drain() so nobody mistakes it for a clean run.
            self._conn.rollback()
            if self._error is None:
                self._error = e
            log.exception("db commit failed, retrying row by row",
                          extra={"statements": len(batch)})
            dropped = 0
            for sql, params in batch:
                try:
                    self._conn.execute(sql, params)
                    self._conn.commit()
                except sqlite3.Error:
                    self._conn.rollback()
                    dropped += 1
                    self.dropped += 1
            log.error("db rows dropped", extra={"dropped": dropped,
                                                "recovered": len(batch) - dropped})
            return False

    def enqueue(self, sql, params=()):
        self._start()
        self._q.put(("sql", (sql, params)))
        return self

    def insert(self, table, row):
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        return self.enqueue(f"INSERT INTO {table} ({cols}) VALUES ({marks})",
                            tuple(row.values()))

    def insert_many(self, table, rows):
        for row in rows:
            self.insert(table, row)
        return self

    def call(self, fn):
        """Run fn(conn) atomically on the writer thread (for Python-side rollups)."""
        self._start()
        self._q.put(("call", fn))
        return self

    def drain(self, timeout=10.0):
        """Block until everything enqueued so far is committed. Test/shutdown.

        If the writer hit an error, re-raise it here so callers notice instead
        of silently losing rows.
        """
        self._start()
        done = threading.Event()
        self._q.put(("flush", done))
        if not done.wait(timeout):
            raise TimeoutError("db writer did not flush within timeout")
        if getattr(self, "_error", None) is not None:
            err, self._error = self._error, None
            raise err
        return self

    def close(self, timeout=10.0):
        if self._thread is None and self._conn is None and getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.drain(timeout)
        finally:
            self._stop.set()
            self._q.put(("flush", None))   # wake a sleeping writer so it can exit
            if self._thread is not None:
                self._thread.join(timeout)
            self._thread = None
    # ---- reads ------------------------------------------------------------
    def query(self, sql, params=()):
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(sql, params)]
        finally:
            conn.close()

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---- maintenance (runs on the writer thread) ---------------------------
    def retention(self, now=None):
        """Drop raw shelf/lane samples older than 7 days; keep the DB bounded."""
        now = self._time() if now is None else now
        cutoff = now - RETENTION_S
        self.enqueue("DELETE FROM shelf_fill WHERE t < ?", (cutoff,))
        self.enqueue("DELETE FROM lane_occ WHERE t < ?", (cutoff,))
        return self

    def rollup(self, now=None):
        """Roll raw samples older than 1h into 1-minute means, keeping 7 days.

        Runs as one atomic op on the writer thread: raw rows are read, deleted,
        and their means inserted, so a crash mid-way cannot lose the data.
        """
        now = self._time() if now is None else now
        lo, hi = now - RETENTION_S, now - ROLLUP_S

        def _rollup(conn):
            # shelf_fill: mean fill per (run_id, minute, roi)
            rows = conn.execute(
                "SELECT run_id, CAST(t / 60 AS INTEGER) * 60 AS bucket, roi, fill "
                "FROM shelf_fill WHERE t >= ? AND t < ? ORDER BY bucket, roi",
                (lo, hi)).fetchall()
            groups = {}
            for run_id, bucket, roi, fill in rows:
                groups.setdefault((run_id, bucket, roi), []).append(fill)
            conn.execute("DELETE FROM shelf_fill WHERE t >= ? AND t < ?", (lo, hi))
            for (run_id, bucket, roi), fills in groups.items():
                conn.execute(
                    "INSERT INTO shelf_fill (run_id, t, roi, fill) VALUES (?, ?, ?, ?)",
                    (run_id, bucket, roi, sum(fills) / len(fills)))

            # lane_occ: byte-wise per-cell mean (cells is a 16-byte blob)
            rows = conn.execute(
                "SELECT run_id, CAST(t / 60 AS INTEGER) * 60 AS bucket, lane, cells "
                "FROM lane_occ WHERE t >= ? AND t < ? ORDER BY bucket, lane",
                (lo, hi)).fetchall()
            groups = {}
            for run_id, bucket, lane, cells in rows:
                groups.setdefault((run_id, bucket, lane), []).append(cells)
            conn.execute("DELETE FROM lane_occ WHERE t >= ? AND t < ?", (lo, hi))
            for (run_id, bucket, lane), cells in groups.items():
                n = len(cells)
                mean = bytes(sum(c[i] for c in cells) // n for i in range(16))
                conn.execute(
                    "INSERT INTO lane_occ (run_id, t, lane, cells) VALUES (?, ?, ?, ?)",
                    (run_id, bucket, lane, mean))

        self.call(_rollup)
        return self

    def integrity_check(self):
        row = self.query_one("PRAGMA integrity_check")
        return row["integrity_check"] if row else None
