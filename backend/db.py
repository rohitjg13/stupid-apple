"""SQLite store with a single writer thread and batched commits.

The board runs on an SD card: WAL + synchronous=NORMAL + one transaction per
second (not per event) keeps writes fast and survives a power pull without
corrupting the file. Readers get their own connection per thread, cached and
kept open, so the analytics pipeline can publish while the dashboard queries and
neither pays to reopen the file.

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
HEATMAP_BUCKET_S = 900   # roll 10 s tiles up to the kpi_15m grain


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
        self._error = None
        self._local = threading.local()
        self._readers = []
        self._readers_lock = threading.Lock()
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
        batch = []
        while not self._stop.is_set():
            try:
                kind, payload = self._q.get(timeout=self.batch_interval_s)
            except queue.Empty:
                self._commit(batch)      # periodic flush even when idle
                batch = []
                continue
            try:
                if kind == "sql":
                    batch.append(payload)
                    if len(batch) >= self.batch_size:
                        self._commit(batch)
                        batch = []
                elif kind == "call":
                    self._commit(batch)
                    batch = []
                    payload(self._conn)     # atomic maintenance op in the writer
                    self._conn.commit()
                else:
                    self._commit(batch)
                    batch = []
                    if isinstance(payload, threading.Event):
                        payload.set()
            except Exception as e:
                # One bad item must not kill the sink for the whole pipeline.
                self._conn.rollback()
                batch = []
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
            self._conn.rollback()
            if self._error is None:
                self._error = e
            return self._commit_one_by_one(batch, e)

    def _commit_one_by_one(self, batch, cause):
        """Re-apply a failed batch statement by statement.

        Acceptance criterion is *zero dropped events*: one malformed row must
        not take the other 499 in its transaction down with it. Only the
        offending statements are counted in `dropped`.

        Logging is one summary line per batch, not one per statement: a
        systematically bad insert would otherwise fill the board's log faster
        than the events it is complaining about.
        """
        cur = self._conn.cursor()
        failed = []
        for sql, params in batch:
            try:
                cur.execute(sql, params)
            except sqlite3.Error:
                failed.append(sql)
        try:
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            self.dropped += len(batch)
            log.error("db retry commit failed", extra={
                "statements": len(batch), "error": str(e)})
            return False
        self.dropped += len(failed)
        log.warning("db batch partially applied", extra={
            "statements": len(batch), "failed": len(failed),
            "error": str(cause), "example_sql": failed[0] if failed else None})
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
            self._close_readers()

    def _close_readers(self):
        with self._readers_lock:
            readers, self._readers = self._readers, []
        for conn in readers:
            try:
                conn.close()
            except sqlite3.Error:       # already closed, or another thread's
                pass
        self._local = threading.local()
    # ---- reads ------------------------------------------------------------
    def _reader(self):
        """One read connection per thread, kept open.

        `live_summary` alone is six queries; opening a connection and replaying
        the PRAGMAs for each one costs more than the queries do. SQLite objects
        are thread-bound, so the cache is thread-local, and WAL means a reader
        never blocks the writer.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            # Per-connection, and only meaningful for writes, but keeping every
            # connection on the same setting means one PRAGMA answers for the
            # whole store rather than depending on which one you asked.
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._readers_lock:
                self._readers.append(conn)
        return conn

    def query(self, sql, params=()):
        return [dict(r) for r in self._reader().execute(sql, params)]

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

            # heatmap: 10-second tiles summed into 15-minute ones. Counts add,
            # they do not average, so the totals the chart shows are unchanged
            # by the rollup — only the resolution in time is. A demo day drops
            # from ~86k rows to ~96 buckets per occupied cell, which is what
            # keeps the whole-day query inside the board budget.
            rows = conn.execute(
                "SELECT run_id, CAST(t_bucket / ? AS INTEGER) * ? AS bucket, "
                "gx, gy, SUM(count) AS n FROM heatmap "
                "WHERE t_bucket >= ? AND t_bucket < ? "
                "GROUP BY run_id, bucket, gx, gy",
                (HEATMAP_BUCKET_S, HEATMAP_BUCKET_S, lo, hi)).fetchall()
            conn.execute("DELETE FROM heatmap WHERE t_bucket >= ? AND t_bucket < ?",
                         (lo, hi))
            for run_id, bucket, gx, gy, n in rows:
                conn.execute(
                    "INSERT INTO heatmap (run_id, t_bucket, gx, gy, count) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(run_id, t_bucket, gx, gy) "
                    "DO UPDATE SET count = count + excluded.count",
                    (run_id, bucket, gx, gy, n))

        self.call(_rollup)
        return self

    def integrity_check(self):
        row = self.query_one("PRAGMA integrity_check")
        return row["integrity_check"] if row else None
