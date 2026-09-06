"""In-process event bus. Analytics must never block on a slow subscriber."""
from __future__ import annotations

import logging
import queue
import threading
import time

log = logging.getLogger(__name__)


class Bus:
    """publish() from any thread; each subscriber drains its own bounded queue.

    ponytail: one worker thread per subscriber. Handful of subscribers, so a
    thread each beats a scheduler. Move to a pool if that stops being true.
    """

    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._subs = []                 # (event_type, Queue)
        self._lock = threading.Lock()
        self.dropped = 0

    def subscribe(self, event_type: str, fn):
        q = queue.Queue(maxsize=self._maxsize)
        t = threading.Thread(target=self._run, args=(q, fn), daemon=True)
        with self._lock:
            self._subs.append((event_type, q))
        t.start()

    def publish(self, event):
        with self._lock:
            subs = list(self._subs)
        for event_type, q in subs:
            if event_type in ("*", event.event_type):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    self.dropped += 1   # drop newest; stale analytics beat a stalled pipeline

    def drain(self, timeout: float = 5.0):
        """Test/shutdown helper: wait until every subscriber has caught up.

        Subscribers publish too -- backend/alerts.py turns a stockout into an
        alert -- so one pass of joins is not enough: the alert lands in a queue
        that was joined a moment ago. Keep going until a whole pass finds
        nothing queued anywhere.

        ponytail: `timeout` bounds the number of passes, not one pass. A
        subscriber that publishes back into its own event type loops until the
        deadline; one that simply blocks forever still blocks here, exactly as
        it did before. Neither has ever happened; make it interruptible if one
        does.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                subs = list(self._subs)
            for _, q in subs:
                q.join()
            with self._lock:
                if all(q.empty() for _, q in self._subs):
                    return
            if time.monotonic() >= deadline:
                log.warning("bus drain timed out with events still queued")
                return

    def _run(self, q, fn):
        while True:
            event = q.get()
            try:
                fn(event)
            except Exception:
                log.exception("subscriber failed", extra={"event_type": event.event_type})
            finally:
                q.task_done()
