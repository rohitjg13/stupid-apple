"""In-process event bus. Analytics must never block on a slow subscriber."""
from __future__ import annotations

import logging
import queue
import threading

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
        """Test/shutdown helper: wait until every subscriber has caught up."""
        with self._lock:
            subs = list(self._subs)
        for _, q in subs:
            q.join()

    def _run(self, q, fn):
        while True:
            event = q.get()
            try:
                fn(event)
            except Exception:
                log.exception("subscriber failed", extra={"event_type": event.event_type})
            finally:
                q.task_done()
