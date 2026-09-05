"""Time on a board with no RTC and possibly no network.

The PYNQ has no battery-backed clock. With no NTP it boots near 1970, which
would write events *before* everything already in the database. So:

- monotonic drives all elapsed time; the wall clock only sets the origin
- if the system clock looks impossible, continue from the last time we saw
  and flag `clock_unsynced`, which cloud sync waits on before uploading
- nothing here touches the network; main.py must never block on it
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

BUILD_EPOCH = 1757030400.0      # 2025-09-05; anything earlier is not a real clock


class Clock:
    def __init__(self, state_path="/var/lib/retail/clock.json", _system_time=time.time):
        self.state_path = Path(state_path)
        self._sys = _system_time
        self._mono0 = time.monotonic()

        last_seen = 0.0
        try:
            last_seen = float(json.loads(self.state_path.read_text())["last_seen"])
        except Exception:
            pass

        sys_now = self._sys()
        if sys_now >= max(BUILD_EPOCH, last_seen):
            self.synced, self._epoch0 = True, sys_now
        else:
            # ponytail: +1 s past the last thing we recorded keeps event order
            # intact without inventing a plausible-looking wall time.
            self.synced = False
            self._epoch0 = max(last_seen + 1.0, BUILD_EPOCH)
            log.warning("clock unsynced, continuing from last known time",
                        extra={"system_time": sys_now, "resumed_at": self._epoch0})

    def now(self) -> float:
        return self._epoch0 + (time.monotonic() - self._mono0)

    def adopt(self, real_epoch: float):
        """NTP (or the dashboard) handed us a real time; re-base on it."""
        if real_epoch < BUILD_EPOCH:
            raise ValueError(f"{real_epoch} is not a plausible epoch time")
        self._epoch0, self._mono0 = real_epoch, time.monotonic()
        self.synced = True
        log.info("clock synced", extra={"epoch": real_epoch})

    def persist(self):
        """Record where we got to, so the next boot can continue from here."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"last_seen": self.now(), "synced": self.synced}))
            tmp.replace(self.state_path)
        except OSError as e:
            log.warning("cannot persist clock state", extra={"error": str(e)})

    def health(self) -> dict:
        return {"t": self.now(), "clock_unsynced": not self.synced,
                "uptime_s": round(time.monotonic() - self._mono0, 1)}
