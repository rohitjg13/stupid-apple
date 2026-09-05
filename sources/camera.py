"""USB camera source. Survives the shelf camera being bumped on stage."""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from pl.contract import FRAME_RESULT_DT, Frame
from pl.reference import letterbox
from sources.base import Source

log = logging.getLogger(__name__)


class CameraSource(Source):
    def __init__(self, cfg, stream="shelf", realtime=True, reconnect_delay=1.0,
                 max_reconnects=30):
        spec = cfg.streams[stream]
        self.stream = stream
        self.device = spec.get("device", 0)
        self.fps = float(spec.get("fps", 5))
        self.fourcc = spec.get("fourcc", "YUYV")
        self.realtime = realtime
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._cap = None
        self._closed = False

    def _open(self):
        if self._cap is not None:
            self._cap.release()
        cap = cv2.VideoCapture(self.device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # newest frame, not a backlog
        self._cap = cap
        return cap

    def frames(self):
        dt = 1.0 / max(self.fps, 1e-6)
        frame_id, failures = 0, 0
        next_t = time.monotonic()
        cap = self._open()
        while not self._closed:
            ok, img = (cap.read() if cap is not None and cap.isOpened() else (False, None))
            if not ok:
                failures += 1
                if failures > self.max_reconnects:
                    raise RuntimeError(
                        f"camera {self.device!r} did not come back after "
                        f"{failures} attempts")
                log.warning("camera read failed, reopening",
                            extra={"device": str(self.device), "attempt": failures})
                time.sleep(self.reconnect_delay)
                cap = self._open()
                continue
            failures = 0

            r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
            r["frame_id"] = frame_id
            yield Frame(self.stream, frame_id, time.monotonic_ns(), r, image=letterbox(img))
            frame_id += 1

            if self.realtime:
                next_t += dt
                time.sleep(max(0.0, next_t - time.monotonic()))

    def close(self):
        self._closed = True
        if self._cap is not None:
            self._cap.release()
