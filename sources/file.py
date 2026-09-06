"""Footage source: MJPEG/MP4 through cv2.VideoCapture, letterboxed to 640x480."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from pl.contract import FRAME_RESULT_DT, Frame
from pl.reference import letterbox
from sources.base import Source

log = logging.getLogger(__name__)


class FileSource(Source):
    """One reader per stream. `loop` replays the clip; frame_id keeps counting."""

    def __init__(self, cfg, stream="overhead", realtime=True, decode=True):
        spec = cfg.streams[stream]
        self.path = Path(spec["path"])
        if not self.path.exists():
            raise FileNotFoundError(f"{stream} clip not found: {self.path}")
        self.stream = stream
        self.fps = float(spec.get("fps", 15))
        self.loop = bool(spec.get("loop", False))
        self.realtime = realtime
        self.decode = decode
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot decode {self.path}; try tools/transcode.sh")
        self._closed = False

    def frames(self):
        dt = 1.0 / max(self.fps, 1e-6)
        frame_id = 0
        next_t = time.monotonic()
        while not self._closed:
            ok, img = self._cap.read()
            if not ok:
                if not self.loop:
                    log.info("clip exhausted", extra={"stream": self.stream, "frames": frame_id})
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
                if not ok:
                    log.error("clip will not rewind", extra={"path": str(self.path)})
                    return

            r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
            r["frame_id"] = frame_id
            yield Frame(self.stream, frame_id, time.monotonic_ns(), r,
                        image=letterbox(img) if self.decode else None)
            frame_id += 1

            if self.realtime:
                next_t += dt
                time.sleep(max(0.0, next_t - time.monotonic()))

    def close(self):
        self._closed = True
        self._cap.release()
