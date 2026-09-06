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
    """One reader per stream. `loop` replays the clip; frame_id keeps counting.

    `paths: [a.mp4, b.mp4]` in store.yaml plays several clips back to back as
    one continuous stream, which is what uploading a few videos means.
    """

    def __init__(self, cfg, stream="overhead", realtime=True, decode=True):
        spec = cfg.streams[stream]
        paths = spec.get("paths") or [spec["path"]]
        self.paths = [Path(p) for p in paths]
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(f"{stream} clip not found: {p}")
        self.path = self.paths[0]
        self.stream = stream
        self.fps = float(spec.get("fps", 15))
        self.loop = bool(spec.get("loop", False))
        self.realtime = realtime
        self.decode = decode
        self._clip = 0
        self._cap = self._open(self.path)
        self._closed = False

    def _open(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot decode {path}; try tools/transcode.sh")
        return cap

    @property
    def total_frames(self):
        """Frames across every clip; 0 when the container does not say."""
        n = 0
        for p in self.paths:
            cap = cv2.VideoCapture(str(p))
            n += max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        return n

    def frames(self):
        """`t_ns` is the clip's own clock, not the wall clock.

        A file has a timeline of its own, and events have to sit on it: process
        an hour of footage in three minutes and the dwells must still be the
        dwells that happened, not a third of them. sources/sim.py already
        stamps virtual time this way; realtime playback makes the two agree.
        """
        dt = 1.0 / max(self.fps, 1e-6)
        frame_id = 0
        t0_ns = time.monotonic_ns()
        next_t = time.monotonic()
        while not self._closed:
            ok, img = self._cap.read()
            if not ok:
                if self._clip + 1 < len(self.paths):        # next clip in the playlist
                    self._clip += 1
                    self._cap.release()
                    self._cap = self._open(self.paths[self._clip])
                    continue
                if not self.loop:
                    log.info("clip exhausted", extra={"stream": self.stream, "frames": frame_id})
                    return
                self._clip = 0
                self._cap.release()
                self._cap = self._open(self.paths[0])
                ok, img = self._cap.read()
                if not ok:
                    log.error("clip will not rewind", extra={"path": str(self.paths[0])})
                    return

            r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
            r["frame_id"] = frame_id
            yield Frame(self.stream, frame_id, t0_ns + int(frame_id * dt * 1e9), r,
                        image=letterbox(img) if self.decode else None)
            frame_id += 1

            if self.realtime:
                next_t += dt
                time.sleep(max(0.0, next_t - time.monotonic()))

    def close(self):
        self._closed = True
        self._cap.release()
