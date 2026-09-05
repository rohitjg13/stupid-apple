"""Synthetic store: walkers on the floor plan, rendered to FrameResults.

Day-1 development source for every module, and the demo fallback if the board
dies. Deterministic given --seed. No OpenCV, no board, no footage.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from core.events import Event
from pl.contract import FLAG_BG_WARM, FRAME_RESULT_DT, MAX_BLOBS, Frame
from sources.base import Source

log = logging.getLogger(__name__)

WARMUP_FRAMES = 100          # matches mog2_lite's warm-up in hls/
PERSON_W_M, PERSON_H_M = 0.5, 1.7
SHELF_STOCKED = (170, 210)   # Sobel edge density of a full facing
SHELF_EMPTY = (25, 50)


class _Walker:
    """Enter -> browse a couple of zones -> queue at checkout -> leave."""

    def __init__(self, rng, zones, tid):
        self.rng, self.tid = rng, tid
        # Spawn *outside* the door, not on it. Walkers step once before their
        # first blob is rendered, so a spawn at Y=0.2 (v=464) is already past the
        # door tripwire at v=460 by the time it is first visible, and the entry
        # can never be observed. Starting outside makes the crossing real.
        self.pos = np.array([rng.uniform(0.5, 2.5), -0.4])
        self.targets = [self._point(rng, z) for z in rng.choice(zones, size=2, replace=False)]
        self.targets.append(np.array([rng.uniform(3.5, 7.5), 1.0]))   # checkout
        # Then back to the door before leaving. Without this, walkers exit from
        # wherever the checkout put them (X 3.5-7.5, so u 280-600) and walk out
        # through the wall, missing the door tripwire's u span of 40-240 entirely
        # -- entries counted, exits not, and occupancy drifts away from the count.
        # Y=1.2 m, not right on the line: walkers dwell at their last target, and
        # a target on top of the tripwire makes them straddle it for 3-25 s and
        # fire a crossing every debounce period.
        self.targets.append(np.array([rng.uniform(0.5, 2.5), 1.2]))     # the door
        self.dwell = 0.0
        self.done = False
        self.size = rng.uniform(0.85, 1.15)     # people are not all one size

    @staticmethod
    def _point(rng, zone):
        poly = np.array(zone["polygon"])
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        return rng.uniform(lo, hi)

    def step(self, dt):
        if self.dwell > 0:
            self.dwell -= dt
            return
        if not self.targets:
            self.pos[1] -= 1.2 * dt                     # walk out through the door
            self.done = self.pos[1] < -0.5
            return
        delta = self.targets[0] - self.pos
        dist = np.linalg.norm(delta)
        if dist < 0.2:
            self.targets.pop(0)
            # No loitering on the last target, which is the door: a crowd standing
            # on the exit merges into shared blobs, churns track ids, and every new
            # id crossing the line counts as another exit.
            self.dwell = self.rng.uniform(3.0, 25.0) if self.targets else 0.0
            return
        self.pos = self.pos + delta / dist * min(1.2 * dt, dist)


class SimSource(Source):
    def __init__(self, cfg, stream="overhead", seed=0, arrivals_per_min=12.0, fps=None):
        if stream not in ("overhead", "shelf"):
            raise ValueError(f"unknown stream {stream!r}")
        self.cfg, self.stream, self.seed = cfg, stream, seed
        self.fps = float(fps or cfg.streams[stream].get("fps", 15))
        self.arrivals_per_min = arrivals_per_min
        self.rng = np.random.default_rng(seed)
        self._closed = False

        self.H_inv = np.linalg.inv(cfg.homography["overhead"])   # floor -> image
        self.walkers, self._next_tid = [], 0
        # ponytail: one stock-out at a time is all the shelf lead needs to develop
        # against; concurrent facings can wait until someone asks.
        self.stockout_roi, self.stockout_until = None, 0.0

    # ---- floor -> image ---------------------------------------------------
    def _to_image(self, xy):
        u, v, w = self.H_inv @ np.array([xy[0], xy[1], 1.0])
        return u / w, v / w

    def _blob(self, walker):
        """A person as an axis-aligned box in full-res 640x480 px."""
        cu, cv = self._to_image(walker.pos)
        # ponytail: the sim floor is orthographic, so there is no perspective
        # foreshortening here. Per-walker jitter keeps the tracker honest; real
        # size-vs-depth arrives with footage in W2.
        h = max(8.0, PERSON_H_M * 80 * walker.size)
        w = max(6.0, PERSON_W_M * 80 * walker.size)
        x, y = cu - w / 2, cv - h
        x0, y0 = int(np.clip(x, 0, 639)), int(np.clip(y, 0, 479))
        x1, y1 = int(np.clip(x + w, 0, 640)), int(np.clip(y + h, 0, 480))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0))

    # ---- shelf ------------------------------------------------------------
    def _roi_fill(self, t):
        n = len(self.cfg.rois)
        fill = self.rng.integers(*SHELF_STOCKED, size=n).astype(np.uint8)
        if self.stockout_roi is None and self.rng.random() < 0.004:
            self.stockout_roi = int(self.rng.integers(n))
            self.stockout_until = t + self.rng.uniform(30.0, 200.0)
            log.info("sim stockout", extra={"roi": self.cfg.rois[self.stockout_roi]["id"]})
        if self.stockout_roi is not None:
            if t >= self.stockout_until:
                self.stockout_roi = None
            else:
                fill[self.stockout_roi] = self.rng.integers(*SHELF_EMPTY)
        return fill

    def _lane_occupancy(self, queued):
        """Cells fill from the counter backwards, so cell 0 is busiest."""
        occ = np.zeros(16, dtype=np.uint8)
        for i, lane in enumerate(self.cfg.lanes):
            filled = queued > lane["cell"]
            occ[i] = self.rng.integers(60, 95) if filled else self.rng.integers(0, 12)
        return occ

    # ---- main loop --------------------------------------------------------
    def frames(self):
        dt = 1.0 / self.fps
        t = 0.0
        t0_ns = time.monotonic_ns()
        for frame_id in range(1 << 32):
            if self._closed:
                return
            r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
            r["frame_id"] = frame_id
            if frame_id >= WARMUP_FRAMES:
                r["flags"] |= FLAG_BG_WARM

            if self.stream == "overhead":
                if self.rng.random() < self.arrivals_per_min / 60.0 * dt:
                    self.walkers.append(_Walker(self.rng, self.cfg.zones, self._next_tid))
                    self._next_tid += 1
                for wk in self.walkers:
                    wk.step(dt)
                self.walkers = [wk for wk in self.walkers if not wk.done]

                blobs = [b for b in (self._blob(wk) for wk in self.walkers) if b]
                blobs.sort(key=lambda b: b[4], reverse=True)
                r["num_blobs"] = min(len(blobs), MAX_BLOBS)
                if len(blobs) > MAX_BLOBS:
                    r["flags"] |= 1 << 1
                for i, b in enumerate(blobs[:MAX_BLOBS]):
                    r["blobs"][i] = b
                queued = sum(1 for wk in self.walkers if wk.targets and len(wk.targets) == 1)
                r["lane_occupancy"] = self._lane_occupancy(queued)
            else:
                r["roi_fill"][: len(self.cfg.rois)] = self._roi_fill(t)

            yield Frame(self.stream, frame_id, t0_ns + int(t * 1e9), r, image=None)
            t += dt

    def close(self):
        self._closed = True

    # ---- backfill ---------------------------------------------------------
    def history(self, days=7, step_s=60.0):
        """Plausible past for the backend lead's dashboard. Events, not frames."""
        rng = np.random.default_rng(self.seed + 1)
        now = time.time()
        store = self.cfg.store_id
        for ts in np.arange(now - days * 86400, now, step_s):
            hour = time.localtime(ts).tm_hour
            busy = 0.2 + 0.8 * np.exp(-((hour - 18) ** 2) / 12.0)    # evening peak
            yield Event(float(ts), store, "overhead", None, "occupancy",
                        {"count": int(rng.poisson(12 * busy))})
            if rng.random() < busy:
                yield Event(float(ts), store, "overhead", "door", "tripwire",
                            {"dir": "in" if rng.random() < 0.52 else "out", "tripwire_id": "door"})
            if rng.random() < 0.3 * busy:
                item = self.cfg.planogram[int(rng.integers(len(self.cfg.planogram)))]
                n = int(rng.integers(1, 9))
                yield Event(float(ts), store, "-", None, "pos_txn",
                            {"lane": int(rng.integers(1, 3)), "items": n,
                             "amount": round(n * item["unit_price"] * rng.uniform(0.8, 1.4), 2),
                             "zone_hint": "checkout"})


def main(argv=None):
    """Backfill for the backend lead:

        python -m sources.sim --history-days 7 > history.jsonl
    """
    import argparse
    import os
    import sys

    from core.config import load_clipset

    ap = argparse.ArgumentParser(description="synthetic store")
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--history-days", type=float, default=7.0)
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    src = SimSource(load_clipset(a.config), stream="overhead", seed=a.seed)
    try:
        for event in src.history(days=a.history_days, step_s=a.step_s):
            print(event.to_json())
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())   # `| head` is fine
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
