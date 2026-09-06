"""Floor heatmap: where shoppers actually spend their time.

Two grids, deliberately:

* a cumulative count per 0.25 m cell, published every `heatmap_period_s` as a
  `heatmap` event -- this is the historical record the dashboard draws over the
  floorplan, and it must not decay or the totals stop meaning anything;
* a decayed copy for the live tile, so the screen shows where people are *now*
  rather than an ever-brightening smear of everywhere anyone has ever stood.

Cells come from geometry.zones.ZoneMap.heatmap_index, which is absolute rather
than raster-relative, so tiles mean the same thing across configs.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Heatmap:
    def __init__(self, params):
        self.p = params
        self.counts = {}            # (gx, gy) -> cumulative samples
        self.live = {}              # (gx, gy) -> decayed samples
        self._last_publish = None
        self._last_decay = None

    def add(self, cell):
        self.counts[cell] = self.counts.get(cell, 0) + 1
        self.live[cell] = self.live.get(cell, 0.0) + 1.0

    def _decay(self, t):
        """Exponential decay at `heatmap_decay_per_s`, applied in wall time."""
        if self._last_decay is None:
            self._last_decay = t
            return
        dt = t - self._last_decay
        if dt <= 0:
            return
        self._last_decay = t
        factor = self.p.heatmap_decay_per_s ** dt
        self.live = {c: v * factor for c, v in self.live.items() if v * factor > 0.01}

    def tick(self, t):
        """Call once a frame. Returns a payload when it is time to publish."""
        self._decay(t)
        if self._last_publish is None:
            self._last_publish = t
            return None
        if t - self._last_publish < self.p.heatmap_period_s:
            return None
        self._last_publish = t
        return self.payload(t)

    def payload(self, t):
        """The `heatmap` event payload from docs/SHARED.md §5."""
        return {"t_bucket": int(t // 60),
                "tiles": [[gx, gy, n] for (gx, gy), n in sorted(self.counts.items())]}

    def live_tiles(self):
        """The decayed grid for the dashboard's live view."""
        return [[gx, gy, round(v, 3)] for (gx, gy), v in sorted(self.live.items())]
