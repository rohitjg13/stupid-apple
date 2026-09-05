"""blobs -> tracks -> floor -> zones -> events. The only entry point main.py uses.

Order per frame: warm-up gate, flood guard, prefilter, track, foot point, floor
metres, zone, then the four shopper events from docs/SHARED.md §5.

Three behaviours here are load-bearing and non-obvious:

* **Occupancy publishes at 1 Hz unconditionally**, reporting 0 while the
  background model is cold. It is a heartbeat, not a detection: a dashboard tile
  that simply stops updating for the first seven seconds looks broken, and
  tests/test_main.py counts these events.
* **Nothing is counted or confirmed until FLAG_BG_WARM is set.** sources/sim.py
  emits blobs during warm-up while pl/reference.py emits none; gating on the flag
  is what makes sim, reference and PL produce the same events, which is one of
  the acceptance criteria.
* **A flooded frame pauses counting** rather than emitting garbage. When someone
  knocks the camera or the lights change, most of the frame goes foreground and
  every "person" in it is fictional.
"""
from __future__ import annotations

import logging

from core.events import Event
from geometry.homography import image_to_floor
from geometry.zones import ZoneMap
from pl.contract import FLAG_BG_WARM
from shopper.blob_prefilter import prefilter
from shopper.heatmap import Heatmap
from shopper.params import TrackerParams
from shopper.tracker import Tracker
from shopper.tripwire import TripwireCounter
from shopper.visits import VisitTracker

log = logging.getLogger(__name__)

FRAME_AREA = 640 * 480


class ShopperPipeline:
    """One overhead stream's worth of shopper analytics."""

    def __init__(self, cfg, bus, params=None, stream="overhead"):
        self.cfg, self.bus, self.stream = cfg, bus, stream
        self.p = params or TrackerParams.load(getattr(cfg, "path", None))
        self.tracker = Tracker(self.p)
        self.visits = VisitTracker(self.p)
        self.heatmap = Heatmap(self.p)
        self.tripwires = TripwireCounter(cfg.tripwires, self.p, stream=stream)
        self.zones = ZoneMap(cfg.zones) if cfg.zones else None
        self.H = cfg.homography.get(stream)
        self._last_occ = None
        self._flood_until = -1e9
        self.state = {"occupancy": 0, "counts": self.tripwires.counts}

    # --- helpers ----------------------------------------------------------
    def _publish(self, t, event_type, payload, zone_id=None):
        self.bus.publish(Event(t, self.cfg.store_id, self.stream, zone_id,
                               event_type, payload))

    def _floor(self, foot):
        if self.H is None:
            return None
        X, Y = image_to_floor(self.H, foot)
        return float(X), float(Y)

    def _is_flooded(self, blobs, t):
        total = sum(int(b["w"]) * int(b["h"]) for b in blobs)
        if total > self.p.flood_area_frac * FRAME_AREA:
            if t >= self._flood_until:
                log.warning("frame flooded, pausing counting",
                            extra={"blob_area_frac": round(total / FRAME_AREA, 3)})
            self._flood_until = t + self.p.flood_pause_s
        return t < self._flood_until

    def _occupancy(self, t, count):
        """1 Hz heartbeat. Shoppers in the entrance are arriving, not shopping."""
        if self._last_occ is not None and t - self._last_occ < self.p.occupancy_period_s:
            return
        self._last_occ = t
        self.state["occupancy"] = count
        self._publish(t, "occupancy", {"count": int(count)})

    # --- per frame --------------------------------------------------------
    def on_frame(self, frame, t):
        """`t` is the epoch time main.py already derived from t_ns. Never recomputed."""
        if frame.stream != self.stream:
            return
        warm = bool(int(frame.result["flags"]) & FLAG_BG_WARM)
        if not warm or self._is_flooded(frame.blobs, t):
            self._occupancy(t, 0)
            return

        blobs = prefilter(frame.blobs, self.p)
        active = self.tracker.update(blobs, t)

        observations, counted = [], 0
        for tr in active:
            floor = self._floor(tr.foot)
            zone = self.zones.zone_of(*floor) if (floor and self.zones) else None
            observations.append((tr.id, zone))
            if zone != self.p.entrance_zone:
                counted += 1
            if floor and self.zones:
                self.heatmap.add(self.zones.heatmap_index(*floor))

        for visit in self.visits.update(observations, t):
            self._emit_visit(visit)

        for wire_id, direction, _ in self.tripwires.update(active, t):
            self._publish(t, "tripwire", {"dir": direction, "tripwire_id": wire_id},
                          zone_id=wire_id)

        for tr in self.tracker.just_deleted:
            visit = self.visits.close_track(tr.id, t)
            if visit is not None and not tr.furniture:   # furniture does not visit
                self._emit_visit(visit)
            self.tripwires.forget(tr.id)

        payload = self.heatmap.tick(t)
        if payload is not None:
            self._publish(t, "heatmap", payload)

        self._occupancy(t, counted)

    def _emit_visit(self, visit):
        self._publish(visit.t_exit, "visit",
                      {"track_id": visit.track_id, "t_enter": visit.t_enter,
                       "t_exit": visit.t_exit}, zone_id=visit.zone_id)

    def close(self, t=None):
        """Shutdown: flush open visits so a demo run does not lose its last dwells."""
        t = self.tracker.tracks[-1].last_t if (t is None and self.tracker.tracks) else t
        if t is None:
            return
        for tr in self.tracker.close():
            self.tripwires.forget(tr.id)
        for visit in self.visits.close(t):
            self._emit_visit(visit)
