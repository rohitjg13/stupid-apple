"""Zone visits and dwell time, with hysteresis on both edges.

A shopper standing on a zone boundary flickers between two zones at frame rate.
Without debouncing you get dozens of half-second visits instead of one real one,
and dwell -- which is the number the layout analysis rests on -- becomes noise.

So a zone change is only believed after `visit_debounce_s` of *continuous*
presence in the new zone, and because every change goes through that same gate,
leaving is debounced exactly as much as arriving. The exit timestamp is when the
shopper was first seen elsewhere, not when we finally believed it, so the
debounce delays the decision without inflating the dwell.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# `None` is a legitimate zone value -- it means "on the floor but inside no zone"
# -- so it cannot double as "no pending candidate". Sharing the sentinel makes a
# shopper walking out of the zoned area look like a zone change to nowhere with a
# stale timestamp, which silently discards the visit they just finished.
_NO_CANDIDATE = object()


@dataclass(frozen=True)
class Visit:
    track_id: int
    zone_id: str
    t_enter: float
    t_exit: float

    @property
    def dwell_s(self) -> float:
        return self.t_exit - self.t_enter


class _State:
    __slots__ = ("zone", "since", "cand", "cand_since")

    def __init__(self):
        self.zone = None            # the zone we believe they are in
        self.since = 0.0            # when that visit started
        self.cand = _NO_CANDIDATE   # a zone change waiting to be believed
        self.cand_since = 0.0


class VisitTracker:
    """Feed it `(track_id, zone_id)` per frame; it yields completed visits."""

    def __init__(self, params):
        self.p = params
        self._st = {}

    def _finish(self, track_id, st, t_exit):
        """Close the open visit if it was long enough to mean anything."""
        if st.zone is None:
            return None
        v = Visit(track_id, st.zone, st.since, t_exit)
        st.zone, st.since = None, 0.0
        if v.dwell_s < self.p.min_visit_s:
            return None             # walked through, did not visit
        log.info("visit", extra={"track_id": v.track_id, "zone": v.zone_id,
                                 "dwell_s": round(v.dwell_s, 2)})
        return v

    def update(self, observations, t):
        """`observations` is an iterable of (track_id, zone_id | None)."""
        out = []
        for track_id, zone in observations:
            st = self._st.setdefault(track_id, _State())
            if zone == st.zone:
                st.cand = _NO_CANDIDATE             # still here; forget any wobble
                continue
            if st.cand != zone:
                st.cand, st.cand_since = zone, t    # a new candidate, start the clock
                continue
            if t - st.cand_since < self.p.visit_debounce_s:
                continue                            # not yet convincing
            # Believe it. They left at the moment they were first seen elsewhere.
            done = self._finish(track_id, st, st.cand_since)
            if done is not None:
                out.append(done)
            st.zone, st.since = zone, st.cand_since
            st.cand = _NO_CANDIDATE
        return out

    def close_track(self, track_id, t):
        """A track was deleted: flush whatever visit it had open."""
        st = self._st.pop(track_id, None)
        if st is None:
            return None
        return self._finish(track_id, st, t)

    def close(self, t):
        """Shutdown: flush every open visit so nothing is silently lost."""
        out = []
        for track_id in list(self._st):
            v = self.close_track(track_id, t)
            if v is not None:
                out.append(v)
        return out
