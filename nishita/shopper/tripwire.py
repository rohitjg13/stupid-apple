"""Entry/exit counting: a track's foot point crossing a line segment.

Tripwires live in `tripwires.json` in **640x480 image coordinates** -- unlike
rois.json and lanes.json, which core/config.py validates at 320x240 PL scale.
Direction comes from the sign of the cross product of the line against the
movement, compared with the configured `in_dir`.

Two details that are easy to get wrong and both cost you every entry:

* A track's foot points are replayed from its *birth*, not from the frame it was
  confirmed. On config/sim a walker spawns at v=464 with the door line at v=460
  and crosses on its second frame, three frames before a 3-hit confirmation.
* Only movement between two *consecutively observed* foot points counts. A track
  born on the far side of the line walked in off-camera and was never seen to
  cross, so no segment of its own can be credited with a crossing.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# in_dir -> the image-space direction that counts as entering the store.
# Image v grows downwards, so "up" means decreasing v.
_IN_VECTOR = {"up": (0.0, -1.0), "down": (0.0, 1.0),
              "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def segments_intersect(p1, p2, q1, q2):
    """True if segment p1->p2 properly crosses q1->q2."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def signed_distance(line, pt):
    """Perpendicular distance from the line, signed by which side `pt` is on."""
    (x1, y1), (x2, y2) = line
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return 0.0
    return ((dx * (pt[1] - y1) - dy * (pt[0] - x1)) / length)


def side_of(line, pt):
    """Which side of the line a point is on: +1, -1, or 0 exactly on it."""
    (x1, y1), (x2, y2) = line
    d = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
    return (d > 0) - (d < 0)


class TripwireCounter:
    """One counter for every tripwire on a stream. Feed it tracks each frame."""

    def __init__(self, tripwires, params, stream="overhead"):
        self.p = params
        self.wires = [w for w in tripwires if w.get("stream", stream) == stream]
        for w in self.wires:
            if w["in_dir"] not in _IN_VECTOR:
                raise ValueError(f"tripwire {w['id']}: bad in_dir {w['in_dir']!r}")
        self._last = {}         # (track_id, wire_id) -> last foot point outside the band
        self._fired = {}        # (track_id, wire_id) -> time of last crossing
        self._side = {}         # (track_id, wire_id) -> last committed side
        self._replayed = set()  # track ids whose birth buffer has been consumed
        self.counts = {"in": 0, "out": 0}

    def _direction(self, wire, a, b):
        """'in' or 'out' for a movement a->b across `wire`."""
        (x1, y1), (x2, y2) = self._line(wire)
        # Normal pointing to the +1 side of the line.
        nx, ny = -(y2 - y1), (x2 - x1)
        want = _IN_VECTOR[wire["in_dir"]]
        # Does the +1-side normal agree with the configured "in" direction?
        normal_is_in = (nx * want[0] + ny * want[1]) > 0
        moved_to_positive = side_of(self._line(wire), b) > 0
        return "in" if moved_to_positive == normal_is_in else "out"

    @staticmethod
    def _line(wire):
        return (tuple(wire["p1"]), tuple(wire["p2"]))

    def _points(self, track):
        """Foot points to test this frame: the whole birth buffer, once."""
        if track.id in self._replayed:
            return [(track.last_t, *track.foot)]
        self._replayed.add(track.id)
        return list(track.trail)

    def update(self, tracks, t):
        """Returns [(wire_id, 'in'|'out', track_id)] for crossings this frame."""
        events = []
        for tr in tracks:
            pts = self._points(tr)
            for wire in self.wires:
                key = (tr.id, wire["id"])
                line = self._line(wire)
                for (_, ux, uy) in pts:
                    cur = (ux, uy)
                    # Hysteresis. A shopper standing on the line -- and on
                    # config/sim the door sits *inside* the walkable entrance zone,
                    # so they legitimately do -- wobbles across it at frame rate
                    # and every wobble is an entry and an exit. Only movement that
                    # clears the band on both sides is a real crossing.
                    dist = signed_distance(line, cur)
                    new_side = 1 if dist > 0 else -1
                    committed = self._side.get(key)
                    if committed is None:
                        # Establishing a side is free. Requiring the band here too
                        # would silently drop anyone who first appears right at the
                        # wire -- which is most people, since that is where the
                        # door is and where they walk in from the frame edge.
                        self._side[key] = new_side
                        self._last[key] = cur
                        continue
                    if new_side == committed:
                        self._last[key] = cur    # anchor stays on the current side
                        continue
                    if abs(dist) < self.p.tripwire_deadband_px:
                        continue    # over the line but still in the band: uncommitted
                    prev = self._last.get(key)
                    self._side[key] = new_side
                    self._last[key] = cur
                    if prev is None or not segments_intersect(prev, cur, *line):
                        continue        # crossed the line, but past the end of it
                    if t - self._fired.get(key, -1e9) < self.p.tripwire_debounce_s:
                        continue
                    self._fired[key] = t
                    direction = self._direction(wire, prev, cur)
                    self.counts[direction] += 1
                    events.append((wire["id"], direction, tr.id))
                    log.info("tripwire", extra={"wire": wire["id"], "dir": direction,
                                                "track_id": tr.id})
        return events

    def forget(self, track_id):
        """Drop a deleted track's state so the dicts do not grow without bound."""
        self._replayed.discard(track_id)
        for d in (self._last, self._fired, self._side):
            for key in [k for k in d if k[0] == track_id]:
                del d[key]
