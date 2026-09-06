"""Multi-object tracking over MOG2 blobs. SORT/ByteTrack-shaped, no appearance.

FrameResult carries no confidence score (docs/SHARED.md §4), so **blob area is
the pseudo-score**: big blobs are trusted for the first association pass, small
ones only get to rescue tracks that pass 1 left unmatched. That is ByteTrack's
insight with the one signal this hardware actually gives us.

Deliberately no appearance features and no re-identification: they would break
the privacy claim in docs/DPDP.md, and a 650 MHz A9 has no budget for them.
Track ids are ints from 0 per process and never leave this package except inside
a `visit` payload.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from shopper.kalman import KalmanBox
from shopper.params import TrackerParams

log = logging.getLogger(__name__)

TENTATIVE, CONFIRMED, LOST, DELETED = "tentative", "confirmed", "lost", "deleted"


class Grave:
    """Where a confirmed track was last actually seen, and where it was heading.

    Kept so that a person who walks behind a shelf and out the other side is
    recognised as the same person rather than counted twice.
    """

    __slots__ = ("id", "box", "foot", "vel", "frame", "expires", "far_frames")

    def __init__(self, track, frame, memory):
        self.id = track.id
        self.box = track.seen_box
        x, y, w, h = track.seen_box
        self.foot = (x + w / 2.0, y + h)
        self.vel = track.kf.velocity
        self.frame = frame
        self.expires = frame + memory
        self.far_frames = track.far_frames


def iou_matrix(a, b):
    """Pairwise IoU between boxes (N,4) and (M,4), each (x, y, w, h). -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ax0, ay0 = a[:, 0, None], a[:, 1, None]
    ax1, ay1 = ax0 + a[:, 2, None], ay0 + a[:, 3, None]
    bx0, by0 = b[None, :, 0], b[None, :, 1]
    bx1, by1 = bx0 + b[None, :, 2], by0 + b[None, :, 3]

    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[:, 2, None] * a[:, 3, None] + b[None, :, 2] * b[None, :, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _assign(cost_iou, threshold):
    """Hungarian on -IoU, dropping pairs below `threshold`. -> list of (i, j)."""
    if cost_iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-cost_iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols)
            if cost_iou[r, c] >= threshold]


class Track:
    """One person, alive across frames."""

    __slots__ = ("id", "kf", "state", "hits", "age", "time_since_update",
                 "shared_frames", "held", "trail", "recent", "seen_speed",
                 "travelled", "far_frames", "born_xy", "furniture", "box", "seen_box",
                 "born_t", "last_t", "just_confirmed")

    def __init__(self, track_id, bbox, t, params):
        self.id = track_id
        self.kf = KalmanBox(bbox)
        self.state = TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.shared_frames = 0
        self.held = 0
        # Speed at the last frame we actually saw them. Once the blob is gone the
        # box only coasts, so its apparent speed is the motion model talking to
        # itself; the honest question is whether they were moving when last seen.
        self.seen_speed = float("inf")
        # Furthest this track has ever been *measured* from where it was born.
        # Not a cumulative sum: a basket jittering a pixel a frame for fifteen
        # seconds sums to hundreds of pixels of "travel" and gets held as a
        # shopper, while a real walker's smoothed box lags and under-counts.
        # Peak excursion on raw measurements is immune to both.
        self.born_xy = (float(bbox[0]), float(bbox[1]))
        self.travelled = 0.0        # peak excursion, for reporting
        self.far_frames = 0         # frames measured a body width or more from birth
        self.furniture = False      # retired as a static object, not a person
        self.just_confirmed = False
        self.box = tuple(float(v) for v in bbox)
        # Where they were last *measured*. After max_age frames of coasting the
        # predicted box is 180 px down the aisle from where the person actually
        # stood, and a grave recorded there never lines up with their next track.
        self.seen_box = self.box
        self.born_t = float(t)
        self.last_t = float(t)
        # Foot points from *birth*, not from confirmation. On config/sim a walker
        # spawns 4 px from the door tripwire and crosses it on the second frame,
        # long before the 3-hit confirmation; without this buffer every entry is
        # missed. Real footage has the same shape -- people enter at a frame edge.
        self.trail = deque(maxlen=max(params.n_init + 2, 4))
        # Where the box actually has been. The Kalman velocity lags hard when
        # someone decelerates -- measured at 1.7-2.1 px/frame for people who had
        # already stopped, against a 2.6-3.6 walking speed -- so judging "is this
        # person standing still" on it drops browsers just as they settle. Real
        # displacement over a second does not lag.
        self.recent = deque(maxlen=8)
        self._push_trail(t)
        self.recent.append((self.box[0], self.box[1]))

    def _push_trail(self, t):
        x, y, w, h = self.box
        self.trail.append((float(t), x + w / 2.0, y + h))

    @property
    def foot(self):
        """Ground contact in full-res px, matching pl.contract.blob_foot_point."""
        x, y, w, h = self.box
        return (x + w / 2.0, y + h)

    def has_walked(self, params) -> bool:
        """Sustained movement away from where it appeared: the mark of a person."""
        return self.far_frames >= params.static_min_far_frames

    @property
    def speed(self) -> float:
        """Average px/frame actually travelled recently, not the KF estimate."""
        if len(self.recent) < 2:
            return float("inf")
        (x0, y0), (x1, y1) = self.recent[0], self.recent[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (len(self.recent) - 1)

    @property
    def is_active(self) -> bool:
        """Confirmed and present: what occupancy, visits and the heatmap count.

        A held track counts. Someone browsing a shelf has not left the shop just
        because the background model stopped noticing them.
        """
        return self.state == CONFIRMED and self.time_since_update == 0

    def predict(self):
        self.box = self.kf.predict()
        self.age += 1
        self.recent.append((self.box[0], self.box[1]))

    def update(self, bbox, t, params):
        bx, by = float(bbox[0]), float(bbox[1])
        excursion = ((bx - self.born_xy[0]) ** 2 + (by - self.born_xy[1]) ** 2) ** 0.5
        self.travelled = max(self.travelled, excursion)
        if excursion >= params.static_min_travel_px:
            self.far_frames += 1
        self.box = self.kf.update(bbox)
        self.seen_box = self.box
        self.last_t = float(t)
        self.time_since_update = 0
        self.shared_frames = 0
        self.held = 0
        self._push_trail(t)
        self.recent.append((self.box[0], self.box[1]))
        self.seen_speed = self.speed
        self.hits += 1
        if self.state == LOST:
            self.state = CONFIRMED          # a lost track was confirmed once already
        elif self.state == TENTATIVE and self.hits >= params.n_init:
            self.state = CONFIRMED
            self.just_confirmed = True      # a candidate for re-link this frame

    def mark_shared(self, t):
        """This track has merged into a blob that was assigned to someone else.

        The person has not gone anywhere, so the track stays *counted* -- an
        occlusion must not make occupancy dip. But it is deliberately not fed the
        other person's box as a measurement: that would overwrite its motion
        model with someone else's, and the two would never separate cleanly
        afterwards. It coasts on its own velocity instead.
        """
        self.shared_frames += 1
        self.time_since_update = 0
        self.last_t = float(t)
        self._push_trail(t)

    def mark_missed(self, t, params):
        # Standing still and the blob went away? They did not walk out -- walking
        # out requires walking. The background model simply learned them as
        # furniture. Keep them, and keep counting their dwell.
        if (self.state in (CONFIRMED, LOST) and self.held < params.static_max_age
                and self.seen_speed < params.static_speed_px
                and self.has_walked(params)):
            # LOST is included deliberately. Judging this on one frame makes it a
            # one-way door: a browser whose measured speed blips over the line for
            # a single frame could never be held again, and died 2 s later.
            self.state = CONFIRMED
            self.held += 1
            self.time_since_update = 0
            # Stop coasting. We have just decided this person is standing still,
            # so the motion model should say so -- otherwise the box drifts off
            # them on stale velocity, which both loses the person and keeps the
            # measured speed hovering at the threshold that decides the hold.
            self.kf.x[4:] = 0.0
            self.box = self.kf.bbox
            self.last_t = float(t)
            self._push_trail(t)
            return
        self.time_since_update += 1
        self.last_t = float(t)
        if self.state == TENTATIVE:
            self.state = DELETED          # unconfirmed and already lost: noise
        elif self.time_since_update > params.max_age:
            self.state = DELETED
        elif self.state == CONFIRMED:
            self.state = LOST


class Tracker:
    """Feed it prefiltered blobs once per frame; it hands back live tracks."""

    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.tracks: list[Track] = []
        self.just_deleted: list[Track] = []
        self._next_id = 0
        self._frame = 0
        self._furniture = []        # [(box, expires_at_frame)] -- do not spawn here
        self._graves = []           # Grave objects, for re-linking after an occlusion
        self.relinks = 0            # how many occluded people were recognised again

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _boxes(blobs):
        """Structured blob array -> (N,4) float boxes and (N,) areas."""
        n = len(blobs)
        if n == 0:
            return np.zeros((0, 4)), np.zeros(0)
        boxes = np.empty((n, 4), dtype=float)
        areas = np.empty(n, dtype=float)
        for i, b in enumerate(blobs):
            boxes[i] = (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
            areas[i] = float(b["area"])
        return boxes, areas

    def _spawn(self, bbox, t):
        tr = Track(self._next_id, bbox, t, self.p)
        self._next_id += 1
        # Born where a confirmed track just died? Same person, recycled id. They
        # inherit having walked, so standing still now does not make them furniture.
        if self._graves:
            overlap = iou_matrix([g.box for g in self._graves], [bbox])[:, 0]
            for gi in np.flatnonzero(overlap >= 0.3):
                tr.far_frames = max(tr.far_frames, self._graves[gi].far_frames)
        self.tracks.append(tr)
        return tr

    def _relink_cost(self, grave, tr):
        """How well a newly confirmed track explains a grave. None = not the same.

        Motion and geometry only -- no appearance features, so this does not put
        a face or a shirt colour anywhere near the pipeline.
        """
        p = self.p
        gap = max(self._frame - grave.frame, 1)
        fx, fy = tr.foot
        dx, dy = fx - grave.foot[0], fy - grave.foot[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > p.relink_max_speed_px * gap:
            return None                                 # nobody walks that fast
        size = tr.box[3] / max(grave.box[3], 1.0)
        if not (1.0 / p.relink_size_ratio <= size <= p.relink_size_ratio):
            return None                                 # people do not change height
        vx, vy = grave.vel
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > 1.0 and dist > 1.0:
            # They were walking somewhere when they vanished. Coming back out on
            # the side they went in is a different person, not the same one.
            if (dx * vx + dy * vy) / (dist * speed) < 0.0:
                return None
        return dist / gap                               # prefer the nearest, soonest

    def _try_relink(self, tr):
        best, cost = None, None
        alive = {t.id for t in self.tracks}
        for g in self._graves:
            if g.id in alive:
                continue
            c = self._relink_cost(g, tr)
            if c is not None and (cost is None or c < cost):
                best, cost = g, c
        if best is None:
            return
        tr.id = best.id                                 # the same person, continued
        tr.far_frames = max(tr.far_frames, best.far_frames)
        self._graves.remove(best)
        self.relinks += 1
        log.info("relinked after occlusion",
                 extra={"track_id": tr.id, "gap_frames": self._frame - best.frame})

    # --- main -------------------------------------------------------------
    def update(self, blobs, t):
        """One frame. Returns the currently active (confirmed, seen) tracks."""
        p = self.p
        self.just_deleted = []
        self._frame += 1
        self._furniture = [(b, e) for b, e in self._furniture if e > self._frame]
        self._graves = [g for g in self._graves if g.expires > self._frame]
        boxes, areas = self._boxes(blobs)

        for tr in self.tracks:
            tr.predict()

        track_boxes = [tr.box for tr in self.tracks]
        iou = iou_matrix(track_boxes, boxes)

        # A blob two confirmed tracks both sit on is not one person, it is two
        # people who have merged -- a background subtractor emits their union.
        # Feeding that union to whichever track wins the assignment wrecks its
        # size and scale velocity, and it can no longer match either person once
        # they come apart. So nobody measures it: every claimant coasts on its own
        # motion model until the blobs separate, bounded by overlap_grace.
        contested, claimants = set(), {}
        best = (np.argmax(iou, axis=1) if iou.size else np.zeros(0, dtype=int))
        for dj in range(len(boxes)):
            # A claimant is a confirmed track whose *own best* match is this blob,
            # at the confident threshold. Both halves matter: without "best" a
            # track that has a perfectly good blob of its own still gets dragged
            # into a neighbour's merge, and at the loose threshold two people
            # merely walking near each other look merged. Either mistake starves
            # real tracks of measurements, so they drift, die and respawn -- 9.6x
            # id churn on config/sim, and more tracked people than there are.
            claim = [ti for ti in range(len(self.tracks))
                     if self.tracks[ti].state == CONFIRMED
                     and best[ti] == dj and iou[ti, dj] >= p.iou_high]
            with_grace = [ti for ti in claim
                          if self.tracks[ti].shared_frames < p.overlap_grace]
            if len(claim) >= 2 and len(with_grace) >= 2:
                contested.add(dj)
                claimants[dj] = claim

        high = np.flatnonzero((areas >= p.high_area)
                              & ~np.isin(np.arange(len(areas)), list(contested)))
        low = np.flatnonzero((areas < p.high_area)
                             & ~np.isin(np.arange(len(areas)), list(contested)))
        matched_tracks, matched_dets = set(), set(contested)
        pairs = []

        for dj in contested:
            for ti in claimants[dj]:
                self.tracks[ti].mark_shared(t)
                matched_tracks.add(ti)

        def match(det_idx, threshold):
            """Associate still-unmatched tracks with `det_idx` at `threshold`."""
            rest_t = [i for i in range(len(self.tracks)) if i not in matched_tracks]
            rest_d = [int(d) for d in det_idx if int(d) not in matched_dets]
            if not rest_t or not rest_d:
                return
            for si, dji in _assign(iou[np.ix_(rest_t, rest_d)], threshold):
                ti, dj = rest_t[si], rest_d[dji]
                pairs.append((ti, dj))
                matched_tracks.add(ti)
                matched_dets.add(dj)

        # Pass 1: confident blobs, confident threshold.
        match(high, p.iou_high)
        # Pass 2: small blobs rescue tracks pass 1 missed. They never *start* a
        # track, which is what keeps shadow fragments from becoming shoppers.
        match(low, p.iou_low)
        # Pass 3: a leftover track and a leftover big blob at 0.49 IoU are the same
        # person. Without this they become two identities -- the track goes lost
        # and the blob spawns a duplicate, which is far worse than a loose match.
        match(high, p.iou_low)

        for ti, dj in pairs:
            self.tracks[ti].update(boxes[dj], t, p)

        # Overlap grace: a track left unmatched but sitting on top of a blob that
        # went to someone else has not left the store, it has merged with another
        # person. Two shoppers passing in an aisle become one blob for a few
        # frames; without this, occupancy dips and the visit is cut in half.
        # Rescue-grade association, so it uses the rescue threshold.
        for ti, tr in enumerate(self.tracks):
            if ti in matched_tracks:
                continue
            if (iou.shape[1] and tr.state == CONFIRMED
                    and tr.shared_frames < p.overlap_grace):
                dj = int(np.argmax(iou[ti]))
                if iou[ti, dj] >= p.iou_low and dj in matched_dets:
                    tr.mark_shared(t)
                    matched_tracks.add(ti)
                    continue
            tr.mark_missed(t, p)

        high_set = set(high.tolist())
        if self._furniture and len(boxes):
            # A blob sitting on remembered furniture is the same basket being
            # flagged again, not a new arrival. Refresh the memory while it is.
            fboxes = [b for b, _ in self._furniture]
            on_furniture = iou_matrix(fboxes, boxes)
            for fi in range(len(fboxes)):
                if on_furniture[fi].max() >= 0.5:
                    self._furniture[fi] = (fboxes[fi], self._frame + p.furniture_memory)
            blocked = set(np.flatnonzero(on_furniture.max(axis=0) >= 0.5).tolist())
        else:
            blocked = set()
        for dj in range(len(boxes)):
            if dj not in matched_dets and dj in high_set and dj not in blocked:
                self._spawn(boxes[dj], t)

        # Furniture filter. Something confirmed for a couple of seconds that has
        # never moved a body width from where it appeared is not a shopper who is
        # standing still -- it is a trolley wheel or a glossy basket that the
        # background model keeps flagging. The hold logic never sees these: they
        # get a real match every frame, so they look like a perfectly tracked
        # person with zero speed. A shopper got wherever they are by walking.
        for tr in self.tracks:
            if (tr.state in (CONFIRMED, LOST) and tr.age > p.furniture_age
                    and not tr.has_walked(p)):
                tr.furniture = True
                tr.state = DELETED
                self._furniture.append((tr.box, self._frame + p.furniture_memory))

        for tr in self.tracks:
            if tr.just_confirmed:
                tr.just_confirmed = False
                self._try_relink(tr)

        alive = []
        for tr in self.tracks:
            if tr.state == DELETED:
                self.just_deleted.append(tr)
                if not tr.furniture and tr.hits >= p.n_init:
                    self._graves.append(Grave(tr, self._frame, p.relink_memory))
            else:
                alive.append(tr)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.is_active]

    def close(self):
        """Shutdown: every surviving track counts as deleted so visits flush."""
        self.just_deleted = list(self.tracks)
        self.tracks = []
        return self.just_deleted
