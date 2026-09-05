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
                 "shared_frames", "trail", "box", "born_t", "last_t")

    def __init__(self, track_id, bbox, t, params):
        self.id = track_id
        self.kf = KalmanBox(bbox)
        self.state = TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.shared_frames = 0
        self.box = tuple(float(v) for v in bbox)
        self.born_t = float(t)
        self.last_t = float(t)
        # Foot points from *birth*, not from confirmation. On config/sim a walker
        # spawns 4 px from the door tripwire and crosses it on the second frame,
        # long before the 3-hit confirmation; without this buffer every entry is
        # missed. Real footage has the same shape -- people enter at a frame edge.
        self.trail = deque(maxlen=max(params.n_init + 2, 4))
        self._push_trail(t)

    def _push_trail(self, t):
        x, y, w, h = self.box
        self.trail.append((float(t), x + w / 2.0, y + h))

    @property
    def foot(self):
        """Ground contact in full-res px, matching pl.contract.blob_foot_point."""
        x, y, w, h = self.box
        return (x + w / 2.0, y + h)

    @property
    def is_active(self) -> bool:
        """Confirmed and seen this frame: what occupancy and the heatmap count."""
        return self.state == CONFIRMED and self.time_since_update == 0

    def predict(self):
        self.box = self.kf.predict()
        self.age += 1

    def update(self, bbox, t, params):
        self.box = self.kf.update(bbox)
        self.last_t = float(t)
        self.time_since_update = 0
        self.shared_frames = 0
        self._push_trail(t)
        self.hits += 1
        if self.state == LOST:
            self.state = CONFIRMED          # a lost track was confirmed once already
        elif self.state == TENTATIVE and self.hits >= params.n_init:
            self.state = CONFIRMED

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
        self.tracks.append(tr)
        return tr

    # --- main -------------------------------------------------------------
    def update(self, blobs, t):
        """One frame. Returns the currently active (confirmed, seen) tracks."""
        p = self.p
        self.just_deleted = []
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
        for dj in range(len(boxes)):
            claim = [ti for ti in range(len(self.tracks))
                     if self.tracks[ti].state == CONFIRMED and iou[ti, dj] >= p.iou_low]
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
        for dj in range(len(boxes)):
            if dj not in matched_dets and dj in high_set:
                self._spawn(boxes[dj], t)

        alive = []
        for tr in self.tracks:
            if tr.state == DELETED:
                self.just_deleted.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.is_active]

    def close(self):
        """Shutdown: every surviving track counts as deleted so visits flush."""
        self.just_deleted = list(self.tracks)
        self.tracks = []
        return self.just_deleted
