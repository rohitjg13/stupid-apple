"""An 8-state constant-velocity Kalman filter over one bounding box.

State is `[cx, cy, w, h, vcx, vcy, vw, vh]` in full-res image pixels; the
measurement is the box alone. Only position is extrapolated -- see `F` for why
size velocity is deliberately left out of the transition.

Noise scales with box height, so a person near the camera (big box, fast
apparent motion) gets a looser gate than one far away -- the standard
SORT/DeepSORT trick, and it matters on an overhead view where apparent speed
varies a lot across the floor.

Hand-rolled on purpose: `filterpy` is not in the board dependency list in
pyproject.toml and this is 8x8 linear algebra, not a library problem.
"""
from __future__ import annotations

import numpy as np

# Height-relative standard deviations. Position noise is generous because the
# blobs come from a background subtractor, which breathes by a few pixels a frame
# even on a stationary person.
STD_POS = 1.0 / 20
STD_VEL = 1.0 / 160


def to_center(bbox):
    """(x, y, w, h) top-left -> (cx, cy, w, h) centre."""
    x, y, w, h = (float(v) for v in bbox)
    return np.array([x + w / 2, y + h / 2, w, h])


def to_corner(z):
    """(cx, cy, w, h) centre -> (x, y, w, h) top-left."""
    cx, cy, w, h = (float(v) for v in z[:4])
    return (cx - w / 2, cy - h / 2, w, h)


class KalmanBox:
    """Track one box. `predict()` then `update(bbox)` once per frame."""

    def __init__(self, bbox, dt: float = 1.0):
        self.F = np.eye(8)
        # Constant velocity on *position* only. Size is a random walk: it still
        # tracks a measured box growing or shrinking, but is never extrapolated.
        # With size velocity, a few frames of a person being absorbed into the
        # background teach the filter a steep negative height rate, and the moment
        # the blob vanishes the coasting box collapses to a 1 px sliver that
        # overlaps nothing -- the track can never re-match and dies. A person does
        # not shrink to nothing because we stopped seeing them.
        for i in range(2):
            self.F[i, i + 4] = dt
        self.H = np.zeros((4, 8))
        self.H[:, :4] = np.eye(4)

        self.x = np.zeros(8)
        self.x[:4] = to_center(bbox)

        # Position is known about as well as a measurement; velocity is not known
        # at all from a single frame, so it starts wide and tightens with hits.
        h = max(self.x[3], 1.0)
        self.P = np.diag(np.square(np.array(
            [2 * STD_POS * h] * 4 + [10 * STD_VEL * h] * 4)))

    # --- noise ------------------------------------------------------------
    def _h(self) -> float:
        return max(float(self.x[3]), 1.0)

    def _Q(self):
        h = self._h()
        return np.diag(np.square(np.array([STD_POS * h] * 4 + [STD_VEL * h] * 4)))

    def _R(self):
        h = self._h()
        return np.diag(np.square(np.array([STD_POS * h] * 4)))

    # --- filter -----------------------------------------------------------
    def predict(self):
        """Advance one frame. Safe to call repeatedly while a track is lost."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()
        # A predicted box may not collapse or invert; IoU on a zero box is 0 and
        # the track would never be recoverable.
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.bbox

    def update(self, bbox):
        """Fold in a measured box."""
        z = to_center(bbox)
        S = self.H @ self.P @ self.H.T + self._R()
        # Solve rather than invert: same answer, better conditioned, and S is 4x4.
        K = np.linalg.solve(S, self.H @ self.P.T).T
        self.x = self.x + K @ (z - self.H @ self.x)
        I_KH = np.eye(8) - K @ self.H
        # Joseph form keeps P symmetric positive-definite over long tracks.
        self.P = I_KH @ self.P @ I_KH.T + K @ self._R() @ K.T
        return self.bbox

    @property
    def bbox(self):
        """Current estimate as (x, y, w, h) top-left, floats."""
        return to_corner(self.x)

    @property
    def velocity(self):
        """(vcx, vcy) px/frame. Used to disambiguate crossing tracks."""
        return float(self.x[4]), float(self.x[5])
