"""Image px <-> floor metres. SHARED.md §6.

`H` is always the image->floor 3x3 loaded from config/homography_<stream>.npy.
"""
from __future__ import annotations

import numpy as np


def _apply(M, pts):
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {M.shape}")
    p = np.asarray(pts, dtype=float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    if p.shape[1] != 2:
        raise ValueError(f"expected (N,2) points, got {p.shape}")
    hom = np.hstack([p, np.ones((len(p), 1))]) @ M.T
    out = hom[:, :2] / hom[:, 2:3]
    return out[0] if single else out


def image_to_floor(H, uv):
    """(u,v) px -> (X,Y) metres. Accepts one point or (N,2)."""
    return _apply(H, uv)


def floor_to_image(H, xy):
    """(X,Y) metres -> (u,v) px."""
    return _apply(np.linalg.inv(np.asarray(H, dtype=float)), xy)
