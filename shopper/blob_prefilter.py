"""Clean up blobs before they reach the tracker.

A background subtractor does not hand you one box per person. It hands you a
person split at the waist because their shirt matches the floor, and a shadow
stretched across an aisle. Both are fixed here rather than in the tracker, so
sim, reference and PL all feed it the same shapes -- one of the acceptance
criteria is that the three backends emit identical events.

Pure numpy on the structured blob array; no OpenCV, so this stays importable on
a board with no display stack.
"""
from __future__ import annotations

import numpy as np

from pl.contract import BLOB_DT
from shopper.params import TrackerParams


def _boxes(blobs):
    n = len(blobs)
    out = np.empty((n, 4), dtype=float)
    for i, b in enumerate(blobs):
        out[i] = (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
    return out


def _x_overlap_frac(a, b):
    """Shared width as a fraction of the narrower box. 1.0 = perfectly stacked."""
    lo = max(a[0], b[0])
    hi = min(a[0] + a[2], b[0] + b[2])
    return max(0.0, hi - lo) / max(min(a[2], b[2]), 1e-9)


def _v_gap(a, b):
    """Vertical gap between the two boxes; negative if their spans overlap."""
    if a[1] <= b[1]:
        top, bottom = a, b
    else:
        top, bottom = b, a
    return bottom[1] - (top[1] + top[3])


def _merge(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1 = max(a[0] + a[2], b[0] + b[2])
    y1 = max(a[1] + a[3], b[1] + b[3])
    return np.array([x0, y0, x1 - x0, y1 - y0])


def prefilter(blobs, params: TrackerParams | None = None):
    """Merge fragments of one person, drop what cannot be a person.

    Returns a new structured array of BLOB_DT, sorted by area descending like the
    contract requires. `area` is recomputed from the merged geometry, so a person
    reassembled from two fragments scores as one whole person on the tracker's
    area pseudo-score instead of two halves that both fall below it.
    """
    p = params or TrackerParams()
    n = len(blobs)
    if n == 0:
        return np.zeros(0, dtype=BLOB_DT)

    boxes = _boxes(blobs)
    alive = [True] * n

    # Union-merge vertically stacked fragments. Repeat to convergence so a person
    # broken into three pieces collapses to one, not two.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not alive[i]:
                continue
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                gap = _v_gap(boxes[i], boxes[j])
                min_h = min(boxes[i][3], boxes[j][3])
                # Fragments of one person are *stacked*: head above legs, a small
                # gap between them. Two people standing side by side also have a
                # tiny "gap" -- a very negative one, because their vertical spans
                # overlap almost completely -- so a gap test alone fuses them into
                # one blob and loses a shopper. Require near-adjacency, not just
                # proximity.
                if (-p.max_v_overlap_frac * min_h <= gap <= p.merge_gap_px
                        and _x_overlap_frac(boxes[i], boxes[j]) >= p.merge_x_tol):
                    boxes[i] = _merge(boxes[i], boxes[j])
                    alive[j] = False
                    changed = True

    kept = []
    for i in range(n):
        if not alive[i]:
            continue
        x, y, w, h = boxes[i]
        if h > 0 and w / h > p.max_aspect:
            continue        # a shadow, a door swing, a reflection. Never a person.
        kept.append((int(x), int(y), int(w), int(h), int(w * h)))

    kept.sort(key=lambda b: b[4], reverse=True)
    out = np.zeros(len(kept), dtype=BLOB_DT)
    for i, b in enumerate(kept):
        out[i] = b
    return out
