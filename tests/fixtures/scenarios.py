"""Synthetic blob sequences for the shopper tests.

Every shopper test rides on this: it builds FRAME_RESULT_DT records directly, so
the tracker can be exercised without OpenCV, without a board and without the sim.

Boxes are `(x, y, w, h)` in full-res 640x480 px, matching what the PL emits after
it doubles its 320x240 coordinates back up (docs/SHARED.md §4). `area` is the box
area, which is what a solid connected component gives you.
"""
from __future__ import annotations

import numpy as np

from pl.contract import FLAG_BG_WARM, FLAG_TRUNCATED, FRAME_RESULT_DT, MAX_BLOBS, Frame

FULL_W, FULL_H = 640, 480
DEFAULT_FPS = 15.0

# A person at the sim's 80 px/m: ~0.5 m wide, ~1.7 m tall.
PERSON_W, PERSON_H = 40, 136


def result_from_boxes(frame_id, boxes, warm=True):
    """Build one FrameResult. Blobs are sorted by area descending, as the PL does."""
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    r["frame_id"] = frame_id
    if warm:
        r["flags"] |= FLAG_BG_WARM

    ordered = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    if len(ordered) > MAX_BLOBS:
        r["flags"] |= FLAG_TRUNCATED
    r["num_blobs"] = min(len(ordered), MAX_BLOBS)
    for i, (x, y, w, h) in enumerate(ordered[:MAX_BLOBS]):
        r["blobs"][i] = (x, y, w, h, w * h)
    return r


def frame_from_boxes(frame_id, boxes, warm=True, stream="overhead", t_ns=None,
                     fps=DEFAULT_FPS):
    t_ns = int(frame_id * 1e9 / fps) if t_ns is None else int(t_ns)
    return Frame(stream, frame_id, t_ns, result_from_boxes(frame_id, boxes, warm))


def sequence(box_lists, warm_from=0, stream="overhead", fps=DEFAULT_FPS):
    """Per-frame box lists -> Frames. `warm_from` mimics the background warm-up."""
    return [frame_from_boxes(i, boxes, warm=i >= warm_from, stream=stream, fps=fps)
            for i, boxes in enumerate(box_lists)]


def times(n, fps=DEFAULT_FPS, t0=1_000_000.0):
    """Epoch seconds for n frames, the way main.py hands them to the pipeline."""
    return [t0 + i / fps for i in range(n)]


# --- scenarios ------------------------------------------------------------
# Each returns a list of per-frame box lists, so they compose and can be sliced.

def linear_walk(n=30, x0=100, y0=200, dx=6, dy=0, w=PERSON_W, h=PERSON_H):
    """One person walking in a straight line. The baseline happy path."""
    return [[(x0 + dx * i, y0 + dy * i, w, h)] for i in range(n)]


def crossing(n=40, y=200, sep=10, speed=8, w=PERSON_W, h=PERSON_H):
    """Two people walking through each other. The classic id-switch test.

    They are offset vertically by `sep` so the overlap is high but never total;
    a tracker with no motion model swaps them at the midpoint.
    """
    out = []
    span = speed * (n - 1)
    for i in range(n):
        left = (100 + speed * i, y, w, h)
        right = (100 + span - speed * i, y + sep, w, h)
        out.append([left, right])
    return out


def union_box(a, b):
    """The single blob a background subtractor gives you for two touching people."""
    x0 = min(a[0], b[0])
    y0 = min(a[1], b[1])
    x1 = max(a[0] + a[2], b[0] + b[2])
    y1 = max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def merge_and_split(n=24, merge=(9, 15), y=200, sep=10, speed=6,
                    w=PERSON_W, h=PERSON_H):
    """Two people who touch, become one blob, and come apart again.

    The classic overhead-footage failure: while merged there is exactly one blob,
    so one of the two tracks has nothing to match against. It must neither die
    nor be dragged onto the other person's motion.
    """
    out = []
    span = speed * (n - 1)
    lo, hi = merge
    for i in range(n):
        left = (100 + speed * i, y, w, h)
        right = (100 + span - speed * i, y + sep, w, h)
        out.append([union_box(left, right)] if lo <= i < hi else [left, right])
    return out


def occlusion(n=40, gap=(15, 25), **kw):
    """One person who vanishes for `gap` frames (a pillar, a merge) and returns."""
    walk = linear_walk(n=n, **kw)
    lo, hi = gap
    return [[] if lo <= i < hi else boxes for i, boxes in enumerate(walk)]


def split_person(n=20, split=(5, 15), x0=100, y0=200, dx=6,
                 w=PERSON_W, h=PERSON_H, gap=6):
    """One person that fragments into head and legs for part of the sequence.

    Two boxes, same x, separated by a small vertical gap: exactly what MOG2 does
    to a person wearing a shirt the colour of the floor. The prefilter must put
    them back together.
    """
    out = []
    lo, hi = split
    for i in range(n):
        x = x0 + dx * i
        if lo <= i < hi:
            top = h // 2 - gap // 2
            out.append([(x, y0, w, top), (x, y0 + top + gap, w, h - top - gap)])
        else:
            out.append([(x, y0, w, h)])
    return out


def noise_slivers(n=10, x=300, y=300, w=120, h=20):
    """Wide flat blobs: a shadow, a door opening, a reflection. Never a person."""
    return [[(x, y, w, h)] for _ in range(n)]


def too_many_blobs(n=5, count=70, w=20, h=20):
    """More blobs than the 64-slot contract holds; sets FLAG_TRUNCATED."""
    return [[((i * 9) % 600, (i * 7) % 440, w, h) for i in range(count)]
            for _ in range(n)]


def empty(n=10):
    """An empty store. Must not crash anything."""
    return [[] for _ in range(n)]


def flood(n=5, w=600, h=440):
    """A lighting change or a camera bump: most of the frame goes foreground."""
    return [[(10, 10, w, h)] for _ in range(n)]


def walk_across_line(n=20, y_line=460, start_below=True, x=120,
                     w=PERSON_W, h=PERSON_H, step=8):
    """A person whose foot point crosses a horizontal tripwire.

    The foot point is `(x + w // 2, y + h)`, so the boxes are positioned by their
    feet: `y = foot_v - h`. Mirrors the sim, where walkers spawn at v=464 and the
    door line sits at v=460 -- four pixels, crossed on the second frame.
    """
    v0 = y_line + 4 if start_below else y_line - 4
    direction = -1 if start_below else 1
    return [[(x, v0 + direction * step * i - h, w, h)] for i in range(n)]
