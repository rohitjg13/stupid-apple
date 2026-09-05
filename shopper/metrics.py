"""Shopper half of the accuracy numbers: exits, dwell, and id switches.

Lives here rather than inside tools/evaluate.py because plans 03, 04 and 05 each
own a "half" of that one file; three people editing one function is a three-way
merge conflict waiting to happen. tools/evaluate.py imports this and calls it.

Two ground-truth shapes are in circulation and both are accepted:

* tools/evaluate.py and tests/test_evaluate.py use `entries: [{t, dir}]` with
  `occupancy` as `[t, count]` pairs;
* plan 06 (the annotation tool) specifies separate `entries`/`exits` lists,
  `occupancy` as `{t, count}` dicts, and `people[].zones` for dwell.

Normalising here means the annotation tool and the existing fixtures can both be
scored without either side having to change first.
"""
from __future__ import annotations

import numpy as np

TARGETS = {
    "exit_count_error": 0.10,       # +/- 10 %
    "dwell_mae": 5.0,               # seconds
}
# Temporal IoU, not overlap-over-the-shorter-interval: the latter lets a 4 s
# sliver at the end of a 30 s visit count as a match, because 2 s of overlap is
# half of the sliver. Intersection over union rejects that at 0.06 while still
# accepting a visit the tracker fragmented in half.
MIN_TEMPORAL_IOU = 0.5


def normalise(gt) -> dict:
    """Accept either ground-truth schema; return one shape."""
    entries, exits = [], []
    for e in gt.get("entries", []):
        t = e["t"] if isinstance(e, dict) else e[0]
        if isinstance(e, dict) and e.get("dir") == "out":
            exits.append(float(t))
        else:
            entries.append(float(t))
    for e in gt.get("exits", []):
        exits.append(float(e["t"] if isinstance(e, dict) else e[0]))

    occupancy = []
    for o in gt.get("occupancy", []):
        if isinstance(o, dict):
            occupancy.append((float(o["t"]), int(o["count"])))
        else:
            occupancy.append((float(o[0]), int(o[1])))

    visits = []
    for person in gt.get("people", []):
        for z in person.get("zones", []):
            visits.append({"gt_id": person.get("gt_id"), "zone": z["zone"],
                           "t_enter": float(z["t_enter"]), "t_exit": float(z["t_exit"])})
    return {"entries": entries, "exits": exits, "occupancy": occupancy, "visits": visits}


def _count_error(got, want):
    return (got - want) / want if want else None


def entry_exit_error(events, gt) -> tuple:
    """Fractional count error for entries and exits. None if nothing is labelled."""
    g = normalise(gt)
    tw = [e for e in events if e.event_type == "tripwire"]
    got_in = sum(1 for e in tw if e.payload["dir"] == "in")
    got_out = sum(1 for e in tw if e.payload["dir"] == "out")
    return _count_error(got_in, len(g["entries"])), _count_error(got_out, len(g["exits"]))


def occupancy_mae(events, gt):
    """Mean absolute error against the labelled samples, nearest report in time."""
    g = normalise(gt)
    reported = sorted((e.t, e.payload["count"]) for e in events
                      if e.event_type == "occupancy")
    if not reported or not g["occupancy"]:
        return None
    ts = np.array([r[0] for r in reported])
    counts = np.array([r[1] for r in reported])
    errs = [abs(int(counts[int(np.argmin(np.abs(ts - t)))]) - c) for t, c in g["occupancy"]]
    return float(np.mean(errs))


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def dwell_and_switches(events, gt) -> dict:
    """Match detected visits to labelled ones by zone and time overlap.

    `id_switches` counts labelled visits that were split across more than one
    track. It is a fragmentation proxy, not full MOTA -- that would need
    per-frame ground-truth boxes, which hand annotation does not produce.
    """
    g = normalise(gt)
    detected = [e for e in events if e.event_type == "visit"]
    if not g["visits"]:
        return {"dwell_mae": None, "id_switches": None,
                "visits_detected": len(detected), "visits_labelled": 0}

    errors, switches = [], 0
    for want in g["visits"]:
        span = want["t_exit"] - want["t_enter"]
        overlapping = []
        for e in detected:
            if e.zone_id != want["zone"]:
                continue
            got0, got1 = e.payload["t_enter"], e.payload["t_exit"]
            ov = _overlap(want["t_enter"], want["t_exit"], got0, got1)
            union = max(want["t_exit"], got1) - min(want["t_enter"], got0)
            if union > 0 and ov / union >= MIN_TEMPORAL_IOU:
                overlapping.append((ov, e))
        if not overlapping:
            continue
        switches += len({id(e) for _, e in overlapping}) - 1
        _, best = max(overlapping, key=lambda p: p[0])
        errors.append(abs((best.payload["t_exit"] - best.payload["t_enter"]) - span))

    return {"dwell_mae": float(np.mean(errors)) if errors else None,
            "id_switches": switches,
            "visits_detected": len(detected),
            "visits_labelled": len(g["visits"]),
            "visits_matched": len(errors)}


def evaluate_shopper(events, gt) -> dict:
    """The shopper metrics tools/evaluate.py does not already compute."""
    _, exit_err = entry_exit_error(events, gt)
    out = {"exit_count_error": exit_err}
    out.update(dwell_and_switches(events, gt))
    return out


def passes(metrics) -> bool:
    """True when every shopper metric that could be computed is within target."""
    for key, limit in TARGETS.items():
        value = metrics.get(key)
        if value is not None and abs(value) > limit:
            return False
    return True
