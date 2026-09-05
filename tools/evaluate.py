"""Score a run against hand-labelled ground truth. Targets: docs/TEST_PROTOCOL.md.

    python -m tools.evaluate footage/groundtruth/overhead_01.json events.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from shopper.metrics import TARGETS as SHOPPER_TARGETS
from shopper.metrics import evaluate_shopper
from shopper.metrics import passes as shopper_passes
from shopper.metrics import write_result

log = logging.getLogger("evaluate")

REQUIRED = ("entries", "occupancy", "stockouts")
TARGETS = {
    "entry_count_error": 0.10,      # +/- 10 %
    "occupancy_mae": 1.5,           # people
    "stockout_recall": 0.95,
    "stockout_false_alarms": 1,     # per 10 minutes
    **SHOPPER_TARGETS,              # exit_count_error, dwell_mae
}


def load_groundtruth(path):
    gt = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in gt]
    if missing:
        raise ValueError(f"{path}: ground truth is missing {missing}")
    return gt


def evaluate(events, gt, stockout_tolerance_s=10.0) -> dict:
    events = list(events)

    # --- entries -----------------------------------------------------------
    got_in = sum(1 for e in events if e.event_type == "tripwire" and e.payload["dir"] == "in")
    want_in = sum(1 for e in gt["entries"] if e["dir"] == "in")
    entry_err = (got_in - want_in) / want_in if want_in else None

    # --- occupancy: compare each labelled sample to the nearest report ------
    reported = sorted((e.t, e.payload["count"]) for e in events if e.event_type == "occupancy")
    if reported and gt["occupancy"]:
        ts = np.array([r[0] for r in reported])
        counts = np.array([r[1] for r in reported])
        errs = [abs(int(counts[int(np.argmin(np.abs(ts - t)))]) - c) for t, c in gt["occupancy"]]
        occ_mae = float(np.mean(errs))
    else:
        occ_mae = None

    # --- stock-outs --------------------------------------------------------
    starts = [(e.t, e.zone_id) for e in events if e.event_type == "stockout_start"]
    matched, used = 0, set()
    for want in gt["stockouts"]:
        for i, (t, facing) in enumerate(starts):
            if i in used or facing != want["facing"]:
                continue
            if abs(t - want["t_start"]) <= stockout_tolerance_s:
                matched += 1
                used.add(i)
                break
    recall = matched / len(gt["stockouts"]) if gt["stockouts"] else None
    false_alarms = len(starts) - len(used)

    m = {
        "entry_count_error": entry_err,
        "occupancy_mae": occ_mae,
        "stockout_recall": recall,
        "stockout_false_alarms": false_alarms,
        "entries_detected": got_in,
        "entries_labelled": want_in,
    }
    # Shopper half lives in shopper/metrics.py: plans 03, 04 and 05 each own a
    # "half" of this file, and three people editing one function conflicts.
    m.update(evaluate_shopper(events, gt))

    m["passes"] = bool(
        entry_err is not None and abs(entry_err) <= TARGETS["entry_count_error"]
        and occ_mae is not None and occ_mae <= TARGETS["occupancy_mae"]
        and (recall is None or recall >= TARGETS["stockout_recall"])
        and false_alarms <= TARGETS["stockout_false_alarms"]
        and shopper_passes(m)
    )
    return m


def main(argv=None):
    from core.events import Event

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("groundtruth")
    ap.add_argument("events", help="JSON lines, one Event per line")
    ap.add_argument("--results-dir", default="results",
                    help="where to write <clip>.json for the accuracy slide")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    gt = load_groundtruth(a.groundtruth)
    events = [Event(**json.loads(l)) for l in Path(a.events).read_text().splitlines() if l.strip()]
    m = evaluate(events, gt)
    log.info("wrote %s", write_result(m, gt, a.results_dir))
    print(json.dumps(m, indent=2))
    return 0 if m["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
