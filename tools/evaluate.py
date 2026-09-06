"""Score a run against hand-labelled ground truth. Targets: docs/TEST_PROTOCOL.md.

    python -m tools.evaluate footage/groundtruth/overhead_01.json events.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("evaluate")

REQUIRED = ("entries", "occupancy", "stockouts")
TARGETS = {
    "entry_count_error": 0.10,      # +/- 10 %
    "occupancy_mae": 1.5,           # people
    "stockout_recall": 0.95,
    "stockout_false_alarms": 1,     # per 10 minutes
    "queue_count_mae": 1.0,         # people, plan acceptance criterion
    "alert_lead_time_s": 120.0,     # "at least 2 min before the annotated peak"
}

LEAD_WINDOW_S = 600.0               # an alert this long before the peak predicts it


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
    m["passes"] = bool(
        entry_err is not None and abs(entry_err) <= TARGETS["entry_count_error"]
        and occ_mae is not None and occ_mae <= TARGETS["occupancy_mae"]
        and (recall is None or recall >= TARGETS["stockout_recall"])
        and false_alarms <= TARGETS["stockout_false_alarms"]
    )
    return m


def evaluate_queue(events, truth, tolerance_s=2.0) -> dict:
    """Queue half of the W5 accuracy numbers.

    events: queue_estimate events
    truth:  list of {t, lane, count, actual_wait_s} from the queue-clip ground truth
    Matches each truth sample to the nearest prediction within `tolerance_s`.
    """
    preds = [(e.t, e.payload["lane"], e.payload["count"], e.payload["pred_wait_s"])
             for e in events if e.event_type == "queue_estimate"]
    if not preds or not truth:
        return {"count_mae": None, "wait_mae": None, "n": 0}

    count_errs, wait_errs, n = [], [], 0
    for row in truth:
        candidates = [(abs(p[0] - row["t"]), p) for p in preds
                      if p[1] == row["lane"] and abs(p[0] - row["t"]) <= tolerance_s]
        if not candidates:
            continue
        _, p = min(candidates, key=lambda x: x[0])
        count_errs.append(abs(p[2] - row["count"]))
        # Wait times are optional: a short clip can be labelled for headcount
        # and still contain too few service completions to time a wait. A
        # missing label must read as "not measured", not as an error of zero.
        if row.get("actual_wait_s") is not None:
            wait_errs.append(abs(p[3] - row["actual_wait_s"]))
        n += 1

    m = {
        "count_mae": float(np.mean(count_errs)) if count_errs else None,
        "wait_mae": float(np.mean(wait_errs)) if wait_errs else None,
        "n": n,
    }
    m["count_pass"] = (m["count_mae"] is not None
                       and m["count_mae"] <= TARGETS["queue_count_mae"])
    return m


QUEUE_TRUTH_KEYS = ("t", "lane", "count")      # actual_wait_s is optional


def load_queue_truth(path):
    """Read a queue clip's ground truth: `footage/groundtruth/<clip>.json`.

    Expected shape (owned by the dashboard/footage lead, due Thu W3):

        {"queue": [{"t": .., "lane": .., "count": .., "actual_wait_s": ..}, ...],
         "peak_t": ..}     # optional: when the queue visibly peaks in the clip

    `peak_t` is what the staffing lead time is measured against. If the file
    arrives without it, `alert_lead_time` falls back to the busiest labelled
    sample, which is the same instant whenever the labels are dense.
    """
    gt = json.loads(Path(path).read_text())
    rows = gt.get("queue")
    if rows is None:
        raise ValueError(f"{path}: no 'queue' key; this is not a queue clip's truth")
    for row in rows:
        missing = [k for k in QUEUE_TRUTH_KEYS if k not in row]
        if missing:
            raise ValueError(f"{path}: a queue truth row is missing {missing}")
    return rows, gt.get("peak_t")


def store_queue_truth(db, rows, clip):
    """Load ground truth into the `queue_truth` table for wait_pred_vs_actual."""
    db.enqueue("DELETE FROM queue_truth WHERE clip = ?", (clip,))
    for row in rows:
        db.insert("queue_truth", {"clip": clip, "t": float(row["t"]),
                                  "lane": int(row["lane"]),
                                  "count": int(row["count"]),
                                  "actual_wait_s": float(row["actual_wait_s"])})
    db.drain()
    return len(rows)


def peak_time(truth, peak_t=None):
    """When the queue peaks: the labelled `peak_t`, else the busiest sample."""
    if peak_t is not None:
        return float(peak_t)
    if not truth:
        return None
    return float(max(truth, key=lambda r: r["count"])["t"])


def alert_lead_time(events, truth, peak_t=None, rules=("staffing",)):
    """How long before the peak the staffing alert fired. The W5 §2 number.

    Positive is good: the alert came *before* the queue built. Only alerts in
    the ten minutes leading up to the peak count — one fired half an hour early,
    for an unrelated blip, is not a prediction of this peak.
    """
    peak = peak_time(truth, peak_t)
    if peak is None:
        return {"peak_t": None, "alert_t": None, "lead_time_s": None, "passes": False}
    candidates = [e.t for e in events
                  if e.event_type == "alert" and e.payload["rule"] in rules
                  and peak - LEAD_WINDOW_S <= e.t <= peak]
    if not candidates:
        return {"peak_t": peak, "alert_t": None, "lead_time_s": None,
                "passes": False}
    first = min(candidates)
    lead = peak - first
    return {"peak_t": peak, "alert_t": first, "lead_time_s": lead,
            "passes": lead >= TARGETS["alert_lead_time_s"]}


def main(argv=None):
    from core.events import Event

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("groundtruth")
    ap.add_argument("events", help="JSON lines, one Event per line")
    ap.add_argument("--queue", action="store_true",
                    help="score the queue half: count MAE, wait MAE, alert lead time")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    events = [Event(**json.loads(l)) for l in Path(a.events).read_text().splitlines() if l.strip()]
    if a.queue:
        truth, peak_t = load_queue_truth(a.groundtruth)
        m = evaluate_queue(events, truth)
        m.update(alert_lead_time(events, truth, peak_t))
        print(json.dumps(m, indent=2))
        return 0 if m.get("count_pass") and m.get("passes") else 1
    gt = load_groundtruth(a.groundtruth)
    m = evaluate(events, gt)
    print(json.dumps(m, indent=2))
    return 0 if m["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
