#!/usr/bin/env python3
"""Score and visualise a queue clip against its hand labels.

    python3 tools/queue_report.py --db /tmp/qrun/queue.db --run-id queue_01 \
        --truth footage/groundtruth/queue_01.json --config config/queue_01 \
        --clip footage/queue_01.mp4 --out out/queue_01

Reads what the pipeline already wrote (`lane_occ`, `queue_estimate`) rather
than re-running the CV, so what is scored is exactly what ran. It then does the
W3 §1 calibration — least squares `count = a*sum(cells) + b` on the labels —
reports R2/MAE, and renders an annotated video so the numbers can be checked
against the footage by eye.

The fit is deliberately reported *both* ways: uncalibrated (the naive rule the
pipeline ships with before a fit exists) and calibrated. On a clip whose true
count barely varies the calibrated MAE will look excellent and mean very
little; the report says so rather than quoting the number alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import DB
from backend.queue_model import fit_regression, naive_count
from tools.evaluate import load_queue_truth

log = logging.getLogger("queue_report")


def load_cells(db, run_id):
    """[(t, [16 cells])] as the pipeline stored them."""
    return [(r["t"], list(r["cells"]))
            for r in db.query("SELECT t, cells FROM lane_occ WHERE run_id = ? "
                              "AND lane = 1 ORDER BY t", (run_id,))]


def load_estimates(db, run_id):
    return [(r["t"], r["count"], r["pred_wait_s"])
            for r in db.query("SELECT t, count, pred_wait_s FROM queue_estimate "
                              "WHERE run_id = ? AND lane = 1 ORDER BY t", (run_id,))]


def nearest(series, t, tolerance_s):
    """The sample closest to `t`, or None if nothing is within tolerance."""
    best, best_d = None, None
    for row in series:
        d = abs(row[0] - t)
        if best_d is None or d < best_d:
            best, best_d = row, d
    return best if best_d is not None and best_d <= tolerance_s else None


def pair_with_truth(cells, truth, t0, exclude, tolerance_s):
    """Match each label to the nearest lane_occ sample. Returns matched rows."""
    out = []
    for row in truth:
        if row.get("frame") in exclude:
            continue
        got = nearest(cells, t0 + row["t"], tolerance_s)
        if got is None:
            continue
        out.append({"t": row["t"], "frame": row.get("frame"),
                    "truth": row["count"],
                    "sum_cells": sum(got[1][:8]),
                    "cells": got[1][:8]})
    return out


def mae(pairs, predict):
    return sum(abs(predict(p) - p["truth"]) for p in pairs) / len(pairs)


def report(db, run_id, truth_path, tolerance_s=1.0):
    truth, peak_t = load_queue_truth(truth_path)
    meta = json.loads(Path(truth_path).read_text())
    exclude = set(meta.get("exclude_from_fit", []))

    cells = load_cells(db, run_id)
    estimates = load_estimates(db, run_id)
    if not cells:
        raise SystemExit(f"no lane_occ rows for run {run_id!r}; run main.py first")

    # Labels are in clip time (t=0 at the first frame); the database is in
    # epoch time. Anchor on the first lane_occ sample.
    t0 = cells[0][0]
    pairs = pair_with_truth(cells, truth, t0, exclude, tolerance_s)
    if len(pairs) < 2:
        raise SystemExit("not enough labels matched lane_occ samples to fit")

    xs = [p["sum_cells"] for p in pairs]
    ys = [p["truth"] for p in pairs]
    a, b, r2, fit_mae = fit_regression(xs, ys)

    naive_mae = mae(pairs, lambda p: naive_count(p["cells"]))
    calib_mae = mae(pairs, lambda p: max(0.0, a * p["sum_cells"] + b))
    # The control that decides whether the fit means anything: a model that
    # ignores the image entirely and always answers the mean. If the calibrated
    # MAE is not clearly better than this, the regression has learnt the
    # average queue length, not how to see one.
    constant = sum(ys) / len(ys)
    constant_mae = mae(pairs, lambda p: constant)

    truth_values = sorted({p["truth"] for p in pairs})
    return {
        "run_id": run_id,
        "clip": meta.get("clip"),
        "labels": {"n": len(pairs), "excluded": sorted(exclude),
                   "distinct_counts": truth_values},
        "signal": {
            "lane_occ_samples": len(cells),
            "sum_cells_min": min(xs), "sum_cells_max": max(xs),
            "sum_cells_mean": round(sum(xs) / len(xs), 1),
        },
        "regression": {"a": a, "b": b, "r2": r2, "fit_mae": fit_mae},
        "count_mae": {"uncalibrated_naive_rule": naive_mae,
                      "calibrated": calib_mae,
                      "constant_predictor": constant_mae},
        "beats_constant_predictor": calib_mae < constant_mae * 0.9,
        "target_count_mae": 1.0,
        "passes_count_target": calib_mae <= 1.0,
        "wait": {
            "measured": False,
            "why": "no actual_wait_s labels: one service completion in 45 s",
            "predicted_range_s": ([round(min(e[2] for e in estimates), 1),
                                   round(max(e[2] for e in estimates), 1)]
                                  if estimates else None),
        },
        "peak_t": peak_t,
        "pairs": pairs,
    }


def render(clip, cfg_path, out_path, db, run_id, a, b, fps=30.0):
    """Annotated video: cells, live occupancy, calibrated count, predicted wait."""
    import cv2
    import numpy as np

    from core.config import load_clipset
    from pl.reference import letterbox

    cfg = load_clipset(cfg_path)
    lanes = sorted(cfg.lanes, key=lambda l: l["cell"])
    cells = load_cells(db, run_id)
    estimates = load_estimates(db, run_id)
    if not cells:
        raise SystemExit("nothing to render")
    t0 = cells[0][0]

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    w, h = 640, 480
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    fid = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = t0 + fid / fps
        vis = letterbox(img)
        sample = nearest(cells, t, 1.0)
        est = nearest(estimates, t, 2.0)
        occ = sample[1][:8] if sample else [0] * 8

        for lane in lanes:
            # Config is at PL 320x240; the frame is 640x480.
            x, y = lane["x"] * 2, lane["y"] * 2
            cw, ch = lane["w"] * 2, lane["h"] * 2
            v = occ[lane["cell"]] if lane["cell"] < len(occ) else 0
            # Green when the cell reads occupied, dim grey when it does not.
            colour = (0, 200, 0) if v > 40 else (90, 90, 90)
            cv2.rectangle(vis, (x, y), (x + cw, y + ch), colour, 2 if v > 40 else 1)
            cv2.putText(vis, str(v), (x + 4, y + ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        count = max(0.0, a * sum(occ) + b)
        cv2.rectangle(vis, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(vis, f"queue {count:.1f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if est:
            cv2.putText(vis, f"pred wait {est[2]:.0f}s", (170, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis, f"sum(cells) {sum(occ)}", (330, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"t {fid / fps:5.1f}s", (530, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        writer.write(vis)
        fid += 1
    cap.release()
    writer.release()
    return fid


def plot(rep, out_path):
    """Predicted vs labelled count over the clip, as an SVG (no matplotlib)."""
    pairs = rep["pairs"]
    a, b = rep["regression"]["a"], rep["regression"]["b"]
    W, H, pad = 900, 320, 50
    tmax = max(p["t"] for p in pairs) or 1.0
    ymax = max(4.0, max(p["truth"] for p in pairs) + 1)

    def X(t):
        return pad + (W - 2 * pad) * t / tmax

    def Y(v):
        return H - pad - (H - 2 * pad) * v / ymax

    truth_pts = " ".join(f"{X(p['t']):.1f},{Y(p['truth']):.1f}" for p in pairs)
    pred_pts = " ".join(
        f"{X(p['t']):.1f},{Y(max(0.0, a * p['sum_cells'] + b)):.1f}" for p in pairs)
    grid = "".join(
        f'<line x1="{pad}" y1="{Y(v):.1f}" x2="{W - pad}" y2="{Y(v):.1f}" '
        f'stroke="#e5e5e5"/><text x="{pad - 8}" y="{Y(v) + 4:.1f}" '
        f'font-size="11" fill="#888" text-anchor="end">{v}</text>'
        for v in range(0, int(ymax) + 1))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui,sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
{grid}
<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#333"/>
<polyline points="{truth_pts}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
<polyline points="{pred_pts}" fill="none" stroke="#d62728" stroke-width="2" stroke-dasharray="5 3"/>
<text x="{pad}" y="24" font-size="14" fill="#222">queue_01 — calibrated count vs hand labels</text>
<text x="{pad}" y="42" font-size="11" fill="#666">MAE {rep["count_mae"]["calibrated"]:.2f} people, R2 {rep["regression"]["r2"]:.2f}, {rep["labels"]["n"]} labels</text>
<text x="{W - pad}" y="24" font-size="11" fill="#1f77b4" text-anchor="end">labelled</text>
<text x="{W - pad}" y="40" font-size="11" fill="#d62728" text-anchor="end">predicted</text>
<text x="{W / 2}" y="{H - 12}" font-size="11" fill="#666" text-anchor="middle">seconds into clip</text>
</svg>'''
    Path(out_path).write_text(svg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--config", default="config/queue_01")
    ap.add_argument("--clip", default=None, help="render an annotated video too")
    ap.add_argument("--out", default="out/queue_01")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    db = DB(a.db)
    try:
        rep = report(db, a.run_id, a.truth)
        (out / "report.json").write_text(json.dumps(rep, indent=2))
        plot(rep, out / "count_vs_truth.svg")
        if a.clip:
            n = render(a.clip, a.config, out / "annotated.mp4", db, a.run_id,
                       rep["regression"]["a"], rep["regression"]["b"], a.fps)
            log.info("rendered %d frames", n)
    finally:
        db.close()

    r = rep["regression"]
    print(f"clip            {rep['clip']}  ({rep['labels']['n']} labels, "
          f"counts {rep['labels']['distinct_counts']})")
    print(f"sum(cells)      min {rep['signal']['sum_cells_min']}  "
          f"mean {rep['signal']['sum_cells_mean']}  max {rep['signal']['sum_cells_max']}")
    print(f"regression      count = {r['a']:.5f} * sum(cells) + {r['b']:.3f}")
    print(f"                R2 {r['r2']:.3f}")
    m = rep["count_mae"]
    print(f"count MAE       {m['uncalibrated_naive_rule']:.2f} uncalibrated "
          f"-> {m['calibrated']:.2f} calibrated  (target <= {rep['target_count_mae']})")
    print(f"                {m['constant_predictor']:.2f} for a constant predictor "
          f"that ignores the image entirely")
    if not rep["beats_constant_predictor"]:
        print("                ^ the fit does NOT clearly beat that control: on this "
              "clip the\n                  regression has learnt the average queue "
              "length, not how to see one.")
    print(f"wait MAE        not measured: {rep['wait']['why']}")
    print(f"output          {out}/")
    return 0 if rep["passes_count_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
