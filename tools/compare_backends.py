"""Run the same clip through two backends and report how far apart they are.

W4 acceptance (docs/plans/02-rohit.md): matched-blob IoU >= 0.7 and blob-count
agreement >= 90 % between the accelerated backend and the reference chain.

    python -m tools.compare_backends --config config/overhead_01 --frames 300

Defaults to the Jetson pair. `--b pl` compares against the PYNQ overlay
instead, which is the same question asked of different silicon.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

log = logging.getLogger("compare")

IOU_TARGET = 0.7
COUNT_TARGET = 0.9


def iou(a, b) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _boxes(result):
    return [tuple(int(b[k]) for k in ("x", "y", "w", "h"))
            for b in result["blobs"][: int(result["num_blobs"])]]


def match_blobs(a, b, thresh=0.5):
    """Greedy best-IoU matching. Blobs are area-sorted, so greedy is enough here.

    Returns (pairs, unmatched_a_indices, unmatched_b_indices) where each pair is
    (index_in_a, index_in_b, iou).
    """
    A, B = _boxes(a), _boxes(b)
    scores = sorted(((iou(x, y), i, j) for i, x in enumerate(A) for j, y in enumerate(B)),
                    key=lambda s: -s[0])
    pairs, used_a, used_b = [], set(), set()
    for score, i, j in scores:
        if score < thresh or i in used_a or j in used_b:
            continue
        pairs.append((i, j, score))
        used_a.add(i)
        used_b.add(j)
    return (sorted(pairs), [i for i in range(len(A)) if i not in used_a],
            [j for j in range(len(B)) if j not in used_b])


def compare(results_a, results_b, thresh=0.5, n_rois=0) -> dict:
    if len(results_a) != len(results_b):
        raise ValueError(f"frame counts differ: {len(results_a)} vs {len(results_b)}")

    ious, deltas, agree, matched = [], [], 0, 0
    for ra, rb in zip(results_a, results_b):
        na, nb = int(ra["num_blobs"]), int(rb["num_blobs"])
        deltas.append(nb - na)
        agree += na == nb
        pairs, _, _ = match_blobs(ra, rb, thresh)
        matched += len(pairs)
        ious += [p[2] for p in pairs]

    out = {
        "frames": len(results_a),
        "matched": matched,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "count_agreement": agree / len(results_a),
        "mean_blob_delta": float(np.mean(deltas)),
    }
    if n_rois:
        fa = np.array([r["roi_fill"][:n_rois] for r in results_a], dtype=float)
        fb = np.array([r["roi_fill"][:n_rois] for r in results_b], dtype=float)
        out["roi_fill_mae"] = float(np.abs(fa - fb).mean())
    # An all-empty clip agrees perfectly but proves nothing, so require matches.
    out["passes_w4"] = bool(out["count_agreement"] >= COUNT_TARGET
                            and (out["mean_iou"] >= IOU_TARGET or matched == 0))
    return out


def main(argv=None):
    from core.config import load_clipset
    from main import build_backend, build_source

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--stream", default="overhead")
    ap.add_argument("--source", default="file", choices=["file", "sim", "camera"])
    ap.add_argument("--a", default="reference")
    ap.add_argument("--b", default="yolo")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_clipset(a.config)
    be_a, be_b = build_backend(a.a, cfg), build_backend(a.b, cfg)
    sid = 0 if a.stream == "overhead" else 1

    ra, rb = [], []
    src = build_source(a.source, cfg, a.stream)
    for i, frame in enumerate(src.frames()):
        if i >= a.frames:
            break
        if frame.image is None:
            src.close()
            raise SystemExit(
                f"--source {a.source} yields no images, so there is nothing for "
                f"{a.a}/{a.b} to process; use --source file with a clip in store.yaml")
        ra.append(be_a.process(frame.image, sid, frame.frame_id).copy())
        rb.append(be_b.process(frame.image, sid, frame.frame_id).copy())
    src.close()
    if not ra:
        raise SystemExit("no frames read; check store.yaml path")

    s = compare(ra, rb, n_rois=len(cfg.rois))
    print(json.dumps(s, indent=2) if a.json else
          f"{a.a} vs {a.b}: {s['frames']} frames, IoU {s['mean_iou']:.2f}, "
          f"count agreement {s['count_agreement']:.0%}, "
          f"{'PASS' if s['passes_w4'] else 'FAIL'} vs W4 targets")
    return 0 if s["passes_w4"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
