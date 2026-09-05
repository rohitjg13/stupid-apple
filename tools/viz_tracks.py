"""Draw tracks over the video. The fastest way to see an id switch.

    python -m tools.viz_tracks --source sim --frames 300 --out tracks.mp4
    python -m tools.viz_tracks --source file --backend reference --config config/overhead_01

On `--source sim` there is no image (sim yields FrameResults, not frames), so
boxes are drawn on a blank canvas. That is still worth having: an id switch is
obvious on screen and tedious to assert.

This is the one file in the shopper work that imports cv2. shopper/ itself must
stay importable on a board with no display stack.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import cv2
import numpy as np

from core.config import load_clipset
from geometry.homography import floor_to_image, image_to_floor
from geometry.zones import ZoneMap
from pl.contract import FLAG_BG_WARM
from shopper.blob_prefilter import prefilter
from shopper.params import TrackerParams
from shopper.tracker import Tracker
from shopper.tripwire import TripwireCounter

log = logging.getLogger("viz_tracks")

W, H = 640, 480
# Distinct hues so neighbouring ids never look alike; an id switch changes colour.
PALETTE = [(66, 135, 245), (245, 176, 66), (66, 245, 129), (245, 66, 152),
           (245, 245, 66), (167, 66, 245), (66, 245, 245), (245, 111, 66)]


def colour(track_id):
    return PALETTE[track_id % len(PALETTE)]


def draw(canvas, tracks, zone_by_track, wires, counts, frame_id, warm):
    for wire in wires:
        p1, p2 = tuple(map(int, wire["p1"])), tuple(map(int, wire["p2"]))
        cv2.line(canvas, p1, p2, (0, 0, 255), 2)
        cv2.putText(canvas, wire["id"], (p1[0], p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    for tr in tracks:
        x, y, w, h = (int(v) for v in tr.box)
        c = colour(tr.id)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), c, 2)
        fu, fv = (int(v) for v in tr.foot)
        cv2.circle(canvas, (fu, fv), 4, c, -1)
        label = f"#{tr.id}"
        zone = zone_by_track.get(tr.id)
        if zone:
            label += f" {zone}"
        cv2.putText(canvas, label, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)

    banner = (f"frame {frame_id}  tracks {len(tracks)}  "
              f"in {counts['in']}  out {counts['out']}"
              f"{'' if warm else '   [background warming up]'}")
    cv2.rectangle(canvas, (0, 0), (W, 24), (24, 24, 24), -1)
    cv2.putText(canvas, banner, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def draw_zones(canvas, cfg, stream):
    """Zone outlines, projected back into the image through the homography."""
    Hm = cfg.homography.get(stream)
    if Hm is None:
        return
    for z in cfg.zones:
        pts = floor_to_image(Hm, np.asarray(z["polygon"], dtype=float))
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, (90, 90, 90), 1)
        cx, cy = poly[:, 0, 0].mean(), poly[:, 0, 1].mean()
        cv2.putText(canvas, z["id"], (int(cx) - 20, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--video", default=None,
                    help="test any video file directly; builds a throwaway clipset")
    ap.add_argument("--bg-lr", type=float, default=0.005,
                    help="MOG2 learning rate. Lower = remembers stationary people "
                         "for longer, but adapts to lighting more slowly")
    ap.add_argument("--morph", type=int, default=1,
                    help="morphological opening passes; raise to 2 to kill noise")
    ap.add_argument("--high-area", type=int, default=None,
                    help="blob area that counts as a confident detection. Lower it "
                         "when people are small in frame")
    ap.add_argument("--source", default="sim", choices=["sim", "file", "camera"])
    ap.add_argument("--backend", default="sim", choices=["sim", "reference", "pl"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--stream", default="overhead")
    ap.add_argument("--out", default=None, help="write an mp4 instead of a window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=25.0, help="playback fps for --video")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from main import build_backend, build_source

    config = a.config
    if a.video:
        # Copy the clipset and point it at this file, so any video can be tried
        # without hand-editing a store.yaml.
        import shutil, tempfile
        from pathlib import Path
        config = Path(tempfile.mkdtemp()) / "clipset"
        shutil.copytree(a.config, config)
        y = config / "store.yaml"
        y.write_text(y.read_text().replace(
            "overhead: {source: sim, fps: 15, loop: true}",
            f"overhead: {{source: file, path: {Path(a.video).resolve()}, "
            f"fps: {a.fps}, loop: false}}"))
        a.source, a.backend = "file", "reference"
        log.info("testing %s", a.video)
    cfg = load_clipset(config)
    src = build_source(a.source, cfg, a.stream, seed=a.seed)
    be = build_backend(a.backend, cfg)
    if a.backend == "reference":
        from pl.reference import ReferenceBackend
        be = ReferenceBackend(cfg, lr=a.bg_lr, morph_iters=a.morph)

    p = TrackerParams.load(config)
    if a.high_area:
        p = replace(p, high_area=a.high_area)
    tracker, wires = Tracker(p), TripwireCounter(cfg.tripwires, p, stream=a.stream)
    zmap = ZoneMap(cfg.zones) if cfg.zones else None
    Hm = cfg.homography.get(a.stream)

    writer = None
    if a.out:
        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src.fps, (W, H))
    shown, counted = 0, []
    try:
        for frame in src.frames():
            if shown >= a.frames:
                break
            if be is not None and frame.image is not None:
                frame.result = be.process(frame.image, 0, frame.frame_id)

            canvas = (frame.image.copy() if frame.image is not None
                      else np.full((H, W, 3), 18, np.uint8))
            draw_zones(canvas, cfg, a.stream)

            warm = bool(int(frame.result["flags"]) & FLAG_BG_WARM)
            t = shown / max(src.fps, 1)
            tracks = tracker.update(prefilter(frame.blobs, p), t) if warm else []
            wires.update(tracks, t)
            for tr in tracker.just_deleted:
                wires.forget(tr.id)

            zone_by_track = {}
            if zmap is not None and Hm is not None:
                for tr in tracks:
                    zone_by_track[tr.id] = zmap.zone_of(*image_to_floor(Hm, tr.foot))
            draw(canvas, tracks, zone_by_track, wires.wires, wires.counts,
                 frame.frame_id, warm)

            if writer is not None:
                writer.write(canvas)
            else:
                cv2.imshow("tracks", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            counted.append(len(tracks))
            shown += 1
    finally:
        src.close()
        if writer is not None:
            writer.release()
            warm = counted[100:] or counted
            log.info("wrote %s (%d frames)", a.out, shown)
            log.info("people tracked: mean %.2f, max %d, seen in %d/%d frames",
                     sum(warm) / max(len(warm), 1), max(warm or [0]),
                     sum(1 for c in warm if c), len(warm))
        else:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
