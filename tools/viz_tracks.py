"""Draw tracks over the video. The fastest way to see whether it works.

    python -m tools.viz_tracks --video clip.mp4 --out out.mp4     # any clip, no config
    python -m tools.viz_tracks --source sim --out sim.mp4          # the simulator
    python -m tools.viz_tracks --config config/overhead_01 --video clip.mp4

With --video and no --config it runs **layout-free**: no door line, no zones, no
floor calibration, because none of those can be inferred from pixels and
painting the simulator's floor plan over real footage only misleads. What it
measures instead needs no layout at all:

* dwell -- how long each person was present, and how long they stood still,
  which is what "browsing near products" looks like to a camera;
* a heatmap of where feet were, in pixels, painted over the actual video.

This is the one file in the shopper work that imports cv2. shopper/ itself must
stay importable on a board with no display stack.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

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
HEAT_CELL = 16                      # px; heatmap resolution in layout-free mode
PALETTE = [(66, 135, 245), (245, 176, 66), (66, 245, 129), (245, 66, 152),
           (245, 245, 66), (167, 66, 245), (66, 245, 245), (245, 111, 66)]


def colour(track_id):
    return PALETTE[track_id % len(PALETTE)]


# --- drawing ----------------------------------------------------------------
def draw_tracks(canvas, tracks, label_by_track):
    for tr in tracks:
        x, y, w, h = (int(v) for v in tr.box)
        c = colour(tr.id)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), c, 2)
        fu, fv = (int(v) for v in tr.foot)
        cv2.circle(canvas, (fu, fv), 4, c, -1)
        cv2.putText(canvas, label_by_track.get(tr.id, f"#{tr.id}"), (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)


def draw_banner(canvas, text):
    cv2.rectangle(canvas, (0, 0), (W, 24), (24, 24, 24), -1)
    cv2.putText(canvas, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)


def draw_wires(canvas, wires):
    for wire in wires:
        p1, p2 = tuple(map(int, wire["p1"])), tuple(map(int, wire["p2"]))
        cv2.line(canvas, p1, p2, (0, 0, 255), 2)
        cv2.putText(canvas, wire["id"], (p1[0], p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)


def draw_zones(canvas, cfg, stream):
    """Zone outlines projected back into the image. Nothing without a homography."""
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


def draw_heat(canvas, grid):
    """Blend the foot-point heatmap over the frame. Hot = people stood here."""
    if grid.max() <= 0:
        return canvas
    big = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), HEAT_CELL / 2)
    # Normalise *after* blurring, or the blur flattens the peak and the hottest
    # spot renders as a faint tint instead of red.
    norm = (big / big.max()) ** 0.5             # sqrt so a few hot cells don't hide the rest
    heat = (norm * 255).astype(np.uint8)
    colour_map = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    alpha = (norm * 0.7)[..., None]
    return (canvas * (1 - alpha) + colour_map * alpha).astype(np.uint8)


# --- layout-free clipset -----------------------------------------------------
def bare_clipset(video, fps, base="config/sim"):
    """A clipset with nothing in it that would have to be true of *this* camera.

    Copies the sim config for the parts the loader insists on (store id, shelf
    ROIs it never uses here), then empties the tripwires and zones and drops the
    homography. The pipeline treats each of those as "not configured".
    """
    d = Path(tempfile.mkdtemp()) / "clipset"
    shutil.copytree(base, d)
    (d / "tripwires.json").write_text("[]")
    (d / "zones.json").write_text("[]")
    for f in d.glob("homography_*.npy"):
        f.unlink()
    # The sim's tracker.yaml is tuned for the sim -- among other things it turns
    # the furniture filter off, because rendered rectangles have no furniture
    # noise. Real footage does. Layout-free means the real-footage defaults.
    (d / "tracker.yaml").unlink(missing_ok=True)
    _point_at_video(d, video, fps)
    return d


def _point_at_video(clipset, video, fps):
    y = clipset / "store.yaml"
    y.write_text(y.read_text().replace(
        "overhead: {source: sim, fps: 15, loop: true}",
        f"overhead: {{source: file, path: {Path(video).resolve()}, fps: {fps}, loop: false}}"))


# --- main ---------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="any video file; layout-free unless --config")
    ap.add_argument("--config", default=None, help="clipset with door/zones/calibration")
    ap.add_argument("--source", default="sim", choices=["sim", "file", "camera"])
    ap.add_argument("--backend", default="sim", choices=["sim", "reference", "pl"])
    ap.add_argument("--frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--fps", type=float, default=None, help="override the file's fps")
    ap.add_argument("--stream", default="overhead")
    ap.add_argument("--out", default=None, help="write an mp4 instead of a window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-lr", type=float, default=None,
                    help="MOG2 learning rate; lower keeps still people longer. "
                         "Default: the 0.005 register default scaled to the clip's "
                         "fps, because the rate is per frame and 0.005 was tuned "
                         "at 15 fps -- a 30 fps clip would otherwise absorb people "
                         "twice as fast")
    ap.add_argument("--morph", type=int, default=1, help="opening passes; 2 kills noise")
    ap.add_argument("--high-area", type=int, default=None,
                    help="confident-blob area; lower when people are small in frame")
    ap.add_argument("--no-heat", action="store_true", help="don't paint the heatmap")
    ap.add_argument("--detector", default="mog2", choices=["mog2", "yolo"],
                    help="mog2 = background subtraction (the FPGA chain); yolo = "
                         "person detector (Jetson). yolo sees people who stand still")
    ap.add_argument("--conf", type=float, default=0.3, help="yolo confidence threshold")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from main import build_backend, build_source

    layout_free = False
    if a.video:
        if a.fps is None:
            cap = cv2.VideoCapture(a.video)
            a.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
        if a.config:
            config = Path(tempfile.mkdtemp()) / "clipset"
            shutil.copytree(a.config, config)
            _point_at_video(config, a.video, a.fps)
        else:
            config = bare_clipset(a.video, a.fps)
            layout_free = True
        a.source, a.backend = "file", "reference"
        log.info("testing %s at %.0f fps%s", a.video, a.fps,
                 "  (layout-free: no door, no zones, no calibration)" if layout_free else "")
    else:
        config = a.config or "config/sim"

    cfg = load_clipset(config)
    src = build_source(a.source, cfg, a.stream, seed=a.seed)
    if a.bg_lr is None:
        a.bg_lr = 0.005 * 15.0 / max(float(src.fps), 1.0)
        if a.video:
            log.info("background rate %.4f (0.005 at 15 fps, scaled to %.0f fps)",
                     a.bg_lr, src.fps)
    be = build_backend(a.backend, cfg)
    if a.backend == "reference":
        from pl.reference import ReferenceBackend
        be = ReferenceBackend(cfg, lr=a.bg_lr, morph_iters=a.morph)
        if a.detector == "yolo":
            from pl.yolo import YoloBackend
            be = YoloBackend(cfg, conf=a.conf, reference=be)
            log.info("detector: yolo (conf %.2f) -- no warm-up, whole-body boxes", a.conf)

    p = TrackerParams.load(config)
    if a.high_area:
        p = replace(p, high_area=a.high_area)
    tracker = Tracker(p)
    wires = TripwireCounter(cfg.tripwires, p, stream=a.stream)
    zmap = ZoneMap(cfg.zones) if cfg.zones else None
    Hm = cfg.homography.get(a.stream)
    fps = float(src.fps)

    heat = np.zeros((H // HEAT_CELL, W // HEAT_CELL), dtype=np.float32)
    present = {}                 # track id -> frames seen
    still = {}                   # track id -> frames standing still
    moved = {}                   # track id -> furthest px from where it appeared
    counted = []
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if a.out else None

    shown = 0
    try:
        for frame in src.frames():
            if a.frames and shown >= a.frames:
                break
            if be is not None and frame.image is not None:
                frame.result = be.process(frame.image, 0, frame.frame_id)

            canvas = (frame.image.copy() if frame.image is not None
                      else np.full((H, W, 3), 18, np.uint8))
            warm = bool(int(frame.result["flags"]) & FLAG_BG_WARM)
            t = shown / fps
            tracks = tracker.update(prefilter(frame.blobs, p), t) if warm else []
            wires.update(tracks, t)
            for tr in tracker.just_deleted:
                wires.forget(tr.id)

            labels = {}
            for tr in tracks:
                present[tr.id] = present.get(tr.id, 0) + 1
                moved[tr.id] = tr.travelled
                standing = tr.held > 0 or tr.seen_speed < p.static_speed_px
                if standing:
                    still[tr.id] = still.get(tr.id, 0) + 1
                fu, fv = tr.foot
                gx, gy = int(min(max(fu, 0), W - 1)) // HEAT_CELL, int(min(max(fv, 0), H - 1)) // HEAT_CELL
                heat[gy, gx] += 1.0
                label = f"#{tr.id} {present[tr.id] / fps:.1f}s"
                if still.get(tr.id, 0) > fps:                     # over a second still
                    label += f" still {still[tr.id] / fps:.0f}s"
                if zmap is not None and Hm is not None:
                    zone = zmap.zone_of(*image_to_floor(Hm, tr.foot))
                    if zone:
                        label += f" {zone}"
                labels[tr.id] = label

            if not a.no_heat:
                canvas = draw_heat(canvas, heat)
            draw_zones(canvas, cfg, a.stream)
            draw_wires(canvas, wires.wires)
            draw_tracks(canvas, tracks, labels)
            banner = f"frame {frame.frame_id}  people {len(tracks)}"
            if wires.wires:
                banner += f"  in {wires.counts['in']}  out {wires.counts['out']}"
            if not warm:
                banner += "   [background warming up]"
            draw_banner(canvas, banner)

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
        else:
            cv2.destroyAllWindows()

    # --- report -----------------------------------------------------------
    warm_counts = (counted[100:] or counted) if a.detector == "mog2" else counted
    if a.out:
        log.info("wrote %s (%d frames)", a.out, shown)
    log.info("people tracked: mean %.2f, max %d, seen in %d/%d frames",
             sum(warm_counts) / max(len(warm_counts), 1), max(warm_counts or [0]),
             sum(1 for c in warm_counts if c), len(warm_counts))
    if present:
        log.info("dwell per person (id: seconds present, seconds standing still, "
                 "how far they walked to get there):")
        for tid in sorted(present, key=present.get, reverse=True)[:12]:
            log.info("  #%-3d %5.1f s present   %5.1f s still   walked %3.0f px",
                     tid, present[tid] / fps, still.get(tid, 0) / fps, moved.get(tid, 0))
    if heat.max() > 0:
        hot = np.unravel_index(int(np.argmax(heat)), heat.shape)
        log.info("hottest spot: around pixel (%d, %d), %d foot-samples",
                 hot[1] * HEAT_CELL + HEAT_CELL // 2, hot[0] * HEAT_CELL + HEAT_CELL // 2,
                 int(heat.max()))
    if a.out:
        summary = {"video": a.video, "frames": shown, "layout_free": layout_free,
                   "dwell_s": {str(k): round(v / fps, 2) for k, v in present.items()},
                   "still_s": {str(k): round(v / fps, 2) for k, v in still.items()}}
        Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
