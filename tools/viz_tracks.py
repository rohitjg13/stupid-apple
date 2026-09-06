"""Run a clip through the shopper pipeline; write a per-video report folder.

    tools/try_video.sh vid/arcade1.mp4 --detector yolo          # the usual way
    python -m tools.viz_tracks --video clip.mp4 --report out/clip

The report folder answers the four shopper-analytics questions, one file each,
and is honest about which ones need a per-camera config:

    tracked.mp4        annotated video: boxes, ids, dwell labels
    heatmap.png        4. dwell heatmap over a still: light green where people
                          walked, yellow -> red where they stood (full red = 10 s
                          standing, or 5 s for two people on the same spot)
    dwell.csv          3. per person: seconds present, seconds standing still,
                          distance walked, which image region they dwelt in
    footfall.csv       2. per time bucket: people present, arrivals, departures
                          (by *store zone* needs zones.json -> --config)
    entries_exits.csv  1. crossings of the configured door (--config), else a
                          labelled proxy: tracks that appeared from / left via a
                          frame edge
    summary.txt/.json  the headline numbers

Layout-free (no --config) means no door, no zones, no floor calibration: none
of those can be inferred from pixels. This is the one file in the shopper work
that imports cv2.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tempfile
from collections import Counter, defaultdict
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
HEAT_CELL = 16                      # px per heatmap cell
EDGE_BAND = 24                      # px from the content edge that counts as "the edge"
PALETTE = [(66, 135, 245), (245, 176, 66), (66, 245, 129), (245, 66, 152),
           (245, 245, 66), (167, 66, 245), (66, 245, 245), (245, 111, 66)]
REGION_ROWS = ("top", "middle", "bottom")
REGION_COLS = ("left", "centre", "right")


def colour(track_id):
    return PALETTE[track_id % len(PALETTE)]


# --- geometry helpers (pure; tested) ------------------------------------------
def content_rect(src_w, src_h):
    """Where the letterboxed video actually is inside the 640x480 frame."""
    s = min(W / src_w, H / src_h)
    rw, rh = int(round(src_w * s)), int(round(src_h * s))
    x0, y0 = (W - rw) // 2, (H - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


def near_edge(u, v, rect, band=EDGE_BAND):
    """Is this foot point within `band` px of the visible frame's edge?

    Judged against the *content* rectangle, not the black letterbox bars --
    a 16:9 clip inside 640x480 has 60 px of bar top and bottom that nobody
    walks through.
    """
    x0, y0, x1, y1 = rect
    return u <= x0 + band or u >= x1 - band or v <= y0 + band or v >= y1 - band


def region_of(u, v, rect):
    """Which of nine image regions a foot point is in: 'bottom-left' etc."""
    x0, y0, x1, y1 = rect
    col = min(2, max(0, int((u - x0) / max((x1 - x0) / 3, 1))))
    row = min(2, max(0, int((v - y0) / max((y1 - y0) / 3, 1))))
    return f"{REGION_ROWS[row]}-{REGION_COLS[col]}"


def bucket_of(frame, fps, bucket_s):
    return int(frame // max(int(round(fps * bucket_s)), 1))


# --- drawing ------------------------------------------------------------------
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


def _impulse(shape):
    g = np.zeros(shape, dtype=np.float32)
    g[shape[0] // 2, shape[1] // 2] = 1.0
    return g


def draw_dwell_heat(canvas, visit, dwell, full):
    """Three tiers on one absolute scale, so nothing is red for no reason.

    * walked through  -> light green, faint
    * stood still     -> yellow shading to red as standing time accumulates
    * two on one spot -> accumulates twice as fast

    `full` is the standing count that means full red (seconds x fps). With a
    per-frame normalisation the first frame anyone stood still was already red;
    an absolute scale is what lets a spot visibly *get darker* over time.
    """
    if visit.max() <= 0:
        return canvas
    up = lambda g: cv2.GaussianBlur(
        cv2.resize(g.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR),
        (0, 0), HEAT_CELL * 0.75)
    # Normalise AFTER blurring, against what one perfectly-still person produces.
    # A real shopper's foot point twitches across two to four adjacent cells, so
    # no single cell ever holds their full standing time, and the blur then cuts a
    # lone cell's peak to ~40%: 15 s of standing rendered as faint green. Dividing
    # by the blurred impulse peak lets the twitching sum back together under the
    # blur, so 10 s means red whether the feet moved a few pixels or not.
    unit_peak = up(_impulse(visit.shape)).max()
    walked = np.clip(up((visit > 0).astype(np.float32)) / unit_peak, 0, 1)
    stood = np.clip(up(dwell) / (max(full, 1.0) * unit_peak), 0, 1)
    # JET: 0.5 = light green, 0.75 = yellow, 0.88 = bright red, 1.0 = MAROON.
    # Cap below 1.0 so full saturation is the brightest red, not a dark one that
    # reads as cooler than 90%.
    value = np.where(walked > 0.05, 0.5 + 0.38 * stood, 0.0)
    colour_map = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(walked * 0.22 + stood * 0.45, 0, 0.72)[..., None]   # walked = *light* green
    return (canvas * (1 - alpha) + colour_map * alpha).astype(np.uint8)


# --- clipsets -----------------------------------------------------------------
def bare_clipset(video, fps, base="config/sim"):
    """A clipset with nothing in it that would have to be true of *this* camera."""
    d = Path(tempfile.mkdtemp()) / "clipset"
    shutil.copytree(base, d)
    (d / "tripwires.json").write_text("[]")
    (d / "zones.json").write_text("[]")
    for f in d.glob("homography_*.npy"):
        f.unlink()
    (d / "tracker.yaml").unlink(missing_ok=True)     # sim tuning is for the sim
    _point_at_video(d, video, fps)
    return d


def _point_at_video(clipset, video, fps):
    y = clipset / "store.yaml"
    y.write_text(y.read_text().replace(
        "overhead: {source: sim, fps: 15, loop: true}",
        f"overhead: {{source: file, path: {Path(video).resolve()}, fps: {fps}, loop: false}}"))


# --- the report ---------------------------------------------------------------
class Report:
    """Everything the four analytics questions need, collected as frames go by."""

    def __init__(self, fps, rect, bucket_s, has_door, has_zones):
        self.fps, self.rect, self.bucket_s = fps, rect, bucket_s
        self.has_door, self.has_zones = has_door, has_zones
        self.people = {}                    # id -> record
        self.per_frame = []                 # people present, per frame
        shape = (H // HEAT_CELL, W // HEAT_CELL)
        self.heat_visit = np.zeros(shape, dtype=np.float32)   # frames anyone's feet were here
        self.heat_dwell = np.zeros(shape, dtype=np.float32)   # frames someone stood still here
        self.crossings = []                 # (frame, dir) from a configured door
        self.zone_frames = defaultdict(Counter)   # id -> zone -> frames
        self.last_image = None

    def frame(self, f, tracks, standing_of, zone_of, image):
        self.per_frame.append(len(tracks))
        if image is not None:
            self.last_image = image
        for tr in tracks:
            fu, fv = tr.foot
            rec = self.people.setdefault(tr.id, {
                "first": f, "born_foot": (fu, fv), "present": 0, "still": 0,
                "regions": Counter()})
            rec["last"], rec["last_foot"], rec["walked"] = f, (fu, fv), tr.travelled
            rec["present"] += 1
            still = standing_of(tr)
            rec["still"] += int(still)
            rec["regions"][region_of(fu, fv, self.rect)] += 1
            gx = int(min(max(fu, 0), W - 1)) // HEAT_CELL
            gy = int(min(max(fv, 0), H - 1)) // HEAT_CELL
            self.heat_visit[gy, gx] += 1.0
            if still:
                self.heat_dwell[gy, gx] += 1.0     # two people here -> twice as fast
            z = zone_of(tr)
            if z:
                self.zone_frames[tr.id][z] += 1

    # --- 3. dwell ----------------------------------------------------------
    def dwell_rows(self):
        rows = []
        for tid, r in sorted(self.people.items(), key=lambda kv: -kv[1]["present"]):
            top_region = r["regions"].most_common(1)[0][0] if r["regions"] else ""
            zones = self.zone_frames.get(tid)
            row = {"person": tid,
                   "first_seen_s": round(r["first"] / self.fps, 2),
                   "last_seen_s": round(r["last"] / self.fps, 2),
                   "present_s": round(r["present"] / self.fps, 2),
                   "standing_still_s": round(r["still"] / self.fps, 2),
                   "walked_px": round(r["walked"]),
                   "mostly_in_region": top_region}
            if self.has_zones:
                row["mostly_in_zone"] = zones.most_common(1)[0][0] if zones else ""
            rows.append(row)
        return rows

    # --- 2. footfall over time -------------------------------------------
    def footfall_rows(self):
        n = len(self.per_frame)
        if n == 0:
            return []
        nb = bucket_of(n - 1, self.fps, self.bucket_s) + 1
        present = [[] for _ in range(nb)]
        for f, c in enumerate(self.per_frame):
            present[bucket_of(f, self.fps, self.bucket_s)].append(c)
        arrivals, departures = Counter(), Counter()
        for r in self.people.values():
            arrivals[bucket_of(r["first"], self.fps, self.bucket_s)] += 1
            if r["last"] < n - 1:                       # still there at the end = not departed
                departures[bucket_of(r["last"], self.fps, self.bucket_s)] += 1
        rows = []
        for b in range(nb):
            rows.append({"from_s": round(b * self.bucket_s, 1),
                         "to_s": round(min((b + 1) * self.bucket_s, n / self.fps), 1),
                         "people_present_mean": round(float(np.mean(present[b])), 2) if present[b] else 0.0,
                         "people_present_max": int(max(present[b])) if present[b] else 0,
                         "arrivals": arrivals[b], "departures": departures[b]})
        return rows

    # --- 1. entries / exits ------------------------------------------------
    def entries_exits_rows(self):
        """Door crossings if a door is configured; else a labelled proxy."""
        n = len(self.per_frame)
        if self.has_door:
            per = defaultdict(Counter)
            for f, d in self.crossings:
                per[bucket_of(f, self.fps, self.bucket_s)][d] += 1
            nb = bucket_of(max(n - 1, 0), self.fps, self.bucket_s) + 1
            return [{"from_s": round(b * self.bucket_s, 1), "entered": per[b]["in"],
                     "exited": per[b]["out"], "basis": "configured door line"}
                    for b in range(nb)]
        rows = []
        min_frames = int(0.5 * self.fps)
        for tid, r in sorted(self.people.items()):
            if r["present"] < min_frames:
                continue
            entered = near_edge(*r["born_foot"], self.rect)
            left = near_edge(*r["last_foot"], self.rect) and r["last"] < n - 1
            rows.append({"person": tid, "entered_view_from_edge": int(entered),
                         "left_view_via_edge": int(left),
                         "basis": "frame edge (no door configured)"})
        return rows

    # --- summary -----------------------------------------------------------
    def summary(self, video, detector):
        n = len(self.per_frame)
        ee = self.entries_exits_rows()
        if self.has_door:
            entered, exited = sum(r["entered"] for r in ee), sum(r["exited"] for r in ee)
            basis = "configured door line"
        else:
            entered = sum(r["entered_view_from_edge"] for r in ee)
            exited = sum(r["left_view_via_edge"] for r in ee)
            basis = "frame edge -- proxy; configure a door line for real store entries"
        dw = self.dwell_rows()
        long_dwell = [r for r in dw if r["standing_still_s"] >= 2.0]
        hot = None
        if self.heat_dwell.max() > 0:
            gy, gx = np.unravel_index(int(np.argmax(self.heat_dwell)), self.heat_dwell.shape)
            hot = {"x": int(gx * HEAT_CELL + HEAT_CELL // 2), "y": int(gy * HEAT_CELL + HEAT_CELL // 2),
                   "region": region_of(gx * HEAT_CELL, gy * HEAT_CELL, self.rect),
                   "standing_s": round(float(self.heat_dwell.max()) / self.fps, 1)}
        return {
            "video": video, "detector": detector, "frames": n,
            "duration_s": round(n / self.fps, 1),
            "1_entries_exits": {"entered": entered, "exited": exited, "basis": basis},
            "2_footfall": {"people_present_mean": round(float(np.mean(self.per_frame)), 2) if n else 0,
                           "people_present_max": int(max(self.per_frame)) if n else 0,
                           "distinct_people_tracked": len(self.people),
                           "by_zone": "needs zones.json (--config)" if not self.has_zones else "see dwell.csv",
                           "by_time": "see footfall.csv"},
            "3_dwell": {"people_who_stood_still_2s_or_more": len(long_dwell),
                        "longest_still_s": max((r["standing_still_s"] for r in dw), default=0.0),
                        "near": "image regions (see dwell.csv); named displays need zones.json"},
            "4_heatmap": {"file": "heatmap.png", "hottest": hot,
                          "cells_walked": int((self.heat_visit > 0).sum()),
                          "legend": "green = walked through, yellow -> red = standing time"},
        }


def write_report(rep, out_dir, video, detector, cfg, stream, heat_full_s=10.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def csv_out(name, rows):
        if not rows:
            (out_dir / name).write_text("")
            return
        with open(out_dir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    csv_out("dwell.csv", rep.dwell_rows())
    csv_out("footfall.csv", rep.footfall_rows())
    csv_out("entries_exits.csv", rep.entries_exits_rows())

    # 4. the heatmap, once, over a still -- this is the afterwards artefact.
    base = rep.last_image.copy() if rep.last_image is not None else np.full((H, W, 3), 18, np.uint8)
    base = draw_dwell_heat(base, rep.heat_visit, rep.heat_dwell, rep.fps * heat_full_s)
    draw_zones(base, cfg, stream)
    peak = rep.heat_dwell.max() / rep.fps
    draw_banner(base, f"dwell heatmap over {len(rep.per_frame) / rep.fps:.0f} s   green = walked "
                      f"through   red = {heat_full_s:.0f} s standing   (peak here {peak:.1f} s)")
    cv2.imwrite(str(out_dir / "heatmap.png"), base)

    s = rep.summary(video, detector)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    ee, ff, dw, hm = s["1_entries_exits"], s["2_footfall"], s["3_dwell"], s["4_heatmap"]
    lines = [
        f"{Path(video).name}  --  {s['duration_s']} s, detector: {detector}",
        "",
        f"1. ENTRIES / EXITS   entered {ee['entered']}   exited {ee['exited']}",
        f"   basis: {ee['basis']}",
        f"2. FOOTFALL          mean {ff['people_present_mean']} people present, peak {ff['people_present_max']}, "
        f"{ff['distinct_people_tracked']} distinct people tracked",
        f"   over time: footfall.csv    by store zone: {ff['by_zone']}",
        f"3. DWELL             {dw['people_who_stood_still_2s_or_more']} people stood still >= 2 s, "
        f"longest {dw['longest_still_s']} s",
        f"   per person: dwell.csv    near what: {dw['near']}",
        f"4. HEATMAP           heatmap.png   green = walked, red = stood" +
        (f"   most standing: {hm['hottest']['region']} ({hm['hottest']['standing_s']} s)" if hm["hottest"] else ""),
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return out_dir


# --- main ---------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="any video file; layout-free unless --config")
    ap.add_argument("--config", default=None, help="clipset with door/zones/calibration")
    ap.add_argument("--report", default=None, help="report folder (default out/<video name>/)")
    ap.add_argument("--out", default=None, help="annotated mp4 path (default <report>/tracked.mp4)")
    ap.add_argument("--source", default="sim", choices=["sim", "file", "camera"])
    ap.add_argument("--backend", default="sim", choices=["sim", "reference", "yolo", "pl"])
    ap.add_argument("--detector", default="mog2", choices=["mog2", "yolo"],
                    help="mog2 = background subtraction (the FPGA chain); yolo = person "
                         "detector (Jetson). yolo sees people who stand still")
    ap.add_argument("--conf", type=float, default=0.3, help="yolo confidence threshold")
    ap.add_argument("--frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--fps", type=float, default=None, help="override the file's fps")
    ap.add_argument("--stream", default="overhead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bucket-s", type=float, default=10.0, help="footfall time bucket")
    ap.add_argument("--bg-lr", type=float, default=None,
                    help="MOG2 learning rate; default 0.005 scaled to the clip's fps")
    ap.add_argument("--morph", type=int, default=1, help="opening passes; 2 kills noise")
    ap.add_argument("--high-area", type=int, default=None,
                    help="confident-blob area; lower when people are small in frame")
    ap.add_argument("--heat-full-s", type=float, default=10.0,
                    help="seconds of standing still that render as full red (two people "
                         "on one spot get there in half the time)")
    ap.add_argument("--no-heat", action="store_true", help="leave the heatmap out of the video")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from main import build_backend, build_source

    layout_free, src_w, src_h = False, W, H
    if a.video:
        cap = cv2.VideoCapture(a.video)
        if a.fps is None:
            a.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or W
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or H
        cap.release()
        if a.config:
            config = Path(tempfile.mkdtemp()) / "clipset"
            shutil.copytree(a.config, config)
            _point_at_video(config, a.video, a.fps)
        else:
            config = bare_clipset(a.video, a.fps)
            layout_free = True
        a.source, a.backend = "file", "reference"
        if a.report is None:
            a.report = Path("out") / Path(a.video).stem
        log.info("testing %s at %.0f fps%s", a.video, a.fps,
                 "  (layout-free: no door, no zones, no calibration)" if layout_free else "")
    else:
        config = a.config or "config/sim"
        if a.report is None:
            a.report = Path("out") / "sim"
    if a.out is None:
        a.out = str(Path(a.report) / "tracked.mp4")
    Path(a.report).mkdir(parents=True, exist_ok=True)

    cfg = load_clipset(config)
    src = build_source(a.source, cfg, a.stream, seed=a.seed)
    if a.bg_lr is None:
        a.bg_lr = 0.005 * 15.0 / max(float(src.fps), 1.0)
    be = build_backend(a.backend, cfg)
    if a.backend == "reference":
        from pl.reference import ReferenceBackend
        be = ReferenceBackend(cfg, lr=a.bg_lr, morph_iters=a.morph)
        if a.detector == "yolo":
            from pl.yolo import YoloBackend
            be = YoloBackend(cfg, conf=a.conf, reference=be)
            log.info("detector: yolo (conf %.2f) -- no warm-up, whole-body boxes", a.conf)
        else:
            log.info("detector: mog2, background rate %.4f", a.bg_lr)

    p = TrackerParams.load(config)
    if a.high_area:
        p = replace(p, high_area=a.high_area)
    if a.detector == "yolo":
        # The furniture filter retires anything that never walked. It exists for
        # background-subtraction noise -- a trolley wheel the model keeps flagging.
        # A detector never emits shelf blobs, so under YOLO the only thing it can
        # retire is a real shopper who arrived before the clip started and stood
        # still: vid1's one man was tracked for 25 frames of 240, then "furniture".
        p = replace(p, furniture_age=10**9)
    tracker = Tracker(p)
    wires = TripwireCounter(cfg.tripwires, p, stream=a.stream)
    zmap = ZoneMap(cfg.zones) if cfg.zones else None
    Hm = cfg.homography.get(a.stream)
    fps = float(src.fps)
    rect = content_rect(src_w, src_h) if a.video else (0, 0, W, H)
    rep = Report(fps, rect, a.bucket_s, has_door=bool(wires.wires),
                 has_zones=zmap is not None and Hm is not None)

    def standing(tr):
        return tr.held > 0 or tr.seen_speed < p.static_speed_px

    def zone_of(tr):
        if zmap is None or Hm is None:
            return None
        return zmap.zone_of(*image_to_floor(Hm, tr.foot))

    heat_full = fps * a.heat_full_s
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
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
            for wire_id, direction, _ in wires.update(tracks, t):
                rep.crossings.append((shown, direction))
            for tr in tracker.just_deleted:
                wires.forget(tr.id)

            rep.frame(shown, tracks, standing, zone_of, frame.image)

            labels = {}
            for tr in tracks:
                r = rep.people[tr.id]
                label = f"#{tr.id} {r['present'] / fps:.1f}s"
                if r["still"] > fps:
                    label += f" still {r['still'] / fps:.0f}s"
                z = zone_of(tr)
                if z:
                    label += f" {z}"
                labels[tr.id] = label

            if not a.no_heat:
                canvas = draw_dwell_heat(canvas, rep.heat_visit, rep.heat_dwell, heat_full)
            draw_zones(canvas, cfg, a.stream)
            draw_wires(canvas, wires.wires)
            draw_tracks(canvas, tracks, labels)
            banner = f"frame {frame.frame_id}  people {len(tracks)}"
            if wires.wires:
                banner += f"  in {wires.counts['in']}  out {wires.counts['out']}"
            if not warm:
                banner += "   [background warming up]"
            draw_banner(canvas, banner)
            writer.write(canvas)
            shown += 1
    finally:
        src.close()
        writer.release()

    out_dir = write_report(rep, a.report, a.video or "sim", a.detector, cfg, a.stream,
                           heat_full_s=a.heat_full_s)
    log.info("wrote %s (%d frames)", a.out, shown)
    log.info("report in %s/", out_dir)
    log.info("%s", (out_dir / "summary.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
