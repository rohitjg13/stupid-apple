"""Write a clipset for footage nobody has calibrated yet.

`core/config.py` wants a folder of seven files before it will run anything. For
the demo flow -- upload a few videos, press next -- there is nobody to hand-author
them, so this generates a defensible default set from the videos themselves:

* homography: a flat scale, `floor_w x floor_h` metres across the 640x480 frame.
  Real overhead cameras have perspective; `geometry/tools/calibrate.py` replaces
  this with a measured one and everything downstream improves. Until then the
  metres are proportional, which is all zones and heatmaps need.
* zones: entrance / checkout / two aisles / promo band, laid out on that floor.
* tripwires: one door line, adjustable from the wizard (the two endpoints are
  the only thing the operator really has to point at).
* rois: a rows x cols grid of shelf facings at 320x240 PL scale.
* lanes: `n_lanes` queue columns of `cells` cells each, in the lower band.
* planogram: one generic SKU per facing, renamed later from the dashboard.

    python -m tools.autoconfig config/my_store --overhead a.mp4 b.mp4 --shelf s.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from pl.regs import MAX_LANES, MAX_ROIS, PL_H, PL_W

log = logging.getLogger(__name__)

FULL_W, FULL_H = 640, 480
FLOOR = (8.0, 6.0)              # metres covered by the frame, x then y
DEFAULT_DOOR_V = 400            # door line height in full-res px
ROWS = "ABCDEFGH"


def probe(path):
    """(fps, frames) from a container, with sane fallbacks."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {path}; try tools/transcode.sh")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return (round(fps, 3) if 0.1 < fps < 240 else 15.0), max(0, frames)


def homography(floor=FLOOR):
    """image px -> floor metres. Y is flipped: v=0 is the far end of the floor."""
    w, h = floor
    return np.array([[w / FULL_W, 0.0, 0.0],
                     [0.0, -h / FULL_H, h],
                     [0.0, 0.0, 1.0]])


def zones_from_rects(rects, floor=FLOOR):
    """Rectangles drawn on the 640x480 overhead frame -> floor polygons.

    The wizard works in pixels because that is what the operator can see; zones
    are stored in metres because that is what dwell, heatmaps and conversion
    are computed in. `homography()` is the bridge, so the two stay consistent
    even after somebody calibrates a real one.
    """
    from geometry.homography import image_to_floor
    H = homography(floor)
    out = []
    for i, r in enumerate(rects):
        x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0:
            raise ValueError(f"zone {r.get('id', i)} has no area: {w}x{h}")
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        poly = [[round(float(X), 3), round(float(Y), 3)]
                for X, Y in image_to_floor(H, corners)]
        out.append({"id": str(r.get("id") or f"zone_{i + 1}"), "polygon": poly})
    return out


def zones(floor=FLOOR):
    w, h = floor

    def box(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    return [
        {"id": "entrance", "polygon": box(0, 0, 0.375 * w, h / 3)},
        {"id": "checkout", "polygon": box(0.375 * w, 0, w, h / 3)},
        {"id": "aisle_a",  "polygon": box(0, h / 3, 0.44 * w, 0.75 * h)},
        {"id": "aisle_b",  "polygon": box(0.44 * w, h / 3, w, 0.75 * h)},
        {"id": "promo",    "polygon": box(0, 0.75 * h, w, h)},
    ]


IN_DIRS = ("up", "down", "left", "right")


def tripwires(door_line=None, in_dir=None):
    """`door_line` is [[u1,v1],[u2,v2]] in full-res px, straight from the wizard.

    `in_dir` is which side of that line counts as walking *into* the store, in
    image space ("up" is towards the top of the frame). Only the operator can
    know it -- a camera behind the till and a camera facing it disagree -- so
    the guess below is only a starting point the wizard can flip.
    """
    p1, p2 = door_line or ([0, DEFAULT_DOOR_V], [FULL_W - 1, DEFAULT_DOOR_V])
    vertical = abs(p2[0] - p1[0]) < abs(p2[1] - p1[1])
    if in_dir is not None and in_dir not in IN_DIRS:
        raise ValueError(f"in_dir must be one of {IN_DIRS}, got {in_dir!r}")
    return [{"id": "door", "stream": "overhead",
             "p1": [int(p1[0]), int(p1[1])], "p2": [int(p2[0]), int(p2[1])],
             "in_dir": in_dir or ("right" if vertical else "up")}]


def rois(rows=2, cols=3, margin=0.08, empty_below=60, low_below=110):
    """rows x cols facings over the shelf frame, at 320x240 PL scale."""
    if rows < 1 or cols < 1:                    # straight from the wizard's inputs
        raise ValueError(f"shelf grid must be at least 1x1, got {rows}x{cols}")
    if rows * cols > MAX_ROIS:
        raise ValueError(f"{rows}x{cols} facings exceeds the hardware table of {MAX_ROIS}")
    if rows > len(ROWS):
        raise ValueError(f"at most {len(ROWS)} shelf rows")
    mx, my = int(PL_W * margin), int(PL_H * margin)
    cw, ch = (PL_W - 2 * mx) // cols, (PL_H - 2 * my) // rows
    pad_x, pad_y = max(1, cw // 12), max(1, ch // 12)
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append({"id": f"{ROWS[r]}{c + 1}", "stream": "shelf",
                        "x": mx + c * cw + pad_x, "y": my + r * ch + pad_y,
                        "w": cw - 2 * pad_x, "h": ch - 2 * pad_y,
                        "empty_below": empty_below, "low_below": low_below})
    return out


def lanes(n_lanes=2, cells=8):
    """Queue cells in the lower band of the overhead frame, PL scale.

    Cell 0 is nearest the counter (bottom of the frame) so the fill order in
    `lane_occupancy` matches what the queue model assumes. Eight cells per lane
    fills the 16-entry hardware table and lets `naive_count` reach 6.4 people --
    with four it tops out at 3.2 and congestion can never be predicted.
    """
    if n_lanes < 1 or cells < 1:
        raise ValueError(f"lane grid must be at least 1x1, got {n_lanes}x{cells}")
    if n_lanes * cells > MAX_LANES:
        raise ValueError(f"{n_lanes}x{cells} cells exceeds the hardware table of {MAX_LANES}")
    band_w = PL_W // (n_lanes + 1)
    ch = 20
    out = []
    for lane in range(n_lanes):
        x = band_w // 2 + lane * band_w
        for cell in range(cells):
            out.append({"id": f"L{lane + 1}C{cell}", "lane": lane + 1, "cell": cell,
                        "x": x, "y": PL_H - 40 - cell * (ch + 2),
                        "w": min(band_w - 10, PL_W - x - 1), "h": ch})
    return out


def planogram(roi_list, unit_price=50, sales_per_hour=5):
    return [{"facing": r["id"], "sku": f"SKU-{r['id']}", "name": f"Facing {r['id']}",
             "unit_price": unit_price, "expected_sales_per_hour": sales_per_hour}
            for r in roi_list]


def make_clipset(dest, overhead=(), shelf=(), store_id="demo-01", shelf_grid=(2, 3),
                 door_line=None, floor=FLOOR, counters=2, target_wait_s=180,
                 lane_grid=(2, 8), zone_rects=None, door_dir=None):
    """Write a complete clipset folder and return its path.

    `overhead` and `shelf` are lists of video paths; several play back to back
    (sources/file.py reads `paths`). An empty `shelf` binds the shelf stream to
    the USB camera instead, which is the live demo.

    `zone_rects` are the zones the operator drew on the overhead frame, in
    full-res pixels; without them the default five-zone floor is used.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    overhead, shelf = [str(p) for p in overhead], [str(p) for p in shelf]

    streams = {}
    if overhead:
        fps, _ = probe(overhead[0])
        streams["overhead"] = {"source": "file", "paths": overhead, "fps": fps,
                               "loop": False}
    else:
        streams["overhead"] = {"source": "camera", "device": 0, "fps": 15}
    if shelf:
        fps, _ = probe(shelf[0])
        streams["shelf"] = {"source": "file", "paths": shelf, "fps": fps, "loop": False}
    else:
        streams["shelf"] = {"source": "camera", "device": "/dev/video0",
                            "fourcc": "YUYV", "fps": 5}

    store = {"store_id": store_id, "streams": streams,
             "checkout": {"counters": counters, "target_wait_s": target_wait_s},
             "pos": {"source": "stub"}, "cloud": {"enabled": False, "url": None}}

    roi_list = rois(*shelf_grid)
    (dest / "store.yaml").write_text(yaml.safe_dump(store, sort_keys=False))
    _write(dest / "zones.json",
           zones_from_rects(zone_rects, floor) if zone_rects else zones(floor))
    _write(dest / "tripwires.json", tripwires(door_line, door_dir))
    _write(dest / "rois.json", roi_list)
    _write(dest / "lanes.json", lanes(*lane_grid))
    _write(dest / "planogram.json", planogram(roi_list))
    np.save(dest / "homography_overhead.npy", homography(floor))
    (dest / "NOTES.md").write_text(
        f"# {store_id} — generated by tools/autoconfig.py\n\n"
        f"Floor assumed {floor[0]} m x {floor[1]} m across the frame, flat scale, no\n"
        f"perspective. Zones, the door line and the {shelf_grid[0]}x{shelf_grid[1]} shelf grid are defaults;\n"
        "run geometry/tools/calibrate.py for a measured homography and edit the\n"
        "JSON by hand for real facings.\n")
    log.info("clipset written", extra={"path": str(dest), "rois": len(roi_list)})
    return dest


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest")
    ap.add_argument("--overhead", nargs="*", default=[])
    ap.add_argument("--shelf", nargs="*", default=[])
    ap.add_argument("--store-id", default="demo-01")
    ap.add_argument("--shelf-rows", type=int, default=2)
    ap.add_argument("--shelf-cols", type=int, default=3)
    ap.add_argument("--counters", type=int, default=2)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(make_clipset(a.dest, a.overhead, a.shelf, a.store_id,
                       (a.shelf_rows, a.shelf_cols), counters=a.counters))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
