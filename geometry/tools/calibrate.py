"""Click 4 floor points on a frame, type their metres, save the homography.

    python -m geometry.tools.calibrate frame.png config/overhead_01/homography_overhead.npy

Left click a point, type its X and Y in metres in the terminal, repeat 4+ times,
press 'w' to write. A reprojected 1 m grid is drawn so you can see whether it
looks square before trusting it.
"""
from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np

from geometry.homography import floor_to_image

log = logging.getLogger("calibrate")


def draw_grid(img, H, extent_m=10.0, step_m=1.0):
    out = img.copy()
    n = int(extent_m / step_m) + 1
    for i in range(n):
        for a, b in (((i * step_m, 0.0), (i * step_m, extent_m)),
                     ((0.0, i * step_m), (extent_m, i * step_m))):
            try:
                p, q = floor_to_image(H, a), floor_to_image(H, b)
            except np.linalg.LinAlgError:
                return out
            if np.all(np.isfinite([p, q])):
                cv2.line(out, tuple(np.int32(p)), tuple(np.int32(q)), (0, 255, 0), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame")
    ap.add_argument("out")
    ap.add_argument("--extent", type=float, default=10.0, help="grid size in metres")
    a = ap.parse_args(argv)

    img = cv2.imread(a.frame)
    if img is None:
        raise SystemExit(f"cannot read {a.frame}")
    if img.shape[:2] != (480, 640):
        log.warning("frame is %s, not 480x640; points must be in full-res coords", img.shape[:2])

    img_pts, floor_pts, H = [], [], None

    def on_click(event, x, y, flags, _):
        nonlocal H
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(f"clicked ({x}, {y})")
        try:
            X = float(input("  floor X in metres: "))
            Y = float(input("  floor Y in metres: "))
        except ValueError:
            print("  not a number, point discarded")
            return
        img_pts.append([x, y])
        floor_pts.append([X, Y])
        if len(img_pts) >= 4:
            H, _ = cv2.findHomography(np.float32(img_pts), np.float32(floor_pts))
            err = np.linalg.norm(
                np.array([floor_to_image(H, p) for p in floor_pts]) - np.array(img_pts, float),
                axis=1)
            print(f"  {len(img_pts)} points, reprojection error "
                  f"mean {err.mean():.1f} px, max {err.max():.1f} px")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    print(__doc__)
    while True:
        view = img if H is None else draw_grid(img, H, a.extent)
        for p in img_pts:
            cv2.circle(view, tuple(p), 4, (0, 0, 255), -1)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("w") and H is not None:
            np.save(a.out, H)
            print(f"wrote {a.out}")
            break
        if k == 27:
            print("aborted, nothing written")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
