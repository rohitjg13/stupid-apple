"""Visual demo script for Shelf & Inventory Intelligence.

Processes video file (e.g. ref_vid/*.mp4) or webcam with real-time HUD showing:
- Shelf ROIs colored by status (Green=Normal, Yellow=Low, Red=Empty)
- Live Fill & Confidence metrics
- Stockout alerts & estimated lost revenue
- Planogram compliance score
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.config import load_clipset
from shelf.pipeline import ShelfPipeline
from shelf.fill import ShelfStatus

def run_shelf_demo(config_path: str, video_source: str, headless: bool = False, max_frames: int = 0):
    cfg = load_clipset(config_path)
    pipe = ShelfPipeline(rois=cfg.rois, planogram=cfg.planogram, store_id=cfg.store_id)

    # Detect if video_source is integer (webcam)
    src = int(video_source) if video_source.isdigit() else video_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    t_start = time.time()

    print(f"Starting Shelf AI Demo on {video_source} (Press 'q' to quit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            # If video file, loop
            if not str(video_source).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_idx += 1
        t_now = t_start + frame_idx / fps

        # Resize to standard 640x480
        frame_disp = cv2.resize(frame, (640, 480))
        gray_320 = cv2.resize(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), (320, 240))

        # Process frame
        readings = pipe.process_frame(t=t_now, image=gray_320)
        compliance = pipe.get_compliance_score(readings)

        if not headless:
            # Draw HUD
            hud = frame_disp.copy()
            # Overlay header bar
            cv2.rectangle(hud, (0, 0), (640, 45), (20, 20, 20), -1)
            cv2.putText(
                hud,
                f"SHELF INTELLIGENCE | Compliance: {compliance}% | Frame: {frame_idx}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 200),
                2,
            )

            # Draw ROIs (scale coordinates from 320x240 to 640x480)
            for r in cfg.rois:
                r_id = r["id"]
                reading = readings.get(r_id)
                status = reading.status if reading else ShelfStatus.NORMAL
                fill = reading.smoothed_fill if reading else 0.0

                x, y, w, h = r["x"] * 2, r["y"] * 2, r["w"] * 2, r["h"] * 2

                if status == ShelfStatus.EMPTY:
                    color = (0, 0, 255)      # Red
                    lost_rev = pipe.get_lost_revenue_estimate(r_id, t_now)
                    label = f"{r_id}: EMPTY (Lost: Rs.{lost_rev})"
                elif status == ShelfStatus.LOW:
                    color = (0, 215, 255)    # Yellow/Amber
                    label = f"{r_id}: LOW ({int(fill)})"
                else:
                    color = (0, 255, 0)      # Green
                    label = f"{r_id}: OK ({int(fill)})"

                cv2.rectangle(hud, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(hud, (x, y - 22), (x + len(label) * 9, y), color, -1)
                cv2.putText(
                    hud,
                    label,
                    (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Edge AI Retail - Shelf Monitor", hud)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                break

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live shelf monitor demo")
    parser.add_argument("--config", default="config/sim", help="Clipset config directory")
    parser.add_argument("--video", default="ref_vid/VID-20260906-WA0004.mp4", help="Video path or camera index")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process")
    args = parser.parse_args()

    run_shelf_demo(args.config, args.video, headless=args.headless, max_frames=args.frames)
