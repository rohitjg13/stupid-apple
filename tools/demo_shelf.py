"""Visual demo script for Shelf & Inventory Intelligence.

Processes video file (e.g. ref_vid/*.mp4) or webcam with real-time HUD showing:
- Shelf ROIs colored by status (Green=Normal, Yellow=Low, Red=Empty)
- Live Item Inventory Count: Stock remaining & Items removed
- Customer pick interactions & hand motion
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

def run_shelf_demo(config_path: str, video_source: str, headless: bool = False, max_frames: int = 0, detector_model: str = "yolo"):
    cfg = load_clipset(config_path)
    pipe = ShelfPipeline(
        rois=cfg.rois,
        planogram=cfg.planogram,
        store_id=cfg.store_id,
        use_detector=True,
        detector_model=detector_model,
    )

    src = int(video_source) if video_source.isdigit() else video_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    t_start = time.time()

    print(f"Starting Shelf & Item Removal AI Demo on {video_source} (Press 'q' to quit, 'r' to simulate remove item)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            if not str(video_source).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_idx += 1
        t_now = t_start + frame_idx / fps

        frame_disp = cv2.resize(frame, (640, 480))
        gray_320 = cv2.resize(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), (320, 240))

        readings = pipe.process_frame(t=t_now, image=gray_320)
        compliance = pipe.get_compliance_score(readings)
        inv_summary = pipe.get_inventory_summary()

        if not headless:
            hud = frame_disp.copy()
            # Top header bar
            cv2.rectangle(hud, (0, 0), (640, 45), (20, 20, 20), -1)
            total_removed = sum(it["removed"] for it in inv_summary.values())
            cv2.putText(
                hud,
                f"SHELF AI | Compliance: {compliance}% | Items Removed: {total_removed} | Frame: {frame_idx}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                2,
            )

            # Draw ROIs
            for r in cfg.rois:
                r_id = r["id"]
                reading = readings.get(r_id)
                status = reading.status if reading else ShelfStatus.NORMAL
                it_stat = inv_summary.get(r_id, {})
                current_qty = it_stat.get("current", "-")
                removed_qty = it_stat.get("removed", 0)

                x, y, w, h = r["x"] * 2, r["y"] * 2, r["w"] * 2, r["h"] * 2

                if status == ShelfStatus.EMPTY or current_qty == 0:
                    color = (0, 0, 255)      # Red
                    lost_rev = pipe.get_lost_revenue_estimate(r_id, t_now)
                    label = f"{r_id}: EMPTY (0 rem) Lost: Rs.{lost_rev}"
                elif status == ShelfStatus.LOW or (isinstance(current_qty, int) and current_qty <= 3):
                    color = (0, 215, 255)    # Yellow/Amber
                    label = f"{r_id}: LOW ({current_qty} left, -{removed_qty})"
                else:
                    color = (0, 255, 0)      # Green
                    label = f"{r_id}: {current_qty} on shelf (-{removed_qty})"

                cv2.rectangle(hud, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(hud, (x, y - 22), (x + max(120, len(label) * 8), y), color, -1)
                cv2.putText(
                    hud,
                    label,
                    (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Edge AI Retail - Shelf Inventory Monitor", hud)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Interactive demo hotkey: simulate customer removing item from A1
                first_roi = cfg.rois[0]["id"]
                pipe.register_manual_removal(t=t_now, facing=first_roi, quantity=1)
                print(f"Customer removed 1 item from facing {first_roi}!")

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live shelf monitor demo")
    parser.add_argument("--config", default="config/shelf_ref", help="Clipset config directory")
    parser.add_argument("--video", default="ref_vid/VID-20260906-WA0004.mp4", help="Video path or camera index")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process")
    parser.add_argument("--detector", default="yolo", choices=["yolo", "fast_cv"], help="Detector backend (yolo or fast_cv)")
    args = parser.parse_args()

    run_shelf_demo(args.config, args.video, headless=args.headless, max_frames=args.frames, detector_model=args.detector)
