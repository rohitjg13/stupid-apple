"""Visual interactive demo script for Shelf & Inventory Intelligence.

Processes video file (e.g. ref_vid/*.mp4) or webcam with real-time HUD showing:
- Shelf ROIs colored by status (Green=Normal, Yellow=Low, Red=Empty)
- Live Item Inventory Count: Stock remaining & Items removed
- Customer pick interactions & hand motion detection
- Stockout alerts & estimated lost revenue
- Planogram compliance score
- Interactive Hotkeys:
    [R]      : Simulate customer removing 1 item from active facing
    [1]..[9] : Remove 1 item from specific facing
    [Tab]/[N]: Cycle selected facing
    [Space]  : Pause / Resume playback
    [Q]      : Quit
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

def run_shelf_demo(
    config_path: str,
    video_source: str,
    headless: bool = False,
    max_frames: int = 0,
    detector_model: str = "yolo",
    loop: bool = True,
):
    cfg = load_clipset(config_path)
    pipe = ShelfPipeline(
        rois=cfg.rois,
        planogram=cfg.planogram,
        store_id=cfg.store_id,
        use_detector=True,
        detector_model=detector_model,
    )

    src = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto aspect ratio
    if src_w > src_h:
        disp_w, disp_h = 800, 480
    else:
        disp_w, disp_h = 540, 960

    scale_x = disp_w / 320.0
    scale_y = disp_h / 240.0

    frame_idx = 0
    t_start = time.time()
    selected_roi_idx = 0
    total_rois = len(cfg.rois)

    toast_msg = None
    toast_until_frame = 0
    flash_facing = None

    paused = False
    last_frame = None

    print("\n" + "=" * 65)
    print("  EDGE AI RETAIL - SHELF & INVENTORY MONITOR")
    print("=" * 65)
    print("Interactive Controls:")
    print("  [R]       : Simulate customer removing 1 item from active facing")
    print("  [1]..[9]  : Remove 1 item directly from facing #")
    print("  [Tab]/[N] : Cycle active facing")
    print("  [Space]   : Pause / Resume playback")
    print("  [Q]       : Quit")
    print("=" * 65 + "\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if loop and not str(video_source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_idx += 1
            last_frame = frame
        else:
            frame = last_frame
            if frame is None:
                break

        t_now = t_start + frame_idx / fps

        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        gray_320 = cv2.resize(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), (320, 240))

        if not paused:
            readings = pipe.process_frame(t=t_now, image=gray_320)
        else:
            readings = {r["id"]: None for r in cfg.rois}

        compliance = pipe.get_compliance_score(readings)
        inv_summary = pipe.get_inventory_summary()
        active_picks = pipe.get_active_picks()

        if not headless:
            hud = frame_disp.copy()

            # 1. Top Header Bar
            cv2.rectangle(hud, (0, 0), (disp_w, 55), (15, 15, 20), -1)
            total_removed = sum(it.get("removed", 0) for it in inv_summary.values())
            total_val = sum(it.get("lost_or_taken_val", 0.0) for it in inv_summary.values())

            active_facing_id = cfg.rois[selected_roi_idx]["id"] if total_rois else "-"
            title_text = f"SHELF AI | Compliance: {compliance:.0f}% | Picked: {total_removed} (Rs.{total_val:.0f})"
            if paused:
                title_text += " [PAUSED]"

            cv2.putText(
                hud,
                title_text,
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                f"Selected Facing: [{active_facing_id}] (Press 'r' to remove item | Space to pause)",
                (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # 2. Draw ROIs
            for idx, r in enumerate(cfg.rois):
                r_id = r["id"]
                reading = readings.get(r_id)
                status = reading.status if reading else ShelfStatus.NORMAL
                it_stat = inv_summary.get(r_id, {})
                current_qty = it_stat.get("current", "-")
                initial_qty = it_stat.get("initial", 10)
                removed_qty = it_stat.get("removed", 0)
                item_name = it_stat.get("name", r_id)

                x = int(r["x"] * scale_x)
                y = int(r["y"] * scale_y)
                w = int(r["w"] * scale_x)
                h = int(r["h"] * scale_y)

                is_selected = (idx == selected_roi_idx)
                is_active_reach = any(p.facing == r_id for p in active_picks)

                if flash_facing == r_id and frame_idx <= toast_until_frame:
                    color = (0, 140, 255) # Bright orange flash
                    thickness = 3
                    label = f"[{idx+1}] {r_id}: ITEM REMOVED (-1) Stock:{current_qty}"
                elif is_active_reach:
                    color = (0, 255, 255) # Cyan reach
                    thickness = 3
                    label = f"[{idx+1}] {r_id}: REACHING IN..."
                elif status == ShelfStatus.EMPTY or current_qty == 0:
                    color = (0, 0, 255) # Red
                    thickness = 2
                    label = f"[{idx+1}] {r_id}: OUT OF STOCK (0 rem)"
                elif status == ShelfStatus.LOW or (isinstance(current_qty, int) and current_qty <= 3):
                    color = (0, 215, 255) # Yellow/Amber
                    thickness = 2
                    label = f"[{idx+1}] {r_id}: LOW ({current_qty}/{initial_qty} left)"
                else:
                    color = (0, 255, 0) # Green
                    thickness = 2
                    label = f"[{idx+1}] {r_id}: {current_qty}/{initial_qty} (-{removed_qty})"

                if is_selected:
                    # Draw a distinct selector ring around active ROI
                    cv2.rectangle(hud, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), 1)

                cv2.rectangle(hud, (x, y), (x + w, y + h), color, thickness)
                lbl_w = max(130, len(label) * 8)
                cv2.rectangle(hud, (x, y - 22), (x + lbl_w, y), color, -1)
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

            # 3. Toast Banner for removals
            if toast_msg and frame_idx <= toast_until_frame:
                cv2.rectangle(hud, (10, 65), (disp_w - 10, 100), (0, 215, 255), -1)
                cv2.putText(
                    hud,
                    toast_msg,
                    (18, 89),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 4. Bottom Controls Strip
            cv2.rectangle(hud, (0, disp_h - 30), (disp_w, disp_h), (20, 20, 20), -1)
            cv2.putText(
                hud,
                "[R]=Pick item  [1-9]=Pick facing  [Tab]=Cycle  [Space]=Pause  [Q]=Quit",
                (12, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Edge AI Retail - Shelf Inventory Monitor", hud)
            delay = 1 if paused else max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord('q'), ord('Q'), 27): # 'q' or ESC
                break
            elif key == ord(' '): # Spacebar: pause/resume
                paused = not paused
                print(f"[DEMO] Playback {'PAUSED' if paused else 'RESUMED'}.")
            elif key in (9, ord('n'), ord('N')): # Tab or 'n': cycle selected ROI
                selected_roi_idx = (selected_roi_idx + 1) % total_rois
                cur_id = cfg.rois[selected_roi_idx]["id"]
                print(f"[DEMO] Selected facing changed to: {cur_id}")
            elif key in (ord('r'), ord('R')): # 'r': remove 1 item from active facing
                target_facing = cfg.rois[selected_roi_idx]["id"]
                evs = pipe.register_manual_removal(t=t_now, facing=target_facing, quantity=1)
                st = pipe.get_inventory_summary().get(target_facing, {})
                toast_msg = f"CUSTOMER PICKED: 1x {st.get('name', target_facing)}! Stock: {st.get('current')}/{st.get('initial')}"
                toast_until_frame = frame_idx + int(fps * 2.5)
                flash_facing = target_facing
                print(f"[DEMO] Manual removal: 1x from {target_facing}! Current stock: {st.get('current')}")
            elif ord('1') <= key <= ord('9'): # Numbers 1..9
                num_idx = key - ord('1')
                if num_idx < total_rois:
                    target_facing = cfg.rois[num_idx]["id"]
                    selected_roi_idx = num_idx
                    pipe.register_manual_removal(t=t_now, facing=target_facing, quantity=1)
                    st = pipe.get_inventory_summary().get(target_facing, {})
                    toast_msg = f"CUSTOMER PICKED: 1x {st.get('name', target_facing)}! Stock: {st.get('current')}/{st.get('initial')}"
                    toast_until_frame = frame_idx + int(fps * 2.5)
                    flash_facing = target_facing
                    print(f"[DEMO] Hotkey {num_idx+1}: Removed 1x from {target_facing}! Stock: {st.get('current')}")

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"Demo complete. Processed {frame_idx} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live shelf monitor demo")
    parser.add_argument("--config", default="config/produce_shelf", help="Clipset config directory")
    parser.add_argument("--video", default="ref_vid/VID-20260906-WA0001.mp4", help="Video path or camera index")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process")
    parser.add_argument("--detector", default="yolo", choices=["yolo", "fast_cv"], help="Detector backend")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video")
    args = parser.parse_args()

    run_shelf_demo(
        args.config,
        args.video,
        headless=args.headless,
        max_frames=args.frames,
        detector_model=args.detector,
        loop=not args.no_loop,
    )
