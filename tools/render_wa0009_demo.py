"""Render object tracking and inventory delta (+/-) video for WA0009.

Features:
- Shopper person tracking (YOLOv8n @ imgsz=320)
- Shopping basket tracking (HSV color & contour)
- Hand reach / interaction tracking on shelf ROIs
- Real-time Inventory Delta (+ / -) starting from 0:
    Delta = 0 at start (t=0..47s)
    Delta = -1 after first pick (t=47.7s)
    Delta = -2 after second pick (t=94.9s)
- Dynamic on-screen HUD with Delta ticker, toast banners, and live stock ledger
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
from ultralytics import YOLO

from core.config import load_clipset
from shelf.pipeline import ShelfPipeline
from shelf.fill import ShelfStatus

def track_orange_basket(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 120, 120], dtype=np.uint8)
    upper_orange = np.array([22, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 800 < area < 25000:
            x, y, w, h = cv2.boundingRect(c)
            if y > 250 and 0.5 < w / float(h) < 2.2:
                if area > max_area:
                    max_area = area
                    best_box = (x, y, w, h)
    return best_box

def render_wa0009_demo(
    config_path: str = "config/overhead_aisle",
    video_path: str = "ref_vid/VID-20260906-WA0009.mp4",
    output_path: str = "scratch/wa0009_demo.mp4",
    max_frames: int = 0,
    stride: int = 2,
):
    cfg = load_clipset(config_path)
    pipe = ShelfPipeline(
        rois=cfg.rois,
        planogram=cfg.planogram,
        store_id=cfg.store_id,
        use_detector=False,
    )
    # Set calibrated cooldown for overhead camera
    pipe.pick_detector.cooldown_seconds = 30.0
    pipe.pick_detector.min_motion_ratio = 0.04
    pipe.pick_detector.motion_threshold = 18.0

    yolo_model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w, out_h = 920, 680
    playback_fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, playback_fps, (out_w, out_h))

    scale_x = out_w / 320.0
    scale_y = out_h / 240.0

    print(f"Rendering '{video_path}' -> '{output_path}' ({out_w}x{out_h} @ {playback_fps:.1f} fps, stride={stride}, frames={total_frames})...")

    frame_idx = 0
    toast_msg = None
    toast_until_frame = 0
    toast_color = (0, 220, 100)

    keyframe_snaps = {}
    cached_person_box = None

    # Track inventory delta (+ / -) starting at 0
    net_inventory_delta = 0
    total_removed = 0
    total_added = 0
    facing_deltas = {r["id"]: 0 for r in cfg.rois}

    t_render_start = time.time()
    frames_written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t_now = frame_idx / fps

        # Shelf pipeline processing runs on every frame to ensure exact event capture
        gray_320 = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
        readings = pipe.process_frame(t=t_now, image=gray_320)
        inv_summary = pipe.get_inventory_summary()
        active_picks = pipe.get_active_picks()

        # Check for newly committed removals
        for f_id, it in inv_summary.items():
            expected_removed = -facing_deltas[f_id]
            actual_removed = it["removed"]
            if actual_removed > expected_removed:
                diff = actual_removed - expected_removed
                facing_deltas[f_id] -= diff
                total_removed += diff
                net_inventory_delta = total_added - total_removed
                toast_msg = f"INVENTORY DELTA: -{diff} | 1x {it['name']} REMOVED (Net Delta: {net_inventory_delta:+d})"
                toast_until_frame = frame_idx + int(fps * 3.5)
                toast_color = (0, 220, 100)
                print(f"Frame {frame_idx:04d} ({t_now:.1f}s): {toast_msg}")

        # Skip rendering intermediate frames if stride > 1
        if stride > 1 and (frame_idx % stride != 0):
            continue

        # Fast YOLO Person Detection (every 8th frame on 320x240)
        if frame_idx % 8 == 0 or cached_person_box is None:
            small_yolo = cv2.resize(frame, (320, 240))
            res = yolo_model(small_yolo, verbose=False, imgsz=320, classes=[0])
            best_box = None
            max_area = 0
            for r in res:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                    bx1 = int(bx1 * (out_w / 320.0))
                    by1 = int(by1 * (out_h / 240.0))
                    bx2 = int(bx2 * (out_w / 320.0))
                    by2 = int(by2 * (out_h / 240.0))
                    if by2 > 260:
                        area = (bx2 - bx1) * (by2 - by1)
                        if area > max_area:
                            max_area = area
                            best_box = (bx1, by1, bx2, by2)
            cached_person_box = best_box

        # Basket Tracking (Orange color blob)
        basket_box = track_orange_basket(frame)

        # Render HUD
        hud = frame.copy()

        # 1. Draw Shelf ROIs with Individual Deltas (+/-)
        for r in cfg.rois:
            r_id = r["id"]
            reading = readings.get(r_id)
            status = reading.status if reading else ShelfStatus.NORMAL
            it_stat = inv_summary.get(r_id, {})
            name = it_stat.get("name", r_id)
            curr = it_stat.get("current", 10)
            init = it_stat.get("initial", 10)
            f_delta = facing_deltas.get(r_id, 0)

            rx = int(r["x"] * scale_x)
            ry = int(r["y"] * scale_y)
            rw = int(r["w"] * scale_x)
            rh = int(r["h"] * scale_y)

            is_active_reach = any(p.facing == r_id for p in active_picks)

            if is_active_reach:
                color = (0, 255, 255) # Bright Yellow reach highlight
                thickness = 3
                badge_text = f"[{r_id}] DELTA: {f_delta:+d} (REACH)"
            elif f_delta < 0:
                color = (0, 200, 255) # Amber: item removed
                thickness = 2
                badge_text = f"[{r_id}] DELTA: {f_delta:+d} ({curr}/{init})"
            else:
                color = (0, 220, 0) # Green
                thickness = 2
                badge_text = f"[{r_id}] DELTA: {f_delta:+d} ({curr}/{init})"

            cv2.rectangle(hud, (rx, ry), (rx + rw, ry + rh), color, thickness)
            
            lbl_w = len(badge_text) * 8 + 14
            bx = min(rx, out_w - lbl_w - 5)
            by = max(24, ry)
            cv2.rectangle(hud, (bx, by - 20), (bx + lbl_w, by), color, -1)
            cv2.putText(
                hud,
                badge_text,
                (bx + 4, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Object Tracking: Shopper & Basket
        if cached_person_box is not None:
            px1, py1, px2, py2 = cached_person_box
            cv2.rectangle(hud, (px1, py1), (px2, py2), (255, 180, 0), 2)
            cv2.rectangle(hud, (px1, py1 - 22), (px1 + 180, py1), (255, 180, 0), -1)
            cv2.putText(
                hud,
                "Shopper #1 (Active)",
                (px1 + 4, py1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if basket_box is not None:
            bx, by, bw, bh = basket_box
            cv2.rectangle(hud, (bx, by), (bx + bw, by + bh), (0, 140, 255), 2)
            cv2.rectangle(hud, (bx, by - 18), (bx + 130, by), (0, 140, 255), -1)
            cv2.putText(
                hud,
                "Shopping Basket",
                (bx + 4, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Top Status Header Overlay
        overlay_top = hud.copy()
        cv2.rectangle(overlay_top, (0, 0), (out_w, 75), (15, 15, 20), -1)
        cv2.addWeighted(overlay_top, 0.82, hud, 0.18, 0, hud)

        cv2.putText(
            hud,
            "EDGE AI RETAIL | OBJECT TRACKING & INVENTORY DELTA",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Big Highlighted Delta Badge (+ / - starting from 0)
        delta_color = (0, 255, 200) if net_inventory_delta == 0 else (0, 215, 255)
        delta_display = f"NET INVENTORY DELTA: {net_inventory_delta:+d}  (-{total_removed} removed, +{total_added} added)"
        cv2.putText(
            hud,
            delta_display,
            (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            delta_color,
            2,
            cv2.LINE_AA,
        )

        total_taken_val = sum(it.get("lost_or_taken_val", 0.0) for it in inv_summary.values())
        time_info = f"Value: Rs.{total_taken_val:.0f}  |  Time: {t_now:.1f}s"
        cv2.putText(
            hud,
            time_info,
            (out_w - 240, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # 4. Toast Banner for Reach / Delta Events
        if active_picks:
            reach_facing = active_picks[0].facing
            toast_text = f"HAND INTERACTION DETECTED: Reaching into {reach_facing}!"
            cv2.rectangle(hud, (15, 85), (out_w - 15, 120), (0, 220, 255), -1)
            cv2.putText(
                hud,
                toast_text,
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        elif toast_msg and frame_idx <= toast_until_frame:
            cv2.rectangle(hud, (15, 85), (out_w - 15, 120), toast_color, -1)
            cv2.putText(
                hud,
                toast_msg,
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 5. Bottom Live Inventory Ledger Strip
        overlay_bottom = hud.copy()
        cv2.rectangle(overlay_bottom, (0, out_h - 75), (out_w, out_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay_bottom, 0.85, hud, 0.15, 0, hud)

        cv2.putText(
            hud,
            "LIVE SHELF INVENTORY & DELTA LEDGER (INITIAL: 0 -> UPDATES ON REMOVAL/RESTOCK):",
            (16, out_h - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        y_offset = out_h - 26
        col_w = out_w // len(cfg.rois)
        for i, (f_id, it) in enumerate(inv_summary.items()):
            col_x = 16 + i * col_w
            f_d = facing_deltas.get(f_id, 0)
            badge = f"[{f_id}] {it['current']}/{it['initial']} (Delta: {f_d:+d})"
            c = (0, 255, 150) if f_d < 0 else (220, 220, 220)
            cv2.putText(
                hud,
                badge,
                (col_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                c,
                1,
                cv2.LINE_AA,
            )

        out.write(hud)
        frames_written += 1

        # Capture keyframe snapshots
        if frame_idx == 100:
            keyframe_snaps["wa0009_keyframe_1_delta_zero.jpg"] = hud
        elif frame_idx == 1500:
            keyframe_snaps["wa0009_keyframe_2_delta_minus_one.jpg"] = hud
        elif frame_idx == 2900:
            keyframe_snaps["wa0009_keyframe_3_delta_minus_two.jpg"] = hud

        if frame_idx % 400 == 0:
            elapsed = time.time() - t_render_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
            print(f"Processed {frame_idx}/{total_frames} frames ({fps_proc:.1f} fps, ETA: {eta:.0f}s)...")

    cap.release()
    out.release()
    print(f"Finished rendering {frames_written} frames to '{output_path}'.")

    scratch_dir = ROOT / 'scratch'
    scratch_dir.mkdir(exist_ok=True)
    for fname, snap in keyframe_snaps.items():
        snap_path = scratch_dir / fname
        cv2.imwrite(str(snap_path), snap)
        print(f"Saved keyframe snapshot: {snap_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render WA0009 Object Tracking and Inventory Delta Demo")
    parser.add_argument("--config", default="config/overhead_aisle", help="Path to config")
    parser.add_argument("--video", default="ref_vid/VID-20260906-WA0009.mp4", help="Video path")
    parser.add_argument("--output", default="scratch/wa0009_demo.mp4", help="Output path")
    parser.add_argument("--frames", type=int, default=0, help="Max frames (0 for all)")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride")
    args = parser.parse_args()

    render_wa0009_demo(
        config_path=args.config,
        video_path=args.video,
        output_path=args.output,
        max_frames=args.frames,
        stride=args.stride,
    )