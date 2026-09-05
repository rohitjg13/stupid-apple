"""Render annotated shelf inventory tracking video.

Processes video through ShelfPipeline with YOLO shopper detection,
edge texture fill monitoring, hand pick interaction detection,
and renders a rich retail analytics HUD overlay to an output MP4.
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

def render_video(
    config_path: str,
    video_path: str,
    output_path: str,
    target_width: int = 720,
    target_height: int = 1280,
    detector_model: str = "yolo",
    max_frames: int = 0,
):
    cfg = load_clipset(config_path)
    pipe = ShelfPipeline(
        rois=cfg.rois,
        planogram=cfg.planogram,
        store_id=cfg.store_id,
        use_detector=False,
    )

    yolo_model = YOLO('yolov8n.pt') if detector_model == 'yolo' else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine display size: maintain aspect ratio
    if src_w > src_h:
        disp_w, disp_h = 1280, 720
    else:
        disp_w, disp_h = target_width, target_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (disp_w, disp_h))

    scale_x = disp_w / 320.0
    scale_y = disp_h / 240.0

    print(f"Rendering '{video_path}' -> '{output_path}' ({disp_w}x{disp_h} @ {fps:.1f} fps)...")

    frame_idx = 0
    t_start = 0.0
    toast_message = None
    toast_until_frame = 0

    keyframe_snaps = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_now = t_start + frame_idx / fps

        disp_frame = cv2.resize(frame, (disp_w, disp_h))
        gray_320 = cv2.resize(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2GRAY), (320, 240))

        # Check inventory before processing
        inv_before = pipe.get_inventory_summary()

        readings = pipe.process_frame(t=t_now, image=gray_320)
        compliance = pipe.get_compliance_score(readings)
        inv_summary = pipe.get_inventory_summary()
        active_picks = pipe.get_active_picks()

        # Check if any item was removed on this frame
        for f_id, it_now in inv_summary.items():
            it_bef = inv_before.get(f_id, {})
            if it_now.get("removed", 0) > it_bef.get("removed", 0):
                toast_message = f"PICK CONFIRMED: 1x {it_now['name']} (-1) taken from {f_id}!"
                toast_until_frame = frame_idx + int(fps * 3.0)
                print(f"Frame {frame_idx:03d} ({t_now:.2f}s): {toast_message}")

        # Run YOLO on every 3rd frame for person tracking
        shopper_boxes = []
        if yolo_model is not None and frame_idx % 3 == 1:
            y_res = yolo_model(disp_frame, verbose=False, classes=[0])
            for r in y_res:
                for b in r.boxes:
                    shopper_boxes.append(b.xyxy[0].cpu().numpy().astype(int))
            setattr(pipe, '_last_shopper_boxes', shopper_boxes)
        else:
            shopper_boxes = getattr(pipe, '_last_shopper_boxes', [])

        hud = disp_frame.copy()

        # 1. Draw Shopper Bounding Box
        for sb in shopper_boxes:
            sx1, sy1, sx2, sy2 = sb
            cv2.rectangle(hud, (sx1, sy1), (sx2, sy2), (255, 180, 0), 2)
            cv2.rectangle(hud, (sx1, sy1 - 24), (sx1 + 170, sy1), (255, 180, 0), -1)
            cv2.putText(
                hud,
                "Shopper Detected",
                (sx1 + 4, sy1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Shelf ROIs & Inventory Counters
        for r in cfg.rois:
            r_id = r["id"]
            reading = readings.get(r_id)
            status = reading.status if reading else ShelfStatus.NORMAL
            it_stat = inv_summary.get(r_id, {})
            sku_name = it_stat.get("name", r_id)
            initial_qty = it_stat.get("initial", 10)
            current_qty = it_stat.get("current", initial_qty)
            removed_qty = it_stat.get("removed", 0)

            rx = int(r["x"] * scale_x)
            ry = int(r["y"] * scale_y)
            rw = int(r["w"] * scale_x)
            rh = int(r["h"] * scale_y)

            # Check if this ROI currently has an active hand reach
            is_active_reach = any(p.facing == r_id for p in active_picks)

            if is_active_reach:
                color = (0, 255, 255) # Bright Yellow reach highlight
                thickness = 3
                badge_text = f"{sku_name}: REACHING IN..."
            elif status == ShelfStatus.EMPTY or current_qty == 0:
                color = (0, 0, 255)
                thickness = 2
                badge_text = f"{sku_name}: OUT OF STOCK (0 left)"
            elif status == ShelfStatus.LOW or (isinstance(current_qty, int) and current_qty <= 3):
                color = (0, 165, 255)
                thickness = 2
                badge_text = f"{sku_name}: LOW ({current_qty}/{initial_qty})"
            else:
                color = (0, 230, 0)
                thickness = 2
                badge_text = f"{sku_name}: {current_qty}/{initial_qty} left (-{removed_qty})"

            cv2.rectangle(hud, (rx, ry), (rx + rw, ry + rh), color, thickness)

            badge_w = max(210, len(badge_text) * 9)
            cv2.rectangle(hud, (rx, ry - 26), (rx + badge_w, ry), color, -1)
            cv2.putText(
                hud,
                badge_text,
                (rx + 4, ry - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Top Status Header Overlay
        overlay_top = hud.copy()
        cv2.rectangle(overlay_top, (0, 0), (disp_w, 80), (15, 15, 18), -1)
        cv2.addWeighted(overlay_top, 0.82, hud, 0.18, 0, hud)

        cv2.putText(
            hud,
            "EDGE AI RETAIL | SHELF & ITEM REMOVAL TRACKER",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        total_items_removed = sum(it.get("removed", 0) for it in inv_summary.values())
        total_taken_val = sum(it.get("lost_or_taken_val", 0.0) for it in inv_summary.values())
        stat_line = f"Planogram: {compliance:.0f}%  |  Items Picked: {total_items_removed}  |  Value: Rs.{total_taken_val:.0f}  |  T: {t_now:.1f}s"
        cv2.putText(
            hud,
            stat_line,
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 240, 200),
            1,
            cv2.LINE_AA,
        )

        # 4. Active Reach / Toast Notification Banner
        if active_picks:
            reach_facing = active_picks[0].facing
            toast_text = f"HAND INTERACTION DETECTED AT {reach_facing}!"
            cv2.rectangle(hud, (15, 90), (disp_w - 15, 130), (0, 200, 255), -1)
            cv2.putText(
                hud,
                toast_text,
                (25, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        elif toast_message and frame_idx <= toast_until_frame:
            cv2.rectangle(hud, (15, 90), (disp_w - 15, 130), (0, 220, 100), -1)
            cv2.putText(
                hud,
                toast_message,
                (25, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 5. Bottom Live Inventory Ledger Strip
        overlay_bottom = hud.copy()
        cv2.rectangle(overlay_bottom, (0, disp_h - 110), (disp_w, disp_h), (15, 15, 18), -1)
        cv2.addWeighted(overlay_bottom, 0.85, hud, 0.15, 0, hud)

        cv2.putText(
            hud,
            "LIVE SHELF STOCK LEDGER:",
            (16, disp_h - 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        y_offset = disp_h - 60
        for i, (f_id, it) in enumerate(inv_summary.items()):
            col_x = 16 + (i % 2) * (disp_w // 2)
            col_y = y_offset + (i // 2) * 26
            badge = f"[{f_id}] {it['name'][:18]}: {it['current']}/{it['initial']} (Picked: {it['removed']})"
            c = (0, 255, 150) if it['removed'] > 0 else (220, 220, 220)
            cv2.putText(
                hud,
                badge,
                (col_x, col_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                c,
                1,
                cv2.LINE_AA,
            )

        out.write(hud)

        if frame_idx == 30:
            keyframe_snaps["keyframe_1_initial.jpg"] = hud
        elif frame_idx == 130:
            keyframe_snaps["keyframe_2_pick_action.jpg"] = hud
        elif frame_idx == 180:
            keyframe_snaps["keyframe_3_confirmed_removal.jpg"] = hud

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()
    print(f"Finished rendering {frame_idx} frames to '{output_path}'.")

    scratch_dir = ROOT / 'scratch'
    scratch_dir.mkdir(exist_ok=True)
    for fname, snap in keyframe_snaps.items():
        snap_path = scratch_dir / fname
        cv2.imwrite(str(snap_path), snap)
        print(f"Saved keyframe snapshot: {snap_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render annotated shelf video")
    parser.add_argument("--config", default="config/produce_shelf", help="Path to clipset config")
    parser.add_argument("--video", default="ref_vid/VID-20260906-WA0001.mp4", help="Input video path")
    parser.add_argument("--output", default="scratch/wa0001_annotated.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to render (0 for all)")
    args = parser.parse_args()

    render_video(
        config_path=args.config,
        video_path=args.video,
        output_path=args.output,
        max_frames=args.frames,
    )
