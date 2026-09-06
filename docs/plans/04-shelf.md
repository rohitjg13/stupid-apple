# Plan 4 — Inventory & Shelf Intelligence + Live Demo (Pustak)

Read docs/SHARED.md first. You own the **Inventory Monitoring & Shelf Intelligence** module (shelf/), the shelf planogram compliance logic, out-of-stock and low-stock detection, and the live camera demo.

## Objectives
1. **Shelf ROI & Fill Monitoring (shelf/fill.py)**:
   - Compute texture / edge density and fill metrics per ROI from roi_fill in FrameResult (or direct image fallback).
   - Classify ROI status into NORMAL, LOW, and EMPTY based on calibrated thresholds (empty_below, low_below).
   - Temporal smoothing to prevent flicker from hand occlusions during product picking.

2. **Stockout & Low-Stock State Tracking (shelf/stockout.py)**:
   - Track state machines per shelf facing.
   - Detect stockout transitions and emit stockout_start and stockout_end events.
   - Calculate out-of-stock duration, loss rate (units/hr), and estimated lost revenue based on planogram.json.

3. **Planogram Compliance (shelf/planogram.py)**:
   - Monitor expected vs observed SKUs across all active facings.
   - Emit planogram_violation when products are empty, misplaced, or unexpected.
   - Track planogram compliance score (%) across the entire store shelf.

4. **Edge CV & Object Detection (shelf/detector.py)**:
   - Lightweight local CV model (supports YOLOv8 / color-texture feature extractor) for fine-grained product identification and count estimation on edge devices.
   - Zero-cloud dependency; executes on local CPU / edge accelerator.

5. **Live Shelf Pipeline & Main Integration (shelf/pipeline.py)**:
   - Connects to core/bus.py.
   - Subscribes to shelf_fill events or processes shelf frames directly.
   - Triggers automated staff alerts (alert event type) for critical replenishments.

6. **Interactive Calibration & Demo Utility (shelf/roi_calib.py, tools/demo_shelf.py)**:
   - Graphical and headless ROI calibration tool for custom shelves and test videos.
