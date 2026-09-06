import numpy as np
import pytest
from shelf.pick_detector import ShelfPickDetector
from shelf.pipeline import ShelfPipeline
from core.bus import Bus

ROIS = [
    {"id": "SHELF-1", "stream": "shelf", "x": 100, "y": 50, "w": 80, "h": 60, "empty_below": 40, "low_below": 80},
]
PLANOGRAM = [
    {"facing": "SHELF-1", "sku": "ITEM-1", "name": "Test Item", "unit_price": 50, "initial_stock": 5, "low_threshold": 2, "expected_sales_per_hour": 5},
]

def test_pick_detector_reach_and_completed_pick():
    detector = ShelfPickDetector(rois=ROIS, motion_threshold=20.0, min_motion_ratio=0.05, cooldown_seconds=0.5)

    # Frame 1: baseline (empty/static frame)
    frame1 = np.full((240, 320, 3), 100, dtype=np.uint8)
    interactions = detector.detect_interactions(frame1, t=1.0)
    assert len(interactions) == 0

    # Frames 2-5: hand moving inside shelf ROI (continuous motion)
    for step in range(4):
        frame_reach = frame1.copy()
        frame_reach[50:110, 100 + step * 5 : 180] = 240 + step * 2
        interactions = detector.detect_interactions(frame_reach, t=1.1 + step * 0.05)
        assert len(interactions) >= 1
        assert interactions[0].facing == "SHELF-1"

    # Frame 6-7: hand leaves shelf ROI (withdrawn with item, then scene settles)
    frame_leave = frame1.copy()
    detector.detect_interactions(frame_leave, t=2.0)
    detector.detect_interactions(frame_leave, t=2.1)
    
    completed = detector.pop_completed_picks()
    assert len(completed) == 1
    assert completed[0].facing == "SHELF-1"

def test_pipeline_automated_pick_removal():
    bus = Bus()
    events = []
    bus.subscribe("*", lambda e: events.append(e))

    pipe = ShelfPipeline(rois=ROIS, planogram=PLANOGRAM, store_id="test-store", bus=bus)
    inv = pipe.get_inventory_summary()
    assert inv["SHELF-1"]["current"] == 5
    assert inv["SHELF-1"]["removed"] == 0

    # Initial frame
    f0 = np.full((240, 320, 3), 100, dtype=np.uint8)
    pipe.process_frame(t=1.0, image=f0)

    # Hand reaches in with continuous movement
    for s in range(4):
        f_reach = f0.copy()
        f_reach[50:110, 100 + s * 5 : 180] = 240 + s * 2
        pipe.process_frame(t=1.1 + s * 0.05, image=f_reach)

    # Hand leaves and shelf settles -> completes pick!
    pipe.process_frame(t=2.0, image=f0)
    pipe.process_frame(t=2.1, image=f0)
    bus.drain()

    inv_after = pipe.get_inventory_summary()
    assert inv_after["SHELF-1"]["current"] == 4
    assert inv_after["SHELF-1"]["removed"] == 1
    assert any(e.event_type == "alert" and e.payload.get("rule") == "customer_pick_detected" for e in events)

