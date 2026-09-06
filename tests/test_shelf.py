import numpy as np
import pytest

from core.bus import Bus
from core.events import Event
from shelf.fill import ShelfFillMonitor, ShelfStatus
from shelf.stockout import StockoutTracker
from shelf.planogram import PlanogramAuditor
from shelf.detector import ShelfEdgeDetector
from shelf.pipeline import ShelfPipeline

ROIS = [
    {"id": "A1", "stream": "shelf", "x": 20, "y": 40, "w": 60, "h": 70, "empty_below": 60, "low_below": 110},
    {"id": "A2", "stream": "shelf", "x": 90, "y": 40, "w": 60, "h": 70, "empty_below": 60, "low_below": 110},
]

PLANOGRAM = [
    {"facing": "A1", "sku": "MAGGI-70G", "name": "Maggi Noodles 70g", "unit_price": 14, "expected_sales_per_hour": 6},
    {"facing": "A2", "sku": "PARLEG-100G", "name": "Parle-G 100g", "unit_price": 10, "expected_sales_per_hour": 8},
]

def test_shelf_fill_monitor_smoothing_and_status():
    monitor = ShelfFillMonitor(rois=ROIS, alpha=0.5, history_len=3)
    
    # Send high fills (full shelf)
    for _ in range(5):
        readings = monitor.update_from_roi_fill({"A1": 180, "A2": 150})
    assert readings["A1"].status == ShelfStatus.NORMAL
    assert readings["A2"].status == ShelfStatus.NORMAL

    # Now drop A1 to empty
    for _ in range(8):
        readings = monitor.update_from_roi_fill({"A1": 20, "A2": 150})
    assert readings["A1"].status == ShelfStatus.EMPTY
    assert readings["A2"].status == ShelfStatus.NORMAL

    # Now drop A2 to low stock
    for _ in range(8):
        readings = monitor.update_from_roi_fill({"A1": 20, "A2": 80})
    assert readings["A2"].status == ShelfStatus.LOW

def test_stockout_tracker_events_and_lost_revenue():
    tracker = StockoutTracker(planogram=PLANOGRAM, store_id="demo-01")
    monitor = ShelfFillMonitor(rois=ROIS, alpha=0.6, history_len=3)

    # Initially normal
    for _ in range(3):
        readings = monitor.update_from_roi_fill({"A1": 180, "A2": 150})
        events = tracker.update(t=100.0, readings=readings)
    assert len(events) == 0

    # Stockout on A1: collect all emitted events across frames
    all_events = []
    start_t = 105.0
    for step in range(8):
        readings = monitor.update_from_roi_fill({"A1": 10, "A2": 150})
        events = tracker.update(t=start_t + step, readings=readings)
        all_events.extend(events)

    # Should have stockout_start event and alert
    event_types = [e.event_type for e in all_events]
    assert "stockout_start" in event_types
    assert "alert" in event_types

    # Find the exact stockout_start event time
    start_event = next(e for e in all_events if e.event_type == "stockout_start")
    t_out = start_event.t

    # Check lost revenue calculation exactly 1 hour after stockout_start
    rev = tracker.get_lost_revenue_estimate("A1", now_t=t_out + 3600.0)
    assert rev == 84.0  # 1 hour * 6 units/hr * 14 = 84.0

    # Restock A1: collect all emitted events
    restock_events = []
    for step in range(8):
        readings = monitor.update_from_roi_fill({"A1": 180, "A2": 150})
        events = tracker.update(t=t_out + 3600.0 + step, readings=readings)
        restock_events.extend(events)

    end_types = [e.event_type for e in restock_events]
    assert "stockout_end" in end_types

def test_planogram_auditor():
    auditor = PlanogramAuditor(planogram=PLANOGRAM, store_id="demo-01")
    monitor = ShelfFillMonitor(rois=ROIS, alpha=0.6, history_len=3)

    for _ in range(3):
        readings = monitor.update_from_roi_fill({"A1": 180, "A2": 150})
    events = auditor.audit_facings(t=10.0, readings=readings)
    assert len(events) == 0
    assert auditor.get_compliance_score(readings) == 100.0

    # Trigger empty
    all_events = []
    for step in range(8):
        readings = monitor.update_from_roi_fill({"A1": 20, "A2": 150})
        events = auditor.audit_facings(t=15.0 + step, readings=readings)
        all_events.extend(events)

    assert any(e.event_type == "planogram_violation" and e.payload["observed"] == "empty" for e in all_events)
    assert auditor.get_compliance_score(readings) == 50.0

def test_shelf_pipeline_end_to_end():
    bus = Bus()
    captured_events = []
    bus.subscribe("*", lambda e: captured_events.append(e))

    pipe = ShelfPipeline(rois=ROIS, planogram=PLANOGRAM, store_id="demo-01", bus=bus)

    # Process simulated empty frames
    for step in range(8):
        pipe.process_frame(t=50.0 + step, direct_fills={"A1": 10, "A2": 150})

    bus.drain()
    types = [e.event_type for e in captured_events]
    assert "stockout_start" in types
    assert "alert" in types
    assert "planogram_violation" in types


# ---- one incident per empty facing ----------------------------------------

def _empty_then_full(pipeline, rois, t0=0.0):
    """Drive the pipeline with every facing empty, then every facing stocked."""
    import numpy as np
    from pl.contract import FRAME_RESULT_DT
    out = []
    for i, fill in enumerate([5] * 12 + [220] * 12):
        r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
        r["roi_fill"][: len(rois)] = fill
        pipeline.process_frame(t=t0 + i, result=r)
    return out


def test_two_detectors_open_one_stockout(tmp_path):
    from core.bus import Bus
    from shelf.pipeline import ShelfPipeline

    rois = [{"id": "A1", "stream": "shelf", "x": 10, "y": 10, "w": 40, "h": 40,
             "empty_below": 60, "low_below": 110}]
    plan = [{"facing": "A1", "sku": "S1", "name": "One", "unit_price": 10,
             "expected_sales_per_hour": 4}]
    bus = Bus()
    seen = []
    bus.subscribe("*", seen.append)
    pipe = ShelfPipeline(rois=rois, planogram=plan, bus=bus, use_detector=False)
    _empty_then_full(pipe, rois)
    bus.drain()

    starts = [e for e in seen if e.event_type == "stockout_start"]
    assert len(starts) == 1, "fill and item count both saw the same empty shelf"
    alerts = [e for e in seen if e.event_type == "alert"
              and e.payload["rule"] == "stockout_detected"]
    assert len(alerts) == 1
