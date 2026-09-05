import logging
from typing import Dict, List, Optional
import numpy as np

from core.bus import Bus
from core.events import Event
from shelf.fill import FacingReading, ShelfFillMonitor, ShelfStatus
from shelf.stockout import StockoutTracker
from shelf.planogram import PlanogramAuditor
from shelf.detector import ShelfEdgeDetector
from shelf.inventory_tracker import InventoryTracker
from shelf.pick_detector import ShelfPickDetector

log = logging.getLogger("shelf.pipeline")

class ShelfPipeline:
    def __init__(
        self,
        rois: List[dict],
        planogram: List[dict],
        store_id: str = "demo-01",
        bus: Optional[Bus] = None,
        use_detector: bool = True,
        detector_model: str = "yolo",
    ):
        self.store_id = store_id
        self.bus = bus
        self.fill_monitor = ShelfFillMonitor(rois=rois)
        self.stockout_tracker = StockoutTracker(planogram=planogram, store_id=store_id)
        self.planogram_auditor = PlanogramAuditor(planogram=planogram, store_id=store_id)
        self.inventory_tracker = InventoryTracker(planogram=planogram, store_id=store_id)
        self.pick_detector = ShelfPickDetector(rois=rois)
        self.detector = ShelfEdgeDetector(model_type=detector_model) if use_detector else None
        self._roi_index = {r["id"]: i for i, r in enumerate(rois)}
        self._rois_raw = rois

    def process_frame(
        self,
        t: float,
        result: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        direct_fills: Optional[Dict[str, int]] = None,
    ) -> Dict[str, FacingReading]:
        if result is not None and "roi_fill" in result.dtype.names:
            readings = self.fill_monitor.update_from_frame_result(result, self._roi_index)
        elif direct_fills is not None:
            readings = self.fill_monitor.update_from_roi_fill(direct_fills)
        elif image is not None:
            gray_320 = image
            if len(image.shape) == 3:
                import cv2
                gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_320 = cv2.resize(gray_full, (320, 240))
            fills = self.fill_monitor.compute_direct_fill(gray_320)
            readings = self.fill_monitor.update_from_roi_fill(fills)
        else:
            return {}

        # Stockout & replenishment alerts
        so_events = self.stockout_tracker.update(t, readings)

        # Inventory item-count / removal tracking based on fill changes
        inv_events = []
        for facing, reading in readings.items():
            evs = self.inventory_tracker.update_from_fractional_fill(t=t, facing=facing, fill_val=reading.smoothed_fill)
            inv_events.extend(evs)

        # Customer hand reaching / pick interactions
        if image is not None:
            picks = self.pick_detector.detect_interactions(image, t)
            # Log pick interaction if needed

        # Planogram compliance audit
        det_skus = None
        if self.detector is not None and image is not None:
            det_results = self.detector.analyze_rois(image, self._rois_raw)

        pl_events = self.planogram_auditor.audit_facings(t, readings, detected_skus=det_skus)

        # Publish events to bus
        if self.bus is not None:
            for ev in so_events + inv_events + pl_events:
                self.bus.publish(ev)

        return readings

    def register_manual_removal(self, t: float, facing: str, quantity: int = 1) -> List[Event]:
        events = self.inventory_tracker.register_removal(t=t, facing=facing, quantity=quantity)
        if self.bus is not None:
            for ev in events:
                self.bus.publish(ev)
        return events

    def get_inventory_summary(self) -> Dict[str, dict]:
        return self.inventory_tracker.get_summary()

    def get_compliance_score(self, readings: Dict[str, FacingReading]) -> float:
        return self.planogram_auditor.get_compliance_score(readings)

    def get_lost_revenue_estimate(self, facing: str, now_t: float) -> float:
        return self.stockout_tracker.get_lost_revenue_estimate(facing, now_t)
