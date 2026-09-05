import logging
from typing import Dict, List, Optional
import numpy as np

from core.bus import Bus
from core.events import Event
from shelf.fill import FacingReading, ShelfFillMonitor, ShelfStatus
from shelf.stockout import StockoutTracker
from shelf.planogram import PlanogramAuditor
from shelf.detector import ShelfEdgeDetector

log = logging.getLogger("shelf.pipeline")

class ShelfPipeline:
    def __init__(
        self,
        rois: List[dict],
        planogram: List[dict],
        store_id: str = "demo-01",
        bus: Optional[Bus] = None,
        use_detector: bool = False,
    ):
        self.store_id = store_id
        self.bus = bus
        self.fill_monitor = ShelfFillMonitor(rois=rois)
        self.stockout_tracker = StockoutTracker(planogram=planogram, store_id=store_id)
        self.planogram_auditor = PlanogramAuditor(planogram=planogram, store_id=store_id)
        self.detector = ShelfEdgeDetector(model_type="fast_cv") if use_detector else None
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

        # Planogram compliance audit
        det_skus = None
        if self.detector is not None and image is not None:
            det_results = self.detector.analyze_rois(image, self._rois_raw)

        pl_events = self.planogram_auditor.audit_facings(t, readings, detected_skus=det_skus)

        # Publish events to bus
        if self.bus is not None:
            for ev in so_events + pl_events:
                self.bus.publish(ev)

        return readings

    def get_compliance_score(self, readings: Dict[str, FacingReading]) -> float:
        return self.planogram_auditor.get_compliance_score(readings)

    def get_lost_revenue_estimate(self, facing: str, now_t: float) -> float:
        return self.stockout_tracker.get_lost_revenue_estimate(facing, now_t)
