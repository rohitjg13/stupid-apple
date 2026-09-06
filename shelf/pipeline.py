import logging
from typing import Any, Dict, List, Optional
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

# Both trackers below raise these for the same physical shelf.
STOCKOUT_RULES = ("stockout_detected", "low_stock_detected")

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
        self._last_active_picks: List[Any] = []
        self._open_stockouts = set()        # facing -> one incident at a time
        self._alerted = set()               # (facing, rule) already raised

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

        # Inventory item-count / removal tracking
        inv_events = []
        for facing, reading in readings.items():
            if reading.status == ShelfStatus.EMPTY:
                evs = self.inventory_tracker.update_from_measurement(t=t, facing=facing, measured_qty=0)
                inv_events.extend(evs)
            elif reading.status == ShelfStatus.LOW:
                item_init = self.inventory_tracker.items[facing].initial_stock if facing in self.inventory_tracker.items else 10
                evs = self.inventory_tracker.update_from_measurement(t=t, facing=facing, measured_qty=int(item_init * 0.3))
                inv_events.extend(evs)
            else:
                item_init = self.inventory_tracker.items[facing].initial_stock if facing in self.inventory_tracker.items else 10
                self.inventory_tracker.update_from_measurement(t=t, facing=facing, measured_qty=item_init)

        # Customer hand reaching / pick interactions
        active_picks = []
        if image is not None:
            active_picks = self.pick_detector.detect_interactions(image, t)
            completed_picks = self.pick_detector.pop_completed_picks()
            for cp in completed_picks:
                pick_evs = self.inventory_tracker.register_removal(t=t, facing=cp.facing, quantity=1)
                inv_events.extend(pick_evs)
                inv_events.append(
                    Event(
                        t=t,
                        store_id=self.store_id,
                        stream="shelf",
                        zone_id=cp.facing,
                        event_type="alert",
                        payload={
                            "severity": "info",
                            "rule": "customer_pick_detected",
                            "message": f"Customer picked 1 item from {cp.facing} (confidence: {cp.confidence})",
                        },
                    )
                )
        self._last_active_picks = active_picks

        # Planogram compliance audit
        det_skus = None
        if self.detector is not None and image is not None:
            det_results = self.detector.analyze_rois(image, self._rois_raw)

        pl_events = self.planogram_auditor.audit_facings(t, readings, detected_skus=det_skus)

        events = self._one_incident(so_events + inv_events + pl_events, readings)
        if self.bus is not None:
            for ev in events:
                self.bus.publish(ev)

        return readings

    def _one_incident(self, events, readings):
        """One empty shelf is one incident, however many detectors noticed.

        `ShelfFillMonitor` watches edge density and `InventoryTracker` watches
        the item count; both fire on the same facing when it empties. Passing
        both through opens two rows in the stockout table, doubles the lost
        revenue, and prints the alert twice on the dashboard.
        """
        for facing, reading in readings.items():
            if reading.status == ShelfStatus.NORMAL:      # restocked: arm it again
                self._alerted -= {(facing, rule) for rule in STOCKOUT_RULES}

        out = []
        for ev in events:
            facing = ev.zone_id
            if ev.event_type == "stockout_start":
                if facing in self._open_stockouts:
                    continue
                self._open_stockouts.add(facing)
            elif ev.event_type == "stockout_end":
                if facing not in self._open_stockouts:
                    continue
                self._open_stockouts.discard(facing)
            elif ev.event_type == "alert" and ev.payload.get("rule") in STOCKOUT_RULES:
                if (facing, ev.payload["rule"]) in self._alerted:
                    continue
                self._alerted.add((facing, ev.payload["rule"]))
            out.append(ev)
        return out

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

    def get_active_picks(self) -> List[Any]:
        return getattr(self, "_last_active_picks", [])
