import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

class ShelfStatus(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    EMPTY = "empty"

@dataclass
class ROIConfig:
    id: str
    x: int
    y: int
    w: int
    h: int
    empty_below: int = 60
    low_below: int = 110

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

@dataclass
class FacingReading:
    facing: str
    raw_fill: int
    smoothed_fill: float
    status: ShelfStatus
    confidence: float

class ShelfFillMonitor:
    def __init__(
        self,
        rois: List[dict],
        alpha: float = 0.35,
        history_len: int = 5,
    ):
        self.rois: Dict[str, ROIConfig] = {
            r["id"]: ROIConfig(
                id=r["id"],
                x=int(r["x"]),
                y=int(r["y"]),
                w=int(r["w"]),
                h=int(r["h"]),
                empty_below=int(r["empty_below"]),
                low_below=int(r["low_below"]),
            )
            for r in rois
        }
        self.alpha = float(alpha)
        self.history_len = int(history_len)

        self._smoothed_fill: Dict[str, float] = {}
        self._history: Dict[str, List[ShelfStatus]] = {r_id: [] for r_id in self.rois}
        self._current_status: Dict[str, ShelfStatus] = {r_id: ShelfStatus.NORMAL for r_id in self.rois}

    def update_from_roi_fill(self, fills: Dict[str, int]) -> Dict[str, FacingReading]:
        readings = {}
        for roi_id, cfg in self.rois.items():
            if roi_id not in fills:
                continue
            raw = int(np.clip(fills[roi_id], 0, 255))
            prev_smooth = self._smoothed_fill.get(roi_id, float(raw))
            smooth = self.alpha * raw + (1.0 - self.alpha) * prev_smooth
            self._smoothed_fill[roi_id] = smooth

            if smooth <= cfg.empty_below:
                inst_status = ShelfStatus.EMPTY
            elif smooth <= cfg.low_below:
                inst_status = ShelfStatus.LOW
            else:
                inst_status = ShelfStatus.NORMAL

            hist = self._history[roi_id]
            hist.append(inst_status)
            if len(hist) > self.history_len:
                hist.pop(0)

            counts = {
                ShelfStatus.NORMAL: hist.count(ShelfStatus.NORMAL),
                ShelfStatus.LOW: hist.count(ShelfStatus.LOW),
                ShelfStatus.EMPTY: hist.count(ShelfStatus.EMPTY),
            }
            dominant = max(counts, key=counts.get)
            confidence = counts[dominant] / len(hist)

            if counts[dominant] >= (len(hist) // 2 + 1):
                self._current_status[roi_id] = dominant

            readings[roi_id] = FacingReading(
                facing=roi_id,
                raw_fill=raw,
                smoothed_fill=round(smooth, 1),
                status=self._current_status[roi_id],
                confidence=round(confidence, 2),
            )
        return readings

    def update_from_frame_result(
        self, result: np.ndarray, roi_index: Dict[str, int]
    ) -> Dict[str, FacingReading]:
        fills = {}
        for roi_id, idx in roi_index.items():
            if idx < len(result["roi_fill"]):
                fills[roi_id] = int(result["roi_fill"][idx])
        return self.update_from_roi_fill(fills)

    def compute_direct_fill(self, gray_320: np.ndarray) -> Dict[str, int]:
        gx = cv2.Sobel(gray_320, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_320, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.convertScaleAbs(np.abs(gx) + np.abs(gy))
        fills = {}
        for roi_id, cfg in self.rois.items():
            x, y, w, h = cfg.x, cfg.y, cfg.w, cfg.h
            crop = mag[y:y + h, x:x + w]
            fills[roi_id] = int(np.clip(crop.mean(), 0, 255)) if crop.size else 0
        return fills

    def get_status(self, roi_id: str) -> ShelfStatus:
        return self._current_status.get(roi_id, ShelfStatus.NORMAL)

    def get_smoothed_fill(self, roi_id: str) -> float:
        return self._smoothed_fill.get(roi_id, 0.0)

    def reset(self):
        self._smoothed_fill.clear()
        for r_id in self.rois:
            self._history[r_id].clear()
            self._current_status[r_id] = ShelfStatus.NORMAL
