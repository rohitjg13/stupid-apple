import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PickInteraction:
    facing: str
    t: float
    confidence: float
    hand_bbox: Tuple[int, int, int, int]

class ShelfPickDetector:
    def __init__(
        self,
        rois: List[dict],
        motion_threshold: float = 25.0,
        min_contour_area: int = 400,
    ):
        self.rois = rois
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.prev_gray: Optional[np.ndarray] = None
        self.baseline_patches: Dict[str, np.ndarray] = {}

    def set_baseline(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]
        sx = w / 320.0
        sy = h / 240.0
        for r in self.rois:
            rx, ry, rw, rh = int(r["x"] * sx), int(r["y"] * sy), int(r["w"] * sx), int(r["h"] * sy)
            self.baseline_patches[r["id"]] = gray[ry:ry + rh, rx:rx + rw].copy()

    def detect_interactions(self, frame: np.ndarray, t: float) -> List[PickInteraction]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        interactions = []

        if self.prev_gray is None:
            self.prev_gray = gray
            return interactions

        diff = cv2.absdiff(self.prev_gray, gray)
        h, w = gray.shape[:2]
        sx = w / 320.0
        sy = h / 240.0

        for r in self.rois:
            r_id = r["id"]
            rx, ry, rw, rh = int(r["x"] * sx), int(r["y"] * sy), int(r["w"] * sx), int(r["h"] * sy)
            roi_diff = diff[ry:ry + rh, rx:rx + rw]
            if roi_diff.size == 0:
                continue

            _, thresh = cv2.threshold(roi_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            motion_pixels = cv2.countNonZero(thresh)
            total_pixels = rw * rh

            motion_ratio = motion_pixels / float(total_pixels)
            if motion_ratio > 0.15:  # At least 15% ROI has movement (hand reaching in)
                interactions.append(
                    PickInteraction(
                        facing=r_id,
                        t=t,
                        confidence=round(motion_ratio, 2),
                        hand_bbox=(rx, ry, rw, rh),
                    )
                )

        self.prev_gray = gray
        return interactions
