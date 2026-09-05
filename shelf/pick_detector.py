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
        min_motion_ratio: float = 0.05,
        min_contour_area: int = 400,
        cooldown_seconds: float = 1.5,
    ):
        self.rois = rois
        self.motion_threshold = motion_threshold
        self.min_motion_ratio = min_motion_ratio
        self.min_contour_area = min_contour_area
        self.cooldown_seconds = cooldown_seconds
        self.prev_gray: Optional[np.ndarray] = None
        self.baseline_patches: Dict[str, np.ndarray] = {}
        # State per ROI: {"active": bool, "start_t": float, "last_t": float, "reach_frames": int, "last_pick_t": float}
        self.roi_states: Dict[str, dict] = {
            r["id"]: {
                "active": False,
                "start_t": 0.0,
                "last_t": 0.0,
                "reach_frames": 0,
                "last_pick_t": -100.0,
            }
            for r in rois
        }
        self.completed_picks: List[PickInteraction] = []

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
            state = self.roi_states.get(r_id, {})
            max_val = roi_diff.max() if roi_diff.size else 0

            # Hand entering / active inside shelf ROI
            is_active_reach = (motion_ratio >= self.min_motion_ratio and max_val > 120) or motion_ratio > 0.12

            if is_active_reach:
                state["reach_frames"] = state.get("reach_frames", 0) + 1
                state["last_t"] = t
                if not state.get("active", False):
                    state["active"] = True
                    state["start_t"] = t

                interactions.append(
                    PickInteraction(
                        facing=r_id,
                        t=t,
                        confidence=round(min(1.0, motion_ratio * 3), 2),
                        hand_bbox=(rx, ry, rw, rh),
                    )
                )
            else:
                # Check if a reach just ended (hand withdrawn after contact)
                if state.get("active", False):
                    contact_duration = t - state.get("start_t", t)
                    reach_frames = state.get("reach_frames", 0)
                    time_since_last_pick = t - state.get("last_pick_t", -100.0)

                    # If hand was inside ROI for at least 3 frames and cooldown has elapsed
                    if reach_frames >= 3 and time_since_last_pick > self.cooldown_seconds and r.get("pick_monitor", True):
                        pick_event = PickInteraction(
                            facing=r_id,
                            t=t,
                            confidence=0.92,
                            hand_bbox=(rx, ry, rw, rh),
                        )
                        self.completed_picks.append(pick_event)
                        state["last_pick_t"] = t

                    # Reset reach state
                    state["active"] = False
                    state["reach_frames"] = 0

        self.prev_gray = gray
        return interactions

    def pop_completed_picks(self) -> List[PickInteraction]:
        picks = self.completed_picks.copy()
        self.completed_picks.clear()
        return picks

