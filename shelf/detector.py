import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class ShelfEdgeDetector:
    def __init__(self, model_type: str = "fast_cv", model_path: Optional[str] = None):
        self.model_type = model_type
        self.model_path = model_path
        self._yolo = None

        if model_type == "yolo":
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(model_path or "yolov8n.pt")
            except Exception as e:
                # Fallback gracefully to fast_cv if YOLO cannot be initialized
                self.model_type = "fast_cv"

    def analyze_rois(
        self,
        image: np.ndarray,
        rois: List[dict],
        scale_from_320: bool = True,
    ) -> Dict[str, dict]:
        h, w = image.shape[:2]
        results = {}

        if self.model_type == "yolo" and self._yolo is not None:
            preds = self._yolo(image, verbose=False)[0]
            boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else []
            classes = preds.boxes.cls.cpu().numpy() if len(preds.boxes) else []
            confs = preds.boxes.conf.cpu().numpy() if len(preds.boxes) else []

            for r in rois:
                rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
                if scale_from_320 and (w != 320 or h != 240):
                    sx = w / 320.0
                    sy = h / 240.0
                    rx, ry, rw, rh = int(rx * sx), int(ry * sy), int(rw * sx), int(rh * sy)

                items_in_roi = 0
                max_conf = 0.0
                detected_class = None

                for box, cls, conf in zip(boxes, classes, confs):
                    bx1, by1, bx2, by2 = box
                    bcx = (bx1 + bx2) / 2.0
                    bcy = (by1 + by2) / 2.0
                    if rx <= bcx <= (rx + rw) and ry <= bcy <= (ry + rh):
                        items_in_roi += 1
                        if conf > max_conf:
                            max_conf = float(conf)
                            detected_class = int(cls)

                results[r["id"]] = {
                    "item_count": items_in_roi,
                    "confidence": max_conf,
                    "class_id": detected_class,
                }
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            for r in rois:
                rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
                if scale_from_320 and (gray.shape[1] != 320 or gray.shape[0] != 240):
                    sx = gray.shape[1] / 320.0
                    sy = gray.shape[0] / 240.0
                    rx, ry, rw, rh = int(rx * sx), int(ry * sy), int(rw * sx), int(rh * sy)

                crop = gray[ry:ry + rh, rx:rx + rw]
                if crop.size == 0:
                    results[r["id"]] = {"item_count": 0, "confidence": 0.0, "class_id": None}
                    continue

                gx = cv2.Sobel(crop, cv2.CV_16S, 1, 0, ksize=3)
                gy = cv2.Sobel(crop, cv2.CV_16S, 0, 1, ksize=3)
                mag = cv2.convertScaleAbs(np.abs(gx) + np.abs(gy))
                mean_density = float(mag.mean())

                # Texture & contour peak count
                _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = len([c for c in contours if cv2.contourArea(c) > 50])

                results[r["id"]] = {
                    "item_count": cnt,
                    "edge_density": round(mean_density, 1),
                    "confidence": round(min(1.0, mean_density / 100.0), 2),
                    "class_id": None,
                }

        return results
