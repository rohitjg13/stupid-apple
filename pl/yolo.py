"""YOLO person detector as a FrameResult backend. For the Jetson, not the FPGA.

    python main.py --source file --backend yolo --config config/overhead_01

Emits the same 856-byte FrameResult as pl/reference.py and pl/driver.py, so the
shopper tracker and everything downstream do not know or care that the boxes
came from a detector. Two things are different, and both are improvements:

* **No warm-up.** A detector has no background model to learn, so
  FLAG_BG_WARM is set from the first frame.
* **It sees people who stand still.** Background subtraction detects change and
  goes blind to a stationary shopper; a detector recognises a person by
  appearance and keeps the whole body boxed. That is the single failure that
  cost most of the tuning on eye-level footage.

roi_fill and lane_occupancy still come from the reference chain (Sobel texture
per shelf ROI, foreground per queue cell) -- a person detector has nothing to
say about a shelf -- and the shelf stream is delegated to it entirely.

`ultralytics` (and torch) are imported lazily inside this file only, the way
pl/driver.py isolates `pynq`. `--source sim` and the whole test suite run
without them installed.
"""
from __future__ import annotations

import logging

import numpy as np

from pl.contract import FLAG_BG_WARM, FLAG_TRUNCATED, MAX_BLOBS
from pl.reference import FULL_H, FULL_W, ReferenceBackend, letterbox

log = logging.getLogger(__name__)

PERSON = 0                      # COCO class id


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def pack_detections(result, boxes):
    """Write (x, y, w, h) person boxes into a FrameResult, contract-shaped.

    Full-res 640x480 px, clipped to the frame, area = w*h, sorted by area
    descending, capped at 64 with FLAG_TRUNCATED. Confidence has nowhere to go --
    the contract has no field for it -- so the tracker keeps using area as its
    pseudo-score, which for whole-body boxes is a fine proxy for "big and near".
    """
    kept = []
    for x, y, w, h in boxes:
        x0, y0 = int(max(0, min(x, FULL_W - 1))), int(max(0, min(y, FULL_H - 1)))
        x1, y1 = int(max(0, min(x + w, FULL_W))), int(max(0, min(y + h, FULL_H)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        kept.append((x0, y0, x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0)))
    kept.sort(key=lambda b: b[4], reverse=True)

    result["blobs"][:] = 0
    result["num_blobs"] = min(len(kept), MAX_BLOBS)
    for i, b in enumerate(kept[:MAX_BLOBS]):
        result["blobs"][i] = b
    flags = int(result["flags"]) | FLAG_BG_WARM
    if len(kept) > MAX_BLOBS:
        flags |= FLAG_TRUNCATED
    result["flags"] = flags
    return result


class YoloBackend:
    """Drop-in for pl.driver / pl.reference: process(image, stream_id, frame_id)."""

    def __init__(self, cfg, model="yolov8n.pt", conf=0.3, device=None,
                 detector=None, reference=None):
        self.cfg = cfg
        self.conf = float(conf)
        self.model_name = model
        self.device = device
        self._model = None
        # Injectable for tests: callable(bgr 640x480) -> [(x, y, w, h), ...]
        self._detector = detector
        self.ref = reference or ReferenceBackend(cfg)

    # --- model ------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO
        self._model = YOLO(self.model_name)
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        log.info("yolo loaded", extra={"model": self.model_name, "device": self.device})
        return self._model

    def _detect(self, image):
        """Person boxes (x, y, w, h) in the letterboxed 640x480 frame."""
        if self._detector is not None:
            return list(self._detector(image))
        model = self._load()
        out = model.predict(image, conf=self.conf, classes=[PERSON], verbose=False,
                            device=self.device, imgsz=640)
        if not out:
            return []
        xyxy = out[0].boxes.xyxy
        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        return [(float(x0), float(y0), float(x1 - x0), float(y1 - y0))
                for x0, y0, x1, y1 in xyxy]

    # --- backend ----------------------------------------------------------
    def process(self, image, stream_id=0, frame_id=0):
        # Reference fills roi_fill and lane_occupancy (and its own blobs, which
        # the overhead path then overwrites). The shelf stream is its job alone.
        result = self.ref.process(image, stream_id, frame_id)
        if int(stream_id) != 0:
            return result
        boxes = self._detect(letterbox(image))
        return pack_detections(result, boxes)

    def available(self):
        return self._detector is not None or available()

    def close(self):
        self.ref.close() if hasattr(self.ref, "close") else None
