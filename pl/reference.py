"""OpenCV clone of the PL chain — the golden model.

Same stage order, same registers, same 856-byte output as pl/driver.py, so
`compare_backends.py` can diff them and Khushwant can match stage by stage:

    BGR 640x480 -> gray -> resize 320x240 (AREA) -> bg model -> morph open xN
                -> CCL -> filter >= MIN_BLOB_AREA -> sort by area -> cap 64 -> x2
                -> Sobel |gx|+|gy| mean per ROI -> 0..255
                -> foreground %% per lane cell -> 0..100

Every parameter comes from the same register values the driver would write.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from pl.contract import FLAG_BG_WARM, FRAME_RESULT_DT, MAX_BLOBS
from pl.regs import MAX_LANES, MAX_ROIS, PL_H, PL_W, q0_16, q8_8

log = logging.getLogger(__name__)

FULL_W, FULL_H = 640, 480
SCALE = FULL_W // PL_W          # PL works at half res and doubles coords back
WARMUP_FRAMES = 100             # matches mog2_lite's warm-up counter in hls/


def letterbox(img):
    """Sources letterbox anything that is not 640x480 (SHARED.md §4)."""
    h, w = img.shape[:2]
    if (h, w) == (FULL_H, FULL_W):
        return img
    s = min(FULL_W / w, FULL_H / h)
    rw, rh = int(round(w * s)), int(round(h * s))
    out = np.zeros((FULL_H, FULL_W, 3), np.uint8)
    y, x = (FULL_H - rh) // 2, (FULL_W - rw) // 2
    out[y:y + rh, x:x + rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    return out


class _RunningGaussian:
    """Option B from plan 01: mean + variance per pixel, no mixture."""

    def __init__(self, var_thresh, lr):
        self.var_thresh, self.lr = var_thresh, lr
        self.mean = self.var = None

    def apply(self, gray):
        g = gray.astype(np.float32)
        if self.mean is None:
            self.mean, self.var = g.copy(), np.full_like(g, 100.0)
            return np.zeros(g.shape, np.uint8)
        d = g - self.mean
        fg = (np.abs(d) > self.var_thresh * np.sqrt(self.var)).astype(np.uint8) * 255
        self.mean += self.lr * d
        self.var = np.maximum(self.var + self.lr * (d * d - self.var), 1.0)
        return fg


class ReferenceBackend:
    """Drop-in for pl.driver: process(image, stream_id) -> FrameResult."""

    def __init__(self, cfg, bg_model="mog2", var_thresh=16.0, lr=0.005,
                 min_blob_area=150, morph_iters=1, warmup_frames=WARMUP_FRAMES,
                 stage_dir=None, debug_stage=None):
        if bg_model not in ("mog2", "rg"):
            raise ValueError(f"unknown bg model {bg_model!r}, expected 'mog2' or 'rg'")
        self.cfg, self.bg_model = cfg, bg_model
        # Round-trip through the register encodings so the reference cannot be
        # tuned to a precision the hardware does not have.
        self.var_thresh = q8_8(var_thresh) / 256.0
        self.lr = q0_16(lr) / 65536.0
        self.min_blob_area = int(min_blob_area)
        self.morph_iters = int(morph_iters)
        self.warmup_frames = int(warmup_frames)
        self.stage_dir = stage_dir
        self.debug_stage = debug_stage

        self.rois = [(r["x"], r["y"], r["w"], r["h"]) for r in cfg.rois][:MAX_ROIS]
        self.lanes = [(l["x"], l["y"], l["w"], l["h"]) for l in cfg.lanes][:MAX_LANES]
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._stream = None
        self._reset_bg()

    # ---- background -------------------------------------------------------
    def _reset_bg(self):
        """One background model exists in BRAM, so switching STREAM_ID re-warms it."""
        self._seen = 0
        if self.bg_model == "mog2":
            self._bg = cv2.createBackgroundSubtractorMOG2(500, self.var_thresh, False)
        else:
            self._bg = _RunningGaussian(self.var_thresh, self.lr)

    def _foreground(self, gray):
        if self.bg_model == "mog2":
            return self._bg.apply(gray, learningRate=self.lr)
        return self._bg.apply(gray)

    # ---- chain ------------------------------------------------------------
    def process(self, image, stream_id=0, frame_id=0):
        stream = int(stream_id)
        if stream != self._stream:
            self._reset_bg()
            self._stream = stream

        r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
        r["frame_id"] = frame_id

        gray_full = cv2.cvtColor(letterbox(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_full, (PL_W, PL_H), interpolation=cv2.INTER_AREA)

        mask = self._foreground(gray)
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel,
                                 iterations=self.morph_iters) if self.morph_iters else mask

        self._seen += 1
        if self._seen >= self.warmup_frames:
            r["flags"] |= FLAG_BG_WARM
        else:
            morph = np.zeros_like(morph)      # a cold model reports no blobs

        n, _, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
        blobs = [tuple(stats[i, :5]) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_area]
        blobs.sort(key=lambda b: b[4], reverse=True)
        if len(blobs) > MAX_BLOBS:
            r["flags"] |= 1 << 1
        r["num_blobs"] = min(len(blobs), MAX_BLOBS)
        for i, (x, y, w, h, area) in enumerate(blobs[:MAX_BLOBS]):
            r["blobs"][i] = (x * SCALE, y * SCALE, w * SCALE, h * SCALE, int(area) * SCALE * SCALE)

        # Sobel edge density per shelf ROI, 0..255 (255 = full texture)
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.convertScaleAbs(np.abs(gx) + np.abs(gy))
        for i, (x, y, w, h) in enumerate(self.rois):
            r["roi_fill"][i] = int(np.clip(mag[y:y + h, x:x + w].mean(), 0, 255))

        # foreground percentage per queue cell, 0..100
        for i, (x, y, w, h) in enumerate(self.lanes):
            cell = morph[y:y + h, x:x + w]
            r["lane_occupancy"][i] = int(round(float((cell > 0).mean()) * 100)) if cell.size else 0

        if self.stage_dir is not None:
            self._dump(frame_id, gray, mask, morph)
        return r

    def _dump(self, frame_id, gray, mask, morph):
        """--stage-outputs: golden PNGs for hls/tb/golden/."""
        from pathlib import Path
        d = Path(self.stage_dir)
        d.mkdir(parents=True, exist_ok=True)
        for idx, (name, img) in enumerate((("gray", gray), ("mask", mask), ("morph", morph))):
            cv2.imwrite(str(d / f"{frame_id:06d}_{idx}_{name}.png"), img)

    def available(self):
        return True

    def close(self):
        pass


def main(argv=None):
    """python -m pl.reference <clip> --config <clipset> --stage-outputs <dir>"""
    import argparse

    from core.config import load_clipset
    from sources.file import FileSource

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--stream", default="overhead")
    ap.add_argument("--bg-model", default="mog2", choices=["mog2", "rg"])
    ap.add_argument("--stage-outputs", default=None, metavar="DIR")
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_clipset(a.config)
    be = ReferenceBackend(cfg, bg_model=a.bg_model, stage_dir=a.stage_outputs)
    src = FileSource(cfg, stream=a.stream, decode=True)
    for i, frame in enumerate(src.frames()):
        if i >= a.frames:
            break
        r = be.process(frame.image, 0 if a.stream == "overhead" else 1, frame.frame_id)
        log.info("frame %d blobs=%d", frame.frame_id, int(r["num_blobs"]))
    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
