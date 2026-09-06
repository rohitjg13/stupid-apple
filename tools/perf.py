"""Per-backend timing table: the numbers behind the "why this box" slide.

    python -m tools.perf --config config/overhead_01 --backends reference yolo

On the Jetson that is the CUDA detector against the CPU chain on the identical
frames -- the honest answer to "why not a Raspberry Pi".

The `PL ms` column stays empty unless the backend is the PYNQ driver, where it
reports the fabric's own first-pixel-in to last-byte-out counter, separate from
DMA and Python overhead.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("perf")

FCLK_HZ = 100_000_000       # 100 MHz, plan 01


@dataclass
class Stats:
    label: str
    frames: int
    mean_ms: float
    p95_ms: float
    fps: float
    latency_cycles: Optional[int] = None

    @property
    def pl_ms_at_100mhz(self):
        return None if self.latency_cycles is None else self.latency_cycles / FCLK_HZ * 1e3


def benchmark(backend, images, label="backend", stream_id=0, warmup=3) -> Stats:
    for i, img in enumerate(images[:warmup]):
        backend.process(img, stream_id, i)

    times = []
    t0 = time.perf_counter()
    for i, img in enumerate(images):
        t = time.perf_counter()
        backend.process(img, stream_id, i)
        times.append((time.perf_counter() - t) * 1e3)
    wall = time.perf_counter() - t0

    cycles = None
    if hasattr(backend, "read_latency_cycles"):
        try:
            cycles = int(backend.read_latency_cycles())
        except Exception as e:          # a driver without a bitstream loaded
            log.warning("cannot read LATENCY", extra={"error": str(e)})

    return Stats(label, len(images), float(np.mean(times)),
                 float(np.percentile(times, 95)), len(images) / wall if wall else 0.0, cycles)


def format_table(rows) -> str:
    head = ("backend", "frames", "mean ms", "p95 ms", "fps", "PL ms")
    body = [(s.label, str(s.frames), f"{s.mean_ms:.2f}", f"{s.p95_ms:.2f}", f"{s.fps:.1f}",
             "-" if s.pl_ms_at_100mhz is None else f"{s.pl_ms_at_100mhz:.2f}") for s in rows]
    w = [max(len(r[i]) for r in (head, *body)) for i in range(len(head))]
    fmt = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip().ljust(sum(w) + 2 * (len(w) - 1))
    return "\n".join([fmt(head), fmt(tuple("-" * x for x in w)), *(fmt(r) for r in body)])


def load_images(cfg, stream, source, n):
    """A raw .npy memmap from tools/transcode.sh --raw skips decode cost."""
    path = cfg.streams[stream].get("path", "")
    if str(path).endswith(".npy"):
        return list(np.load(path, mmap_mode="r")[:n])
    from main import build_source
    src = build_source(source, cfg, stream)
    try:
        return [f.image for _, f in zip(range(n), src.frames()) if f.image is not None]
    finally:
        src.close()


def main(argv=None):
    from core.config import load_clipset
    from main import build_backend

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/sim")
    ap.add_argument("--stream", default="overhead")
    ap.add_argument("--source", default="file", choices=["file", "sim", "camera"])
    ap.add_argument("--backends", nargs="+", default=["reference", "yolo"])
    ap.add_argument("--frames", type=int, default=200)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_clipset(a.config)
    images = load_images(cfg, a.stream, a.source, a.frames)
    if not images:
        raise SystemExit("no frames decoded; check store.yaml path")

    rows = [benchmark(build_backend(b, cfg), images, label=b,
                      stream_id=0 if a.stream == "overhead" else 1) for b in a.backends]
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
