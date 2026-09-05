"""Retail analytics entry point.

capture thread -> Queue(maxsize=2) -> analytics thread -> bus -> subscribers.
Runs on a laptop with no board: `python main.py --source sim`.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import signal
import sys
import threading
import time

from core.bus import Bus
from core.clock import Clock
from core.config import load_clipset
from core.events import Event
from pathlib import Path

log = logging.getLogger("main")


class JsonLines(logging.Formatter):
    def format(self, rec):
        out = {"t": round(rec.created, 3), "level": rec.levelname, "logger": rec.name,
               "msg": rec.getMessage()}
        out.update({k: v for k, v in rec.__dict__.items()
                    if k not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__
                    and k not in ("message", "asctime")})
        return json.dumps(out)


def build_source(source, cfg, stream):
    if source == "sim":
        from sources.sim import SimSource
        return SimSource(cfg, stream=stream)
    if source == "file":
        from sources.file import FileSource
        return FileSource(cfg, stream=stream)
    if source == "camera":
        from sources.camera import CameraSource
        return CameraSource(cfg, stream=stream)
    raise SystemExit(f"unknown --source {source!r}; expected sim, file or camera")


def build_backend(backend, cfg):
    """None means the source already carries a FrameResult (sim)."""
    if backend == "sim":
        return None
    if backend == "reference":
        from pl.reference import ReferenceBackend
        return ReferenceBackend(cfg)
    if backend == "pl":
        from pl import driver
        # Khushwant owns pl/driver.py. Accept either shape his plan allows:
        # a Driver class, or module-level process()/read_latency_cycles().
        return driver.Driver(cfg) if hasattr(driver, "Driver") else driver
    raise SystemExit(f"unknown --backend {backend!r}; expected sim, reference or pl")


def check_backend(backend):
    """`--backend pl` must never silently degrade to reference on stage."""
    if backend != "pl":
        return
    try:
        from pl import driver
        ok = driver.available()
    except Exception as e:
        ok, reason = False, e
    else:
        reason = "driver.available() is False"
    if not ok:
        log.error("pl backend unavailable", extra={"reason": str(reason)})
        raise SystemExit("--backend pl requested but no board is available; "
                         "refusing to start (use --backend reference deliberately)")


def capture(src, q, stop):
    for frame in src.frames():
        if stop.is_set():
            break
        try:
            q.put(frame, timeout=1.0)
        except queue.Full:
            log.warning("analytics behind, dropping frame", extra={"frame_id": frame.frame_id})
    q.put(None)


def write_health(path, payload):
    """Offline-first: a file the backend's /health can serve. Never a network call."""
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(p)
    except OSError as e:
        log.warning("cannot write health file", extra={"error": str(e)})


def run(config, source="sim", backend="sim", frames=0, realtime=True, bus=None,
        streams=("overhead", "shelf"), stop=None, clock=None, health_file=None,
        clock_state="/var/lib/retail/clock.json"):
    cfg = load_clipset(config)
    check_backend(backend)
    bus = bus or Bus()
    stop = stop or threading.Event()      # per-run: run() must be re-callable in one process

    sources = {s: build_source(source if cfg.streams[s].get("source") != "sim" else "sim", cfg, s)
               for s in streams}
    be = build_backend(backend, cfg)
    q = queue.Queue(maxsize=2)
    threads = [threading.Thread(target=capture, args=(src, q, stop), daemon=True, name=f"cap-{s}")
               for s, src in sources.items()]
    for t in threads:
        t.start()

    clock = clock or Clock(clock_state)
    if not clock.synced:
        log.warning("starting with an unsynced clock; cloud sync will wait")
    processed, finished, last_pub = 0, 0, {}
    epoch_base, ns_base = clock.now(), {}      # per stream: each has its own t_ns origin
    last_health = 0.0
    t_wall = time.monotonic()
    while finished < len(sources):
        item = q.get()
        if item is None:
            finished += 1
            continue
        frame = item
        if be is not None and frame.image is not None:
            frame.result = be.process(frame.image, 0 if frame.stream == "overhead" else 1,
                                      frame.frame_id)
        base = ns_base.setdefault(frame.stream, frame.t_ns)
        t = epoch_base + (frame.t_ns - base) / 1e9         # SHARED.md §5: converted once

        if frame.stream == "overhead":
            # ponytail: raw blob count until Manan's tracker lands in W2. It is the
            # number the Friday demo needs and it is honest about what it measures.
            if t - last_pub.get("overhead", -1e9) >= 1.0:
                last_pub["overhead"] = t
                bus.publish(Event(t, cfg.store_id, "overhead", None, "occupancy",
                                  {"count": int(frame.result["num_blobs"])}))
        else:
            for roi_id, i in cfg.roi_index.items():
                if t - last_pub.get(roi_id, -1e9) >= 1.0:
                    last_pub[roi_id] = t
                    bus.publish(Event(t, cfg.store_id, "shelf", roi_id, "shelf_fill",
                                      {"fill": int(frame.result["roi_fill"][i])}))

        processed += 1
        if t - last_health >= 10.0:
            last_health = t
            clock.persist()
            write_health(health_file, dict(clock.health(), frames=processed,
                                           store_id=cfg.store_id, backend=backend,
                                           source=source, streams=list(streams)))
        if frames and processed >= frames:
            break
        if realtime:
            time.sleep(max(0.0, 1.0 / max(sources[frame.stream].fps, 1) - 0.001))

    stop.set()
    for src in sources.values():
        src.close()
    clock.persist()
    write_health(health_file, dict(clock.health(), frames=processed, store_id=cfg.store_id,
                                   backend=backend, source=source, streams=list(streams),
                                   running=False))
    log.info("run complete", extra={"frames": processed,
                                    "fps": round(processed / max(time.monotonic() - t_wall, 1e-6), 1)})
    return processed


def main(argv=None):
    p = argparse.ArgumentParser(description="Intelligent retail analytics on PYNQ-Z2")
    p.add_argument("--source", default="sim", choices=["sim", "file", "camera"])
    p.add_argument("--backend", default="sim", choices=["sim", "reference", "pl"])
    p.add_argument("--config", default="config/sim")
    p.add_argument("--frames", type=int, default=0, help="0 = run forever")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--health-file", default=None, help="JSON status for the backend /health")
    p.add_argument("--clock-state", default="/var/lib/retail/clock.json")
    rt = p.add_mutually_exclusive_group()
    rt.add_argument("--realtime", dest="realtime", action="store_true", default=True)
    rt.add_argument("--fast", dest="realtime", action="store_false")
    a = p.parse_args(argv)

    h = logging.StreamHandler()
    h.setFormatter(JsonLines())
    logging.basicConfig(level=logging.INFO, handlers=[h])

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    bus = Bus()
    bus.subscribe("occupancy", lambda e: log.info("occupancy", extra={"count": e.payload["count"]}))
    try:
        run(a.config, a.source, a.backend, a.frames, a.realtime and not a.headless,
            bus=bus, stop=stop, health_file=a.health_file, clock_state=a.clock_state)
    finally:
        bus.drain()
    return 0


if __name__ == "__main__":
    sys.exit(main())
