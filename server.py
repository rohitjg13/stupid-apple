"""The whole product behind one port: upload clips, press next, watch the store.

    python server.py            # http://<jetson>:8000

Three things live here and nothing else:

1. the upload wizard's API -- files in, a generated clipset out, a run started
   on a worker thread (`main.run`, the same entry point the CLI uses);
2. `/api/dashboard`, one composed document per poll, so the browser does not
   fan out into a dozen requests on a device that is busy doing inference;
3. the static dashboard from `web/dist`.

Everything analytical is imported, never reimplemented: `tools/autoconfig.py`
writes the config, `backend/aggregates.py` answers every question about the
data. No network calls leave the box.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import main as runner
from backend.aggregates import Aggregates
from backend.api import create_router
from backend.db import DB
from core.config import load_clipset, state_dir
from tools import autoconfig

log = logging.getLogger("server")

ROOT = Path(__file__).parent
DATA = state_dir()
WEB = ROOT / "web" / "dist"
MAX_UPLOAD_BYTES = 2 * 1024**3          # 2 GB per clip; an SD card is not infinite
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".mjpeg", ".mjpg", ".m4v", ".webm"}
HEATMAP_W, HEATMAP_H = 16, 12           # the dashboard's grid, in cells
STARTED = time.monotonic()

db = DB(DATA / "retail.db")
aggregates = Aggregates(db)
runs = {}                               # run_id -> progress dict, this process only
app = FastAPI(title="Intelligent Retail Analytics")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(create_router(aggregates))


# ---- uploads --------------------------------------------------------------
def safe_name(name: str) -> str:
    """Client-supplied filenames are hostile until proven otherwise."""
    stem = SAFE_NAME.sub("_", Path(name or "clip").name).lstrip(".") or "clip"
    if Path(stem).suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(400, f"{name!r} is not a video "
                                 f"({', '.join(sorted(VIDEO_SUFFIXES))})")
    return stem


def save(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"{dest.name} is over {MAX_UPLOAD_BYTES // 1024**3} GB")
            out.write(chunk)
    return dest


@app.post("/api/runs")
def create_run(files: list[UploadFile] = File(...), roles: list[str] = Form(default=[])):
    """Upload clips. `roles[i]` is 'overhead' or 'shelf'; default: first is overhead."""
    if not files:
        raise HTTPException(400, "no files uploaded")
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    media = DATA / "runs" / run_id / "media"

    videos = []
    for i, f in enumerate(files):
        role = roles[i] if i < len(roles) else ("overhead" if i == 0 else "shelf")
        if role not in ("overhead", "shelf"):
            raise HTTPException(400, f"unknown role {role!r}")
        path = save(f, media / f"{i:02d}_{safe_name(f.filename)}")
        fps, frames = autoconfig.probe(path)
        videos.append({"name": path.name, "path": str(path), "role": role,
                       "fps": fps, "frames": frames,
                       "seconds": round(frames / fps, 1) if frames else None})

    runs[run_id] = {"run_id": run_id, "state": "uploaded", "processed": 0, "total": 0,
                    "fps": 0.0, "error": None, "videos": videos, "backend": None}
    log.info("upload complete", extra={"run_id": run_id, "videos": len(videos)})
    return runs[run_id]


@app.get("/api/runs/{run_id}/preview")
def preview(run_id: str, role: str = "overhead"):
    """First frame of that role's first clip, letterboxed exactly as the pipeline sees it."""
    import cv2
    from pl.reference import letterbox

    run = runs.get(run_id) or _fail_unknown(run_id)
    clips = [v for v in run["videos"] if v["role"] == role]
    if not clips:
        raise HTTPException(404, f"no {role} clip in this run")
    cap = cv2.VideoCapture(clips[0]["path"])
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(422, f"cannot decode {clips[0]['name']}")
    ok, buf = cv2.imencode(".jpg", letterbox(img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


# ---- processing -----------------------------------------------------------
def default_backend() -> str:
    """YOLO on the Jetson, the OpenCV reference chain anywhere else."""
    from pl import yolo
    return "yolo" if yolo.available() else "reference"


@app.post("/api/runs/{run_id}/start")
def start(run_id: str, body: dict = Body(default={})):
    run = runs.get(run_id) or _fail_unknown(run_id)
    if run["state"] == "running":
        raise HTTPException(409, "already running")

    overhead = [v["path"] for v in run["videos"] if v["role"] == "overhead"]
    shelf = [v["path"] for v in run["videos"] if v["role"] == "shelf"]
    try:
        clipset = autoconfig.make_clipset(
            DATA / "runs" / run_id / "config", overhead=overhead, shelf=shelf,
            store_id=body.get("store_id") or "demo-01",
            shelf_grid=(int(body.get("shelf_rows", 2)), int(body.get("shelf_cols", 3))),
            door_line=body.get("door_line"),
            counters=int(body.get("counters", 2)),
            target_wait_s=float(body.get("target_wait_s", 180)))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    backend = body.get("backend") or default_backend()
    streams = tuple(s for s, clips in (("overhead", overhead), ("shelf", shelf)) if clips)
    if not streams:
        raise HTTPException(400, "upload at least one video")
    run.update(state="running", processed=0, total=0, error=None, backend=backend,
               clipset=str(clipset), started=time.time())
    threading.Thread(target=_work, args=(run, str(clipset), backend, streams),
                     name=f"run-{run_id}", daemon=True).start()
    return run


def _work(run, clipset, backend, streams):
    t0 = time.monotonic()

    def progress(processed, total):
        run.update(processed=processed, total=total,
                   fps=round(processed / max(time.monotonic() - t0, 1e-6), 1))

    try:
        runner.run(clipset, source="file", backend=backend, realtime=False,
                   streams=streams, db_path=str(DATA / "retail.db"),
                   run_id=run["run_id"], on_progress=progress,
                   clock_state=str(DATA / "clock.json"))
        run["state"] = "done"
    except Exception as e:                      # a bad clip must not kill the server
        log.exception("run failed", extra={"run_id": run["run_id"]})
        run.update(state="error", error=f"{type(e).__name__}: {e}")


@app.get("/api/runs")
def list_runs():
    """In-flight runs first, then anything already in the database."""
    rows = db.query("SELECT run_id, store_id, started, clipset FROM run ORDER BY started DESC")
    out = [dict(r, **{k: v for k, v in runs.get(r["run_id"], {}).items()
                      if k in ("state", "processed", "total", "fps", "backend")})
           for r in rows]
    seen = {r["run_id"] for r in rows}
    return [r for r in runs.values() if r["run_id"] not in seen] + out


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    return runs.get(run_id) or _fail_unknown(run_id)


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    """Drop the clips and every row. Video is never kept longer than it is needed."""
    if runs.get(run_id, {}).get("state") == "running":
        raise HTTPException(409, "still running")
    for table in ("run", "tripwire", "occupancy", "visit", "heatmap", "shelf_fill",
                  "stockout", "planogram_violation", "lane_occ", "queue_estimate",
                  "pos_txn", "alert", "kpi_15m"):
        db.enqueue(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
    db.drain()
    shutil.rmtree(DATA / "runs" / run_id, ignore_errors=True)
    runs.pop(run_id, None)
    return {"deleted": run_id}


def _fail_unknown(run_id):
    raise HTTPException(404, f"unknown run {run_id!r}")


# ---- the dashboard document ----------------------------------------------
@lru_cache(maxsize=32)
def _clipset(path):
    """Clipsets never change once written, so the 2 s poll re-reads nothing."""
    return load_clipset(path)


def _config_of(run_id):
    clipset = (runs.get(run_id) or {}).get("clipset")
    if not clipset:
        row = db.query_one("SELECT clipset FROM run WHERE run_id = ?", (run_id,))
        clipset = row["clipset"] if row else None
    return _clipset(clipset) if clipset and Path(clipset).is_dir() else None


def _state_of(fill, roi):
    if fill is None:
        return "unknown"
    if fill < roi.get("empty_below", 60):
        return "empty"
    if fill < roi.get("low_below", 110):
        return "low"
    return "partial" if fill < 180 else "full"


def _heatmap_grid(tiles):
    """Absolute 0.25 m cells -> the dashboard's HEATMAP_W x HEATMAP_H, 0..1."""
    grid = [0.0] * (HEATMAP_W * HEATMAP_H)
    if not tiles:
        return grid
    gxs = [t["gx"] for t in tiles] or [0]
    gys = [t["gy"] for t in tiles] or [0]
    span_x = max(max(gxs) - min(gxs) + 1, 1)
    span_y = max(max(gys) - min(gys) + 1, 1)
    for t in tiles:
        cx = int((t["gx"] - min(gxs)) / span_x * HEATMAP_W)
        # Floor Y grows away from the camera; the grid's first row is the top.
        cy = HEATMAP_H - 1 - int((t["gy"] - min(gys)) / span_y * HEATMAP_H)
        i = min(HEATMAP_H - 1, max(0, cy)) * HEATMAP_W + min(HEATMAP_W - 1, max(0, cx))
        grid[i] += t["count"]
    top = max(grid) or 1.0
    # sqrt, not linear: one aisle end always dominates a floor heatmap, and a
    # linear scale renders everywhere else as the same near-black blue.
    return [round((v / top) ** 0.5, 3) for v in grid]


def _spark(t0, t1, run_id, buckets=12):
    rows = db.query("SELECT t FROM tripwire WHERE run_id = ? AND dir = 'in'", (run_id,))
    out = [0] * buckets
    width = max((t1 - t0) / buckets, 1e-6)
    for r in rows:
        out[min(buckets - 1, int((r["t"] - t0) / width))] += 1
    return out


@app.get("/api/dashboard")
def dashboard(run_id: str = None):
    """Everything the dashboard renders, in one document, from one poll."""
    run_id = run_id or _latest_run_id()
    if run_id is None:
        return {"run_id": None, "empty": True}
    window = aggregates.run_window(run_id)
    t0 = window["t0"] or 0.0
    t1 = (window["t1"] or 0.0) + 1.0
    cfg = _config_of(run_id)
    plan = cfg.planogram if cfg else []
    roi_by_id = {r["id"]: r for r in cfg.rois} if cfg else {}
    checkout = cfg.checkout if cfg else {}
    by_sku = {p["sku"]: p for p in plan}
    by_facing = {p["facing"]: p for p in plan}

    totals = aggregates.footfall_totals(run_id)
    # Per shopper, not per visit row: one person browsing three aisles is three
    # visits, and a funnel whose second step outnumbers its first is nonsense.
    shoppers = db.query(
        "SELECT track_id, MAX(dwell_s) AS longest, SUM(dwell_s) AS total "
        "FROM visit WHERE run_id = ? GROUP BY track_id", (run_id,))
    dwells = [s["longest"] for s in shoppers if s["longest"] is not None]
    txns = db.query_one("SELECT COUNT(*) AS n, SUM(amount) AS amount "
                        "FROM pos_txn WHERE run_id = ?", (run_id,)) or {}
    browsed = sum(1 for s in shoppers if (s["longest"] or 0) >= 5.0)
    engaged = sum(1 for s in shoppers if (s["total"] or 0) >= 30.0)
    purchased = txns.get("n") or 0
    # Anyone already inside when the clip starts never crosses the door line.
    footfall = max(totals["entries"], len(shoppers))

    shelf = []
    for row in aggregates.shelf_state(run_id):
        item = by_facing.get(row["roi"], {})
        shelf.append({"id": row["roi"], "fill": int(row["fill"]),
                      "state": "empty" if row["stockout"]
                               else _state_of(row["fill"], roi_by_id.get(row["roi"], {})),
                      "sku": item.get("sku"), "name": item.get("name")})

    stockouts = []
    for row in db.query("SELECT roi, sku, t_start, t_end FROM stockout "
                        "WHERE run_id = ? ORDER BY t_start DESC LIMIT 20", (run_id,)):
        end = row["t_end"] if row["t_end"] is not None else t1
        rate = by_sku.get(row["sku"], {})
        hours = max(0.0, end - row["t_start"]) / 3600.0
        stockouts.append({
            "facing": row["roi"], "sku": row["sku"],
            "duration_s": round(end - row["t_start"], 1),
            "lost": round(hours * rate.get("expected_sales_per_hour", 0)
                          * rate.get("unit_price", 0), 2),
            "status": "resolved" if row["t_end"] is not None else "active"})

    zones = [{"name": zone, **stats}
             for zone, stats in sorted(aggregates.dwell_by_zone(t0, t1, run_id).items())]
    progress = runs.get(run_id, {})
    return {
        "run_id": run_id,
        "state": progress.get("state", "done"),
        "processed": progress.get("processed", 0),
        "total": progress.get("total", 0),
        "fps": progress.get("fps", 0.0),
        "backend": progress.get("backend") or "reference",
        "store_id": cfg.store_id if cfg else None,
        "target_wait_s": checkout.get("target_wait_s", 180),
        "counters": checkout.get("counters", 2),
        "t0": t0, "t1": t1,
        "occupancy": aggregates.live_summary(run_id)["occupancy"],
        "entries": totals["entries"], "exits": totals["exits"],
        "avg_dwell_s": round(sum(dwells) / len(dwells), 1) if dwells else 0.0,
        "zones": zones,
        "queue": aggregates.queue_status(run_id),
        "shelf": shelf,
        "stockouts": stockouts,
        "lost_revenue": aggregates.lost_revenue(t1, plan, run_id),
        "replenishment": aggregates.replenishment_list(run_id, plan)[:5],
        "alerts": db.query("SELECT t, severity, rule, message FROM alert "
                           "WHERE run_id = ? ORDER BY t DESC LIMIT 12", (run_id,)),
        "heatmap": _heatmap_grid(aggregates.heatmap(t0, t1, run_id)),
        "footfall_hourly": _hours_of_day(
            aggregates.footfall_by_hour(t1 or time.time(), run_id)),
        "system": _system(progress, cfg),
        "footfall_spark": _spark(t0, t1, run_id),
        "revenue": round(txns.get("amount") or 0.0, 2),
        "funnel": [
            {"label": "Footfall", "value": footfall, "pct": 100.0},
            {"label": "Browsed", "value": browsed, "pct": _pct(browsed, footfall)},
            {"label": "Engaged", "value": engaged, "pct": _pct(engaged, footfall)},
            {"label": "Purchased", "value": purchased, "pct": _pct(purchased, footfall)},
        ],
        "conversion_rate": _pct(purchased, footfall),
    }


def _hours_of_day(rows):
    """All 24 hours, zero-filled. The chart is "footfall by hour"; the hours
    with nobody in them are part of the answer."""
    got = {r["hour"]: r["entries"] for r in rows}
    return [{"hour": h, "entries": got.get(h, 0)} for h in range(24)]


def _system(progress, cfg):
    """The system tile: measured, never typed in."""
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mb = rss / 1024**2 if sys.platform == "darwin" else rss / 1024   # bytes vs KB
    load = os.getloadavg()[0] / (os.cpu_count() or 1) * 100
    fps = progress.get("fps") or 0.0
    up = int(time.monotonic() - STARTED)
    return {
        "latency_ms": round(1000.0 / fps, 1) if fps else 0.0,
        "cpu_pct": round(min(load, 100.0)),
        "memory_mb": round(mb),
        "streams": len(progress.get("videos") or []) or (2 if cfg else 0),
        "uptime": f"{up // 3600}h {up % 3600 // 60:02d}m" if up >= 3600 else f"{up // 60}m",
    }


def _pct(n, total):
    return round(100.0 * n / total, 1) if total else 0.0


def _latest_run_id():
    if runs:
        return sorted(runs, reverse=True)[0]
    row = db.query_one("SELECT run_id FROM run ORDER BY started DESC LIMIT 1")
    return row["run_id"] if row else None


# ---- static ---------------------------------------------------------------
if WEB.is_dir():
    if (WEB / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = WEB / path
        return FileResponse(target if path and target.is_file() else WEB / "index.html")
else:
    @app.get("/")
    def unbuilt():
        return HTMLResponse(
            "<h1>Dashboard not built</h1><p>Run <code>cd web && npm install && "
            "npm run build</code>, or <code>npm run dev</code> for the dev server "
            "on :5173.</p>", status_code=503)


def cli(argv=None):
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    h = logging.StreamHandler()
    h.setFormatter(runner.JsonLines())
    logging.basicConfig(level=logging.INFO, handlers=[h])
    uvicorn.run(app, host=a.host, port=a.port, log_config=None)


if __name__ == "__main__":
    cli()
