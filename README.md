# Intelligent Retail Analytics — Edge AI

Shopper analytics, shelf stock-outs and queue intelligence from ordinary store
cameras, computed **entirely on one edge device**. No cloud, no faces, no stored
video. Runs on a Jetson Orin Nano; the same tree still runs the PYNQ-Z2 build.

**Upload a few clips, press next, and it processes them.** That is the whole
operating procedure.

```bash
python server.py            #  ->  http://<device>:8000
```

---

## Quick start

```bash
pip install -e ".[dev]"                 # + ".[yolo]" for the detector
cd web && npm install && npm run build  # or use the web/dist already committed
cd .. && python server.py
```

Open the page, drop your videos in, mark each one **overhead** or **shelf**,
drag the entry line onto the doorway, press **Process**. The dashboard opens
when the run finishes.

No board, no camera, no footage? `python main.py --source sim --frames 300
--headless` runs the whole pipeline against a synthetic store.

### On the Jetson

```bash
sudo tools/setup_jetson.sh    # deps, YOLO weights, dashboard build, systemd unit
```

Do this once, with a network. Everything afterwards works offline: the weights
live in `models/`, the dashboard is static, and nothing in the request path
talks to the internet.

---

## What it does

| The brief asks for | Where it lives |
|---|---|
| Count entries and exits, footfall by hour and zone | `shopper/tripwire.py`, `backend/aggregates.py` |
| Dwell time near products and displays | `shopper/visits.py` |
| Movement heatmaps | `shopper/heatmap.py`, `geometry/zones.py` |
| Low-stock and out-of-stock detection | `shelf/fill.py`, `shelf/stockout.py` |
| Planogram compliance | `shelf/planogram.py`, `shelf/detector.py` |
| Replenishment alerts, real-time availability | `shelf/inventory_tracker.py`, `backend/alerts.py` |
| Queue length per counter, congestion prediction | `backend/pipeline.py`, `backend/queue_model.py` |
| "Open counter N" recommendation | `backend/staffing.py` (Erlang C) |
| All inference local, works through an outage | `pl/yolo.py`, `pl/reference.py`, no network in the request path |
| Anonymous detection, no PII | no frames, crops, faces or persisted track IDs — ever |
| Alerts, reports, KPI dashboard | `web/`, `backend/api.py` |
| POS / ERP integration | `backend/pos_stub.py`, `backend/pos_csv.py` (the stub is simulated — say so) |
| Multi-store, central monitoring | `store_id` on every event, `backend/cloud_sync.py` (aggregates only) |

## How a frame becomes a number

```
video / camera ─► sources/ ─► detector ─► FrameResult (856 B, frozen)
                                              │
             shopper/  ◄────── overhead ──────┤
             shelf/    ◄────── shelf ─────────┤
             backend/pipeline.py ◄── queue cells
                                              ▼
                            core/bus.py  ── Event (frozen envelope)
                                              ▼
                     backend/sink.py ─► SQLite ─► aggregates ─► /api ─► web/
```

Three detectors produce the same 856-byte `FrameResult`, so nothing downstream
knows or cares which one ran:

| `--backend` | Runs on | What it is |
|---|---|---|
| `yolo` | Jetson (CUDA) | YOLOv8 person detector. No warm-up, sees people who stand still. |
| `reference` | any laptop | OpenCV background subtraction — the golden model of the FPGA chain. |
| `pl` | PYNQ-Z2 | the FPGA overlay (`hls/`, `pl/driver.py`). |
| `sim` | anywhere | synthetic store, for development and as a demo fallback. |

`server.py` picks `yolo` when the box has it and `reference` otherwise.

## Layout

```
core/       bus.py events.py config.py clock.py     the frozen contracts
sources/    sim.py file.py camera.py                a Source yields Frames
pl/         contract.py regs.py reference.py yolo.py  detectors
geometry/   homography.py zones.py tools/calibrate.py
shopper/    tracker.py tripwire.py visits.py heatmap.py pipeline.py
shelf/      fill.py stockout.py planogram.py pick_detector.py pipeline.py
backend/    db.py sink.py aggregates.py alerts.py queue_model.py pipeline.py api.py
web/        the dashboard and the upload wizard (Svelte)
tools/      autoconfig.py evaluate.py perf.py setup_jetson.sh deploy.sh
hls/        the FPGA build
main.py     the CLI runner        server.py   the web app
```

## Config

A **clipset** is a folder describing one store: `store.yaml`, `zones.json`,
`rois.json`, `lanes.json`, `tripwires.json`, `planogram.json`,
`homography_overhead.npy`. `config/sim` is the hand-written one.

The wizard never asks you to write those. `tools/autoconfig.py` generates a
complete clipset from the uploaded videos — a flat floor scale, five zones, the
entry line you dragged, a rows × cols shelf grid, two queue lanes, one SKU per
facing. It is a starting point, not a survey:

```bash
python -m tools.autoconfig config/my_store --overhead a.mp4 b.mp4 --shelf s.mp4
python geometry/tools/calibrate.py    # a measured homography, when accuracy matters
```

## Rules that are not negotiable

- `docs/SHARED.md` §4 and §5 are frozen contracts. Tests enforce them.
- Modules talk through `core/bus.py` events, never by importing each other.
- No frames, crops, faces or persistent track IDs are ever stored.
- Everything must work with no network.

## Tests

```bash
pytest -q                                        # 498 tests
python main.py --source sim --frames 300 --headless
```
