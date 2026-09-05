# Shared Context — Intelligent Retail Analytics on PYNQ-Z2 (SIH 2026)

**Everyone pastes this file + their own plan into their agent (Claude Code / Claude) at the start of every session.** Commit it as `docs/SHARED.md`; the `CLAUDE.md` at the bottom goes in the repo root.

---

## 1. Ground truth about our situation
- One PYNQ-Z2. One USB camera. No supermarket.
- **Overhead / floor analytics and queue analytics run on recorded footage from the internet**, played through the real board pipeline.
- **Shelf / inventory is the one feature demoed live**: the USB camera points at a mock shelf; a judge removes a product; the alert fires.
- Two logical streams, `overhead` and `shelf`. Each is bound at startup (`store.yaml`) to a file or the camera. They time-share the single PL chain; the PS sets `STREAM_ID` before each DMA.
- Fully functional with no network. Cloud sync (aggregates only) is opportunistic.
- Nothing personal is stored: no frames, no crops, no persisted track IDs.

## 2. Team

| # | Person | Role | Plan |
|---|---|---|---|
| 1 | Khushwant | FPGA / hardware acceleration, `hls/`, `pl/driver.py` | `01-khushwant.md` |
| 2 | Rohit | System architecture, contracts, integration, floor geometry, cuts | `02-rohit.md` |
| 3 | Manan | Shopper analytics (tracking, counting, dwell, heatmap) | `03-manan.md` |
| 4 | *(name)* | Inventory & shelf intelligence + mock shelf + live demo | `04-shelf.md` |
| 5 | *(name)* | Backend, data layer, queue intelligence, POS stub, conversion | `05-backend-queue.md` |
| 6 | *(name)* | Dashboard, footage corpus, ground truth, deck, submission | `06-dashboard-footage.md` |

## 3. Timeline (6 weeks, Sep 7 → Oct 18, 2026; shift if the SIH date differs)

| Week | Dates | Must be true by Sunday night |
|---|---|---|
| W1 | Sep 7–13 | Both contracts frozen. Repo + CI. `--source sim` runs end to end on a laptop. Footage shortlist. Loopback IP started. Mock shelf materials bought. |
| W2 | Sep 14–20 | Loopback bitstream passes. `pl/reference.py` (OpenCV clone of the PL chain) gives FrameResults from footage on a laptop. Tracker v1. Shelf ROI tool + first captures. Dashboard live tiles on sim. DB schema live. |
| W3 | Sep 21–27 | HLS CV chain matches reference in C-sim. Tripwire/visits/heatmap on footage. Stock-out detection on captures. Queue regression + M/M/c. Historical dashboard. Ground truth for 3 clips. |
| W4 | Sep 28–Oct 4 | **Integration week.** Phase-2 bitstream on board. Footage → PL → analytics → DB → dashboard on the PYNQ. Live shelf on the board. |
| W5 | Oct 5–11 | Phase-3 accumulators. Tuning. Accuracy numbers vs ground truth. Cloud sync. Deck v1. DPDP write-up. |
| W6 | Oct 12–18 | Code freeze Wed. 5 board rehearsals + 2 laptop-fallback rehearsals. Deck final, docs, video. |

Friday 8 pm every week: integration call, everyone shows their piece running. No new features after W5. Rohit decides cuts.

## 4. Contract A — `FrameResult` (PL → PS), 856 bytes, frozen W1 (Rohit owns; Khushwant co-signs)

```c
struct Blob { uint16_t x, y, w, h; uint32_t area; };        // 12 bytes
struct FrameResult {
  uint32_t frame_id;          // echoed from input header
  uint16_t num_blobs;         // 0..64
  uint16_t flags;             // bit0 = bg model warm, bit1 = >64 blobs truncated
  Blob     blobs[64];         // FULL-RES 640x480 pixel coords, sorted by area desc
  uint8_t  roi_fill[64];      // Sobel edge density per shelf ROI, 0..255 (255 = full texture)
  uint8_t  lane_occupancy[16];// foreground % per queue cell, 0..100
};
```
```python
# pl/contract.py
BLOB_DT = np.dtype([("x","<u2"),("y","<u2"),("w","<u2"),("h","<u2"),("area","<u4")])
FRAME_RESULT_DT = np.dtype([("frame_id","<u4"),("num_blobs","<u2"),("flags","<u2"),
    ("blobs",BLOB_DT,(64,)),("roi_fill","u1",(64,)),("lane_occupancy","u1",(16,))])
assert FRAME_RESULT_DT.itemsize == 856
```
PS wrapper: `Frame(stream, frame_id, t_ns, result, image | None)`; `t_ns = time.monotonic_ns()` from the capture thread.

Input to PL: 640×480 BGR8 contiguous; sources letterbox anything else. AXI-Lite register map (truth in `hls/regs.h`, mirror in `pl/regs.py`):

| Offset | Name | Notes |
|---|---|---|
| 0x00 | CTRL | bit0 start, bit1 reset bg, bit2 dump mask, bit3 result+mask mode |
| 0x04 | STREAM_ID | 0 overhead, 1 shelf |
| 0x08 | MOG2_VAR_THRESH | Q8.8, default 16.0 |
| 0x0C | MOG2_LR | Q0.16, default 0.005 |
| 0x10 | MIN_BLOB_AREA | at 320×240, default 150 |
| 0x14 | MORPH_ITERS | 0..3 |
| 0x18 | LATENCY | cycles first-pixel-in → last-byte-out, read-only |
| 0x1C | DEBUG_STAGE | 0 gray, 1 mask, 2 morph |
| 0x100–0x2FF | ROI_TABLE | 64 × {x,y,w,h} u16, 320×240 scale = 512 bytes |
| 0x300–0x37F | LANE_TABLE | 16 × {x,y,w,h} u16 = 128 bytes |

Tables load from JSON at startup. **Adding a shelf facing never requires a rebuild.**

## 5. Contract B — event envelope (analytics → backend → dashboard), frozen W1 (Rohit owns)

```python
@dataclass
class Event:
    t: float            # unix epoch seconds (converted once from t_ns)
    store_id: str       # "demo-01"; multi-store view needs it
    stream: str         # "overhead" | "shelf" | "-"
    zone_id: str | None # zone / roi / lane id, or None
    event_type: str     # see table
    payload: dict       # event-specific, keys below
```

| event_type | emitted by | payload |
|---|---|---|
| `tripwire` | Manan | `{"dir": "in"|"out", "tripwire_id"}` |
| `occupancy` (1 Hz) | Manan | `{"count"}` |
| `visit` | Manan | `{"track_id", "t_enter", "t_exit"}` (zone_id = zone) |
| `heatmap` (every 10 s) | Manan | `{"t_bucket", "tiles": [[gx,gy,count],...]}` |
| `shelf_fill` (1 Hz per ROI) | Shelf lead | `{"fill"}` (zone_id = roi id) |
| `stockout_start` / `stockout_end` | Shelf lead | `{"sku", "facing"}` |
| `planogram_violation` | Shelf lead | `{"expected_sku", "observed": "empty"|"low"|"unknown"}` |
| `lane_occ` (2 Hz) | Backend lead (from raw) | `{"lane", "cells": [..16]}` |
| `queue_estimate` | Backend lead | `{"lane", "count", "pred_wait_s"}` |
| `pos_txn` | Backend lead (stub) | `{"lane", "items", "amount", "zone_hint"}` |
| `alert` | Backend lead (rules) | `{"severity", "rule", "message"}` |

Events go on an in-process bus (`core/bus.py`, Rohit): `bus.publish(event)`, `bus.subscribe(event_type | "*", fn)`. Backend persists; dashboard pushes over websocket; both subscribe. Nobody calls another module's functions directly across this boundary.

## 6. Coordinates & config
- Image `(u,v)` px, origin top-left, 640×480 on the PS side. Foot point of a blob: `(x + w/2, y + h)`.
- Floor `(X,Y)` metres, `config/homography_<stream>.npy` (3×3 image→floor). Zones are polygons in metres. Heatmap 0.25 m cells.
- `config/<clipset>/`: `store.yaml zones.json rois.json lanes.json tripwires.json planogram.json homography_*.npy floorplan.png NOTES.md`.

```jsonc
// store.yaml
store_id: demo-01
streams:
  overhead: {source: file, path: footage/overhead_01.mjpeg, fps: 15, loop: true}
  shelf:    {source: camera, device: /dev/video0, fourcc: YUYV, fps: 5}
checkout: {counters: 2, target_wait_s: 180}
pos: {source: stub}
cloud: {enabled: false, url: null}
// rois.json      [{"id":"A1","stream":"shelf","x":..,"y":..,"w":..,"h":..,"empty_below":60,"low_below":110}]
// planogram.json [{"facing":"A1","sku":"MAGGI-70G","name":"...","unit_price":14,"expected_sales_per_hour":6}]
// lanes.json     [{"id":"L1C0","lane":1,"cell":0,"x":..,"y":..,"w":..,"h":..}]
// tripwires.json [{"id":"door","stream":"overhead","p1":[u,v],"p2":[u,v],"in_dir":"down"}]
```

## 7. Sources / backends (Rohit)

| `--source` / `--backend` | Runs on | FrameResult from | Purpose |
|---|---|---|---|
| `sim` | anywhere | `sources/sim.py` | day-1 dev, demo fallback |
| `file` / `reference` | laptop | `pl/reference.py` (OpenCV clone of PL chain) | real footage before the bitstream; Khushwant's golden model |
| `file` / `pl` | board | overlay via `pl/driver.py` | the footage demo |
| `camera` / `pl` | board | overlay | the live shelf demo |

## 8. Repo
```
retail/
  CLAUDE.md docs/SHARED.md docs/plans/*.md docs/DEMO_SCRIPT.md docs/TEST_PROTOCOL.md docs/DPDP.md
  config/<clipset>/...   footage/ (download script, not committed) footage/groundtruth/*.json
  core/      bus.py events.py config.py
  sources/   base.py sim.py file.py camera.py
  pl/        contract.py regs.py driver.py reference.py        # driver.py = Khushwant, rest = Rohit
  hls/       src/ tb/ scripts/ + overlays/retail.bit retail.hwh (always a pair)
  shopper/   tracker.py tripwire.py visits.py heatmap.py pipeline.py
  shelf/     roi_calib.py fill.py stockout.py planogram.py dataset/ (captures + labels)
  backend/   schema.sql db.py aggregates.py alerts.py queue_model.py pos_stub.py conversion.py cloud_sync.py api.py
  dashboard/ server.py web/ static/
  geometry/  homography.py zones.py tools/calibrate.py
  tools/     transcode.sh download_footage.sh annotate.py evaluate.py perf.py compare_backends.py
  tests/     main.py retail.service
```
`main` always runs on `--source sim`. Branch per person. One reviewer per PR. CI = `pytest` + `python main.py --source sim --frames 300 --headless`.

## 9. Demo (W6, ~7 min)
1. Problem + one board, no cloud, no faces (30 s).
2. Overhead footage through the board: occupancy, entries, zone headcount, heatmap growing; blobs panel shows the detections (2 min).
3. **Live shelf**: camera on the mock shelf; judge removes a product; fill drops; stock-out + planogram alert; duration timer and lost-revenue tick (1.5 min).
4. Queue clip: count, predicted vs actual wait, "open counter 2" alert (1 min).
5. Accuracy slide (vs ground truth), silicon slide (utilization + PL vs CPU latency on the same clip), DPDP slide (1.5 min).
6. Fallback: laptop, `--source file --backend reference`, same dashboard.

## 10. Who blocks whom
- Rohit → all: contracts, bus, sim (W1); reference + sources (W2).
- Khushwant → integration: loopback (W2), phase 2 (Mon W4), phase 3 (Wed W5).
- Dashboard/footage lead → Manan, Backend, Khushwant: clips (W1), configs (W2), ground truth (W3).
- Shelf lead → Backend: `shelf_fill`/`stockout` events (W3); planogram.json (W2).
- Manan → Backend: shopper events (W3). Backend → Dashboard: `api.py` (W3).

---

## `CLAUDE.md` (repo root)
```markdown
# Retail Analytics on PYNQ-Z2 — agent instructions
- Read docs/SHARED.md first. FrameResult, the register map, the Event envelope, coordinates and config schemas are frozen; never change them silently.
- Python 3.10 on the board (PYNQ 3.x). No 3.11+ syntax. Board deps only: numpy, opencv-python-headless, fastapi, uvicorn, pyyaml, scipy. No torch, no pandas on the board.
- Everything runs with `python main.py --source sim` on a laptop without a board. `import pynq` only inside pl/driver.py.
- Modules talk only through core/bus.py Events, never by importing each other's internals.
- pytest for every module you touch; fixtures use sources/sim.py or a 5-second clip in tests/fixtures/.
- Never store frames, crops, faces or persistent track IDs. Track IDs restart at 0 per process.
- logging module, JSON lines at INFO. No print.
- Commit overlays/.bit and .hwh together. Only Khushwant's agent edits hls/ or pl/driver.py.
- Unsure about an interface? docs/SHARED.md, then ask. Never invent a field.
```
