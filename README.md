# Intelligent Retail Analytics on PYNQ-Z2

Store analytics — occupancy, zones, queues, shelf stock-outs — running entirely
on one PYNQ-Z2 with no cloud, no faces, and no stored video.

Read `docs/SHARED.md` first. Plans are in `docs/plans/`.

## Run it without a board

```bash
pip install -e . pytest
pytest -q
python main.py --source sim --frames 300 --headless
```

`--source sim` is a synthetic store: walkers arrive, browse, queue and leave,
and shelf facings empty out. Everything downstream can be developed against it.

## Real footage, still no board

```bash
python main.py --source file --backend reference --config config/<clipset>
```

`pl/reference.py` is an OpenCV clone of the FPGA pipeline in the same stage
order, so it is both the laptop path and the golden model the HLS is matched
against.

## On the board

See `docs/BOARD.md`. Short version: `tools/deploy.sh`, then
`systemctl start retail`.

## Dashboard (Svelte + Vite)

The frontend dashboard is located in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

## Directory Structure

| Path | Description |
|---|---|
| `frontend/` | Svelte + Vite web dashboard source & assets (`mridhula` branch) |
| `shopper/` | Shopper tracking, heatmaps, dwell times & tripwires (`nishita` branch) |
| `shelf/` | Stockout monitoring, pick detection & planogram compliance (`pustak` branch) |
| `backend/` | Database, API, POS stub, queue intelligence & cloud sync (`mrkr` branch) |
| `core/` | Event envelope, message bus, clock & config validation (`main` branch) |
| `pl/` | Contract A: 856-byte `FrameResult`, register maps, YOLO & reference backends |
| `sources/` | Video file, camera live feed, and synthetic simulation input sources |
| `geometry/` | Homography calibration & zone rasterization |
| `tools/` | Performance, accuracy evaluation, video rendering, and deployment scripts |
| `docs/` | Shared documentation, architectural plans, demo scripts, DPDP, and risks |

## Rules that are not negotiable

- `docs/SHARED.md` §4 and §5 are frozen contracts. Tests enforce them.
- Modules talk through `core/bus.py` events, never by importing each other.
- No frames, crops, faces or persistent track IDs are ever stored (`docs/DPDP.md`).
- Everything must work with no network.
