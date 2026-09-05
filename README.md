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

The frontend dashboard is built with Svelte and Vite.

```bash
npm install
npm run dev
```

## Layout

| Path | What |
|---|---|
| `pl/` | Contract A: the 856-byte `FrameResult`, register map, reference model |
| `core/` | Contract B: events, bus, config validation, RTC-less clock |
| `sources/` | sim, file, camera — all yield `Frame` |
| `geometry/` | image↔floor homography, zone raster, calibration tool |
| `tools/` | perf, backend comparison, accuracy evaluation, deploy, transcode |
| `docs/` | shared context, plans, demo script, test protocol, DPDP, risks, cuts |
| `src/` | Svelte frontend dashboard source code |

## Rules that are not negotiable

- `docs/SHARED.md` §4 and §5 are frozen contracts. Tests enforce them.
- Modules talk through `core/bus.py` events, never by importing each other.
- No frames, crops, faces or persistent track IDs are ever stored (`docs/DPDP.md`).
- Everything must work with no network.
