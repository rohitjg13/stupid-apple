# Intelligent Retail Analytics on PYNQ-Z2

Store analytics — occupancy, zones, queues, shelf stock-outs — running entirely
on one PYNQ-Z2 with no cloud, no faces, and no stored video.

## Repository Layout by Branch

Work in this repository is organized by branch:

| Branch Directory | Description |
|---|---|
| `rohitjg/` | Core PYNQ-Z2 pipeline, geometry, sources, reference model, and CLI runner |
| `mridhula/` | Frontend analytics dashboard (Svelte + Vite) |
| `mrkr/` | Backend data layer, queue intelligence, POS stub, conversion, & cloud sync |
| `nishita/` | Shopper tracking, heatmap, tripwires, and YOLO backend detector |
| `pustak/` | Shelf monitoring, inventory tracking, pick detector, & stockout detection |

---

## Getting Started

### Core Pipeline (`rohitjg/`)
```bash
cd rohitjg
pip install -e . pytest
pytest -q
python main.py --source sim --frames 300 --headless
```

### Frontend Dashboard (`mridhula/`)
```bash
cd mridhula
npm install
npm run dev
```

---

## Rules that are not negotiable

- `rohitjg/docs/SHARED.md` §4 and §5 are frozen contracts. Tests enforce them.
- Modules talk through `core/bus.py` events, never by importing each other.
- No frames, crops, faces or persistent track IDs are ever stored.
- Everything must work with no network.
