# Retail Analytics on the edge — agent instructions

- Read docs/SHARED.md first. FrameResult, the register map, the Event envelope,
  coordinates and config schemas are frozen; never change them silently.
- **Everything deploys to a Jetson Orin Nano** (`--backend yolo`, what
  `server.py` picks when the box has it). The PYNQ-Z2 path (`--backend pl`,
  `hls/`) is a second target kept working, not the one being shipped: never
  make the Jetson path depend on it, and never assume an FPGA is present.
- Python 3.10 on the board (PYNQ 3.x). No 3.11+ syntax. Board deps only: numpy,
  opencv-python-headless, fastapi, uvicorn, python-multipart, pyyaml, scipy.
  `torch`/`ultralytics` are Jetson-only and imported lazily inside pl/yolo.py,
  the way `pynq` is imported only inside pl/driver.py.
- Everything runs with `python main.py --source sim` on a laptop with no
  hardware at all, and `python server.py` serves the whole product on one port.
- Modules talk only through core/bus.py Events, never by importing each other's
  internals. main.py and server.py are the only composition roots.
- pytest for every module you touch; fixtures use sources/sim.py or a short clip
  written in the test.
- Never store frames, crops, faces or persistent track IDs. Track IDs restart at
  0 per process.
- logging module, JSON lines at INFO. No print.
- Commit overlays/.bit and .hwh together. Only Khushwant's agent edits hls/ or
  pl/driver.py.
- web/ is a Svelte app; commit `web/dist` with the source so a box with no npm
  can still serve the dashboard.
- Unsure about an interface? docs/SHARED.md, then ask. Never invent a field.
