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
