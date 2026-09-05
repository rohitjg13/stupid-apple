# Plan 2 — Rohit: System architecture, contracts, integration, floor geometry, cuts

Read `docs/SHARED.md` first. You own the two contracts everyone codes against (FrameResult, Event envelope), the sim, the reference model, the sources, the bus, the process/boot/offline layer, the floor-geometry layer, W4 integration, and every cut decision. No model deployment work exists on this project (Vitis AI doesn't target 7020; the pipeline is classic CV), so don't let anyone spend a week on it.

## Ownership
| Area | Files |
|---|---|
| Contract A | `pl/contract.py`, `pl/regs.py`, `docs/SHARED.md` §4 |
| Contract B | `core/events.py`, `core/bus.py`, `docs/SHARED.md` §5 |
| Sources | `sources/base.py sim.py file.py camera.py` |
| Reference | `pl/reference.py` (OpenCV clone of the PL chain; Khushwant's golden model) |
| Process | `main.py` (3 threads), `core/config.py`, `retail.service`, `tools/setup_board.sh`, `tools/deploy.sh`, offline-first behaviour |
| Geometry | `geometry/homography.py`, `geometry/zones.py`, `geometry/tools/calibrate.py` |
| Integration | W4 bring-up, `tools/perf.py`, `tools/compare_backends.py`, CI, `docs/INTEGRATION_LOG.md` |
| Program | weekly Friday call, cut list, risk register |

## Week by week

### W1 (Sep 7–13) — contracts, skeleton, sim
1. Repo, `CLAUDE.md`, `docs/SHARED.md`, `pyproject.toml`, pytest, GitHub Actions (`pytest` + `main.py --source sim --frames 300 --headless`).
2. `pl/contract.py`: dtype, `Frame` dataclass, `Blob.foot_point()`, `to_dict()`, test `itemsize == 856` + bytes round-trip. `pl/regs.py`: `IntEnum` offsets, `pack_roi_table`, `pack_lane_table`, Q8.8/Q0.16 helpers. Monday: 30 min with Khushwant to co-sign both.
3. `core/events.py` (`Event` dataclass, validation of `event_type`/payload keys per the table in `docs/SHARED.md` §5, `to_json`), `core/bus.py` (thread-safe publish/subscribe, wildcard, bounded per-subscriber queues so a slow dashboard can't stall analytics). Monday: 30 min with Manan, shelf lead, backend lead to co-sign payloads.
4. `sources/sim.py`: floor-plan random walkers (Poisson arrivals, dwell in zones, walk to checkout, exit), rendered to blobs via inverse homography; shelf events (a ROI's `roi_fill` drops 180→40 for 30–200 s); lane occupancy following a queue length. Scenarios in `config/sim/*.yaml`; `--seed`. Also `sources/sim.py --history-days 7` for the backend lead's backfill.
5. `main.py`: capture → `queue.Queue(maxsize=2)` → analytics → bus → (backend, dashboard). Flags `--source --backend --config --frames --headless --realtime/--fast`. SIGTERM-clean. `core/config.py` loads a clipset folder and validates every JSON against a schema (catches the shelf lead's and footage lead's typos early).
6. `config/sim/` sample clipset so every module has something to load.
7. Friday: `--source sim` prints `occupancy=3` lines and a blank dashboard page from the dashboard lead updates.

### W2 (Sep 14–20) — reference, sources, geometry
1. `pl/reference.py`: OpenCV clone of the PL chain, stage order fixed so Khushwant can match stage by stage: `cvtColor` → `resize(320×240, AREA)` → bg model (`--bg-model mog2` = `createBackgroundSubtractorMOG2(500, 16, False)`; `--bg-model rg` = numpy running Gaussian, added once Khushwant decides in W3) → `morphologyEx(OPEN, 3×3, MORPH_ITERS)` → `connectedComponentsWithStats` → filter/sort/cap/×2 → Sobel mean per ROI → 0..255 → fg % per lane cell. `--stage-outputs` writes golden PNGs. All parameters read from the same `regs` values the driver would write.
2. `sources/file.py`: MJPEG/MP4 via `VideoCapture`, letterbox to 640×480, throttle to `fps`, `loop`, one reader per stream. `sources/camera.py`: V4L2 YUYV 640×480, drop when queue full, reconnect on unplug (the live shelf camera will get bumped on stage).
3. `geometry/homography.py` (`image_to_floor`, `floor_to_image`, batch) + `geometry/tools/calibrate.py`: click 4 floor points on `frame.png`, type metres, save `.npy`, draw a reprojected 1 m grid for sanity. Run it with the footage lead on overhead clip 1 this week; they own the "which points and how many metres" judgement.
4. `geometry/zones.py`: `zone_of(X,Y)` via a precomputed 0.25 m raster (no shapely on the board), `heatmap_index(X,Y)`. Monday with Manan: he receives `(track_id, u, v, w, h)` after tracking and calls these himself in `shopper/pipeline.py`.
5. Friday: `--source file --backend reference --config config/overhead_01` shows blobs from real footage; Manan's tracker on top.

### W3 (Sep 21–27) — board prep, offline-first, perf
1. `tools/setup_board.sh` (PYNQ 3.x, deps, `retail.service` with `Restart=always`, log rotation), `tools/deploy.sh` (rsync + restart + `journalctl -f`), `docs/BOARD.md`.
2. Offline-first: `main.py` never blocks on network; time source is monotonic + RTC-less fallback (board has no RTC: on boot, if no NTP, continue from the last DB timestamp + 1 s and flag `clock_unsynced` in `/health`; the backend lead's cloud sync waits for a synced clock).
3. `tools/perf.py`: DMA ms, PL latency (LATENCY reg), analytics ms, end-to-end fps, per backend, as a table. `tools/transcode.sh --raw` variant (with footage lead) writes a 20 s clip as a raw BGR `.npy` memmap so the PL can be benchmarked without decode cost.
4. `tools/compare_backends.py`: same clip through `reference` and `pl`, per-frame blob count agreement and matched-blob IoU.
5. Integrate Khushwant's `driver.py` v0 into `main.py --backend pl` path (he wrote it; you wire it and own the fallback: if `driver.available()` is false, log loudly and refuse to start in `pl` mode rather than silently running reference on stage).

### W4 (Sep 28–Oct 4) — integration week (you run it, on the board daily)
1. Mon: phase-2 bitstream. `perf.py` on the raw clip. Fix DMA/alignment/cache issues with Khushwant.
2. Tue–Wed: overhead clip through PL; `compare_backends.py` ≥ 0.7 IoU, ≥ 90 % count agreement; tune via registers, never rebuilds.
3. Thu: shelf stream live on the board with the shelf lead's camera and ROI table; `roi_fill` vs reference.
4. Fri: full path on board, all three segments; screen-record it (safety video). Update `docs/INTEGRATION_LOG.md` every day.
5. Cut list v1: rank every remaining feature by demo value ÷ risk; publish it.

### W5 (Oct 5–11) — hardening, cuts
1. Power-pull test ×3 (service returns, DB opens). SD backup image. Boot-to-dashboard time < 90 s.
2. 30-minute soak: no DMA hangs, no memory growth, camera unplug/replug survives.
3. Review PRs from Manan / shelf / backend against `SHARED.md` for interface drift; your agent does the first pass.
4. Wed W5: cut decision. Candidates in order of first-to-go: FINN, cloud sync, multi-store view, planogram diff beyond empty/low, live segment if flaky.
5. Own the architecture slide and hand `perf.py` numbers to Khushwant for the silicon slide.

### W6 — freeze and rehearse
Freeze Wed. Run the demo script 5× on board + 2× on laptop fallback. You are on the board during the demo; the dashboard lead drives the screen.

## Acceptance criteria
- CI green on a laptop every week; sim / reference / PL emit identical Event schemas.
- Board: ≥ 15 fps end-to-end on the overhead MJPEG clip, PL ≤ 5 ms/frame, 0 DMA hangs in 30 min, boots to dashboard < 90 s, survives power pull.
- Calibration grid looks square; a known object (door ≈ 0.9 m) within 20 %.
- Cut list published W4, executed W5.

## How to use your Claude agent
- Claude Code with `CLAUDE.md`. Opener: "Read docs/SHARED.md and docs/plans/02-rohit.md. W<n>. Task: <item>. Write the test first."
- Strong delegations: contract dtypes + tests; the bus with bounded queues; the sim (give it the floor plan and scenarios); `reference.py` stage by stage with golden dumps; config schema validation; perf and compare tools; systemd/deploy scripts; PR interface-drift review ("compare this diff against docs/SHARED.md §4–5 and list violations").
- Do yourself: first DMA bring-up with Khushwant, every contract decision, the cut list, the integration log.
