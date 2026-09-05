# Plan 1 — Khushwant: FPGA / hardware acceleration + `pl/driver.py`

Read `docs/SHARED.md` first. You own the overlay end to end **and** `pl/driver.py`, so Rohit calls the PL in one line: `driver.process(image, stream_id) -> FrameResult`. Nobody else touches `hls/` or the driver. Nobody waits on you before W4 by design; protect your calendar. Your one input is Rohit's `pl/reference.py` (W2), the golden model, and 10 golden frames per clip from the dashboard/footage lead.

Scope simplification to enjoy: video enters through the PS (file or USB), so there is no MIPI/CSI receiver, no debayer, no VDMA-to-HDMI. Just AXI DMA in, AXI DMA out, AXI-Lite for control.

## Toolchain (pin in W1, write `hls/README.md`)
- Vivado + Vitis HLS version matching the board's PYNQ image (PYNQ 3.0.x → 2022.1; check on the board). Same version for IP export and block design or the `.hwh` won't load.
- Vitis Vision Library (`xf::cv`) at the matching tag for `rgb2gray`, `resize`, `Sobel`, erode/dilate. **MOG2 and connected components you write yourself.**
- 100 MHz FCLK first. Everything scripted: `hls/scripts/run_hls.tcl`, `hls/scripts/build_bd.tcl`. Commit tcl + `overlays/retail.bit` + `retail.hwh`, never project dirs.

## The chain (one HLS `dataflow` region)
```
axis in 640x480 BGR
 → rgb2gray → resize 2:1 (320x240) ─┬→ mog2_lite → morph open ×N → ccl_stream → 64 blobs (×2 → full-res)
                                    └→ sobel |gx|+|gy| → roi_accum (64 ROIs) ; fg mask → lane_accum (16 cells)
 → pack FrameResult 856 B → axis out (TLAST on last beat)
```

## Build order (strict)

### Phase 1 — Pixel loopback (W1–W2)
1. IP `loopback`: axis in → axis out, AXI-Lite block with the full register map from `docs/SHARED.md` §4 already present (unused regs fine; **note the map now needs a 1 KB aperture** — see Constraints) **including the LATENCY cycle counter** (latch at first pixel in, last byte out). Rohit's perf tool and your silicon slide both read it.
2. Block design: Zynq PS → AXI DMA (MM2S + S2MM, simple mode) → IP → DMA. Export bit + hwh.
3. **`pl/driver.py` v0** (yours): `Overlay()`, two `pynq.allocate` buffers (in 640×480×3, out 856 B), `dma.sendchannel.transfer / recvchannel.transfer / wait`, `flush()` on input, `invalidate()` on output, register write helpers from `pl/regs.py`, a `timeout` + overlay reload watchdog, `process(image, stream_id)`, `load_tables(rois, lanes)`, `read_latency_cycles()`, `dump_mask()`. Keep `import pynq` inside this file; provide `driver.available()` so `main.py` can fall back.
4. Accept: 1,000 loopback frames bit-exact; then switch to the 856-byte result with `num_blobs=0` + `frame_id` echo, 1,000 frames correct; DMA time per frame recorded (expect ~2.5 ms).

Deliver Sun W2.

### Phase 2 — CV chain (W2–W4)
Develop each stage in C-sim against `pl/reference.py --stage-outputs` on real frames (`hls/tb/vectors/<clip>/*.png`, `hls/tb/golden/`).
1. `rgb2gray` + `resize` (AREA): MAE ≤ 2 vs reference gray-320.
2. **`mog2_lite`** — the risk item.
   - Do not implement K=5. Option A: K=2 Gaussians, 8-bit mean, 8-bit var (Q4.4), 8-bit weight → 6 B/px × 76,800 = 461 KB BRAM. Option B: running Gaussian (mean + var, 2 B/px = 154 KB) with a slow learning rate. Synthesize A first; if BRAM > 85 % with line buffers, ship B.
   - Match: `|p − mean| < VAR_THRESH·sqrt(var)`; update with `MOG2_LR`; foreground if no match; warm-up counter sets `flags.bit0` after 100 frames.
   - One background model in BRAM. Two won't fit. The shelf stream runs at 5 fps and re-warms whenever `STREAM_ID` changes → **announce this to Rohit and the shelf lead by Fri W3**; Rohit switches `reference.py` to the matching `--bg-model` and the shelf lead uses Sobel density (not the mask) for fill anyway, so shelf is unaffected.
   - Accept: mask IoU ≥ 0.8 vs reference after 200-frame warm-up on the same sequence.
3. Morph open ×`MORPH_ITERS` via `xf::cv` erode/dilate.
4. **`ccl_stream`**: single-pass run-length CCL. Per-row run table (≤128 runs), merge with previous row by overlap, union-find over ≤256 provisional labels, accumulate bbox + area, resolve at end of frame, filter `≥ MIN_BLOB_AREA`, sort by area (≤64, insertion sort is fine), ×2 coords, `flags.bit1` if truncated. Accept: count agreement ≥ 95 %, bbox IoU ≥ 0.9 vs `cv2.connectedComponentsWithStats`.
5. Explicit FIFO depths on every inter-stage stream (CCL's end-of-frame burst deadlocks default depths). Timing at 100 MHz. Cosim 30 frames. Export, rebuild BD.
6. **Debug path**: `CTRL.bit2` dumps the 320×240 stage selected by `DEBUG_STAGE` after the 856-byte result (`CTRL.bit3` mode, 77,656 B transfer). `driver.dump_mask()` returns it as a numpy array. Build this in phase 2; it will save you days in W4.

Deliver Mon W4. Accept on board (with Rohit): one walking person → one stable blob at ≥ 25 fps on the raw-frame benchmark; `compare_backends.py` blob IoU ≥ 0.7 vs reference.

### Phase 3 — ROI + lane accumulators (W4–W5)
1. Sobel 3×3 on gray-320, magnitude `|gx|+|gy|`.
2. `roi_accum`: 64 axis-aligned ROIs from ROI_TABLE. Precompute per-row active-ROI bitmask at row start to avoid 64 comparators per pixel. Mean magnitude → 0..255. Empty ROI → 0.
3. `lane_accum`: same structure over the foreground mask → 0..100 %.
4. Accept: `roi_fill` within ±10 and `lane_occupancy` within ±5 of reference on golden frames.

Deliver Wed W5.

### Phase 4 — FINN 64×64 shelf classifier: only if phase 3 shipped on time and BRAM allows. The shelf lead is collecting the training set regardless; if it's a no, the deck shows the FINN flow as future work.

## Constraints and traps
- BRAM before LUTs: 140 × 36 Kb = 630 KB. Check utilization after every csynth.
- TLAST on the last beat of the result or S2MM hangs; the driver watchdog masks it, fix it properly.
- Stale results = cache flush/invalidate in the driver, not your IP.
- Fixed-point formats for registers are in `docs/SHARED.md` §4; match Rohit's `pl/regs.py` exactly.
- **Register map changed 2026-09-05 (Rohit).** `docs/SHARED.md` §4 originally gave ROI_TABLE `0x100–0x1FF`, but 64 × {x,y,w,h} u16 is 512 bytes, not 256 — it overran LANE_TABLE. Corrected and frozen:
  `ROI_TABLE = 0x100` (0x100–0x2FF, 512 B) · `LANE_TABLE = 0x300` (0x300–0x37F, 128 B).
  Size the AXI-Lite aperture 0x400 (1 KB) — the old map only reached 0x240. `docs/SHARED.md` §4 and `pl/regs.py` are already updated; `pl/regs.py` is on `main` with tests — read it, don't retype the offsets. Write `hls/regs.h` from it and tell Rohit when it exists so he can add a test that diffs the `#define`s against `Reg` (two copies of one truth, nothing else catches drift).
- `pl/contract.py` (FRAME_RESULT_DT, 856 B, asserted in CI) and `pl/regs.py` (offsets, Q8.8/Q0.16 helpers, `pack_roi_table`/`pack_lane_table`) already exist. `driver.py` imports them; never re-derive the layout.
- Never GUI-edit the BD without regenerating `build_bd.tcl`.

## The silicon slide (yours, W5–W6)
Resource utilization table (LUT/FF/BRAM/DSP, % of 7020), 100 MHz, per-frame PL latency from LATENCY, and **PL vs CPU on the same footage**: run `tools/perf.py` on the board with `--backend pl` and `--backend reference` on the identical 20 s clip; report fps and ms/frame side by side. Add a block diagram PNG. Judges ask "why an FPGA and not a Pi"; this is the answer.

## Deliverables
| When | What |
|---|---|
| Sun W2 | Loopback bit/hwh, `pl/driver.py` v0, LATENCY reg, `hls/README.md` |
| Fri W3 | Phase-2 C-sim report; bg-model decision (A or B) announced |
| Mon W4 | Phase-2 bitstream + driver v1 (`process`, `load_tables`, `dump_mask`) |
| Wed W5 | Phase-3 bitstream |
| W6 | Utilization + timing + PL-vs-CPU table + block diagram |

## How to use your Claude agent
- Claude Code in `hls/` and `pl/driver.py`. Opener: "Read docs/SHARED.md §4 and docs/plans/01-khushwant.md. Toolchain: Vitis HLS <ver>, Vitis Vision <tag>. Task: <stage>. Write the C++ and a C-sim testbench that loads tb/vectors and diffs tb/golden with a tolerance, printing IoU/MAE."
- Strong delegations: run-length CCL (ask for a Python model proven equal to OpenCV on 50 masks first, then the HLS port); the per-row ROI bitmask; tcl scripts; the driver's DMA/watchdog boilerplate and its tests against a mock `pynq`; paste synthesis/timing reports and ask "what to fold first".
- Do yourself: the BRAM decision, BD wiring, first on-hardware bring-up, the silicon slide narrative.
