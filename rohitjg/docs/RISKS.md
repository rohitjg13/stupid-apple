# Risk register

Rohit owns this. Reviewed at the Friday call. Ordered by expected damage.

| # | Risk | Likelihood | If it happens | Mitigation | Owner |
|---|---|---|---|---|---|
| 1 | `mog2_lite` does not fit in BRAM at K=2 | Medium | Phase 2 slips, no blobs on the board | Option B (running Gaussian, 154 KB) is already specified and `reference.py --bg-model rg` exists to match it. Decision by Fri W3, not later | Khushwant |
| 2 | No usable overhead footage found | Medium | Segments 2 and 4 have nothing to show | Shortlist due W1; if empty by W2, film our own corridor clip with the USB camera on a tripod | Dashboard/footage lead |
| 3 | DMA hang on stage | Medium | Demo dies mid-segment | Driver watchdog reloads the overlay; laptop fallback rehearsed twice; safety video recorded Fri W4 | Khushwant / Rohit |
| 4 | Live shelf segment is flaky under judges' lighting | Medium | The one live moment fails | Fixed lamp on the shelf, ROI thresholds tuned under that lamp, `empty_below`/`low_below` per facing in `rois.json`. If still flaky at W5 cut it to footage | Shelf lead |
| 5 | CCL truncates or merges people in crowded frames | Medium | Occupancy reads low, accuracy slide is weak | `flags.bit1` reports truncation honestly; report the number rather than hiding it | Khushwant / Manan |
| 6 | Integration slips because contracts drifted | Low | W4 is spent on glue instead of tuning | Contracts frozen W1 and enforced by tests; `core/events.py` rejects invented payload keys | Rohit |
| 7 | SD card fills, board stops mid-demo | Low | Total failure, looks amateur | Journal capped at 200 MB, no frames written to disk, health file is small and atomic | Rohit |
| 8 | Board has no RTC and boots at 1970 | High | Events land before existing rows, dashboard looks broken | `core/clock.py` continues from last persisted time and flags `clock_unsynced` | Rohit |
| 9 | Someone spends a week on Vitis AI / FINN | Low | A week gone for a slide | Explicitly out of scope; FINN is phase 4 and only if phase 3 shipped early | Rohit |
| 10 | Two people edit `docs/SHARED.md` §4 at once | Low | Merge conflict on the one file everyone depends on | Rohit owns the file; Khushwant co-signs but does not edit; `hls/regs.h` generated from `pl/regs.py` | Rohit |

## Standing decisions
- No new features after W5. Anything proposed later goes on the cut list, not the plan.
- Anything that only works with network access is optional by definition.
- If a number is not measured by `tools/`, it does not go on a slide.
