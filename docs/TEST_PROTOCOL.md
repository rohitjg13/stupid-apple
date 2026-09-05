# Test protocol

## Every PR
`pytest -q` and `python main.py --source sim --frames 300 --headless`. Both run
in CI. One reviewer. If the diff touches `docs/SHARED.md` §4 or §5, Rohit reviews.

## Per module
Every module ships tests using `sources/sim.py` or a 5 s clip in `tests/fixtures/`.
No board required for anything except `pl/driver.py`.

## Accuracy (W5, vs `footage/groundtruth/*.json`)

| Metric | How | Target |
|---|---|---|
| Entry/exit count | tripwire events vs hand-labelled crossings on 3 clips | ±10 % |
| Occupancy | 1 Hz count vs labelled count, mean abs error | ≤ 1.5 people |
| Zone dwell | visit durations vs labels | ±20 % |
| Stock-out detection | labelled removals on the mock shelf | ≥ 95 % detected, ≤ 1 false alarm / 10 min |
| Queue wait | predicted vs actual on the queue clip | ±30 s |

## Backend agreement (W4)
`python3 -m tools.compare_backends --config config/demo --frames 300`
→ matched-blob IoU ≥ 0.7, blob-count agreement ≥ 90 %. The tool exits non-zero
if either fails, so it can gate the day's work.

## Performance (W4–W5, on the board)
`python3 -m tools.perf --backends reference pl --frames 200` on the raw `.npy`
clip. Targets: end-to-end ≥ 15 fps on the overhead clip, PL ≤ 5 ms/frame.

## Soak and recovery (W5)
1. 30 minutes continuous: no DMA hang, RSS flat within 10 %, no dropped-frame
   warnings accumulating.
2. Power pull ×3 mid-run: service returns by itself, DB opens, health file
   valid, `clock_unsynced` handled.
3. Camera unplug/replug ×3 during the shelf segment: stream resumes without a
   restart.
4. Boot to dashboard < 90 s, timed from power on.

## Rehearsal (W6)
5 full runs on the board, 2 on the laptop fallback, timed. A run counts only if
nobody touches a keyboard outside the script.
