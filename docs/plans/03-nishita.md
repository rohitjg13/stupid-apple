# Plan 3 — Nishita: Shopper analytics (tracking, counting, dwell, heatmap)

Read `docs/SHARED.md` first. This module consumes `FrameResult.blobs` (from sim,
reference or PL — it never knows which) and Rohit's geometry, and publishes the
shopper events on the bus: `tripwire`, `occupancy`, `visit`, `heatmap`. Footfall
and the heatmap are the first things judges look at.

**No detector, no MobileNet-SSD, no ReID, no appearance embeddings.** A 650 MHz
A9 manages 1–2 fps on a detector, and appearance features break the privacy claim
in `docs/DPDP.md`. MOG2 blobs from the fabric plus IoU tracking is the design.

## Ownership

| Area | Files |
|---|---|
| Tracking | `shopper/tracker.py`, `shopper/kalman.py` |
| Pre-filter | `shopper/blob_prefilter.py` |
| Counting | `shopper/tripwire.py` |
| Visits | `shopper/visits.py` |
| Heatmap | `shopper/heatmap.py` |
| Glue | `shopper/pipeline.py` |
| Tuning | `shopper/params.py`, `config/<clipset>/tracker.yaml` |
| Eval | `shopper/metrics.py` (called by `tools/evaluate.py`), `tools/viz_tracks.py` |

## Interfaces

- **In**: `frame.blobs` (already sliced to `num_blobs`), full-res 640×480 px,
  sorted by area descending, no confidence score. `flags & FLAG_BG_WARM == 0`
  means do not count and do not confirm.
- **Geometry**: `image_to_floor(H, uv)`, `ZoneMap.zone_of(X, Y)`,
  `ZoneMap.heatmap_index(X, Y)`. Foot point is `(x + w/2, y + h)`.
- **Out**: Events per `SHARED.md` §5. Track ids are ints from 0 per process and
  appear only inside a `visit` payload.
- **Entry point**: `ShopperPipeline(cfg, bus).on_frame(frame, t)`, where `t` is
  the epoch time `main.py` already derived from `t_ns`. Never recomputed here.

## Things that are not obvious from the spec

1. **`blob["area"]` is full-resolution.** `pl/reference.py` writes
   `area * SCALE²` with `SCALE = 2`, so the `MIN_BLOB_AREA = 150` register
   default arrives here as **600**, and the ByteTrack high-score threshold is
   **1200**. Off by 4× puts every blob in the low-score pass.
2. **Area is the pseudo-score.** `FrameResult` has no confidence field, so
   "low score" and "small box" are the same thing. That geometrically couples
   score to IoU: a small blob can never overlap a large one much, so the second
   association pass only rescues people whose blob sits near the threshold. A
   third pass (leftover tracks against leftover big blobs at the loose threshold)
   matters more than the second — without it a track at 0.49 IoU goes lost while
   its blob spawns a duplicate identity.
3. **Sim and reference disagree before warm-up.** `sources/sim.py` emits blobs
   throughout warm-up; `pl/reference.py` zeroes the mask and emits none. Gating on
   `FLAG_BG_WARM` is what makes the three backends produce identical events.
4. **Occupancy publishes at 1 Hz unconditionally**, reporting 0 while cold. It is
   a heartbeat, not a detection; `tests/test_main.py` counts these events, and
   only ~50 of 150 frames in that test are warm.
5. **Merged blobs must not be measured.** When two shoppers touch, the background
   subtractor emits their *union*. Feeding that to whichever track wins the
   assignment wrecks its size and scale velocity so it matches neither person on
   separation. Both claimants coast on their own motion model instead.
6. **A claimant must be strict.** Claiming at the loose IoU meant two people
   merely walking near each other were treated as merged, starving real tracks of
   measurements: 9.6× id churn and 15.6 tracked people where there were 12.6.
   A claimant is now a confirmed track whose *own best* match is that blob, at
   the confident threshold.
7. **A background subtractor cannot see people who stand still**, which is
   exactly the shopper dwell time exists to measure. MOG2 learns a stationary
   person into the background as furniture: on a browsing clip it found *zero*
   blobs with four people standing in frame. The tracker compensates -- a track
   that loses its blob while stationary almost certainly did not leave, because
   leaving requires moving -- so it is held, frozen in place, and keeps being
   counted. Mean occupancy error on that clip went from 2.89 people to **0.00**.
   The other half of the fix is `MOG2_LR`, which is Rohit's and Khushwant's
   register: at the 0.005 default, people were visible in 136 of 368 frames of
   real supermarket footage; at 0.0005, 363 of 368.

   Holding still people has a mirror-image failure that real footage exposed
   immediately: a trolley wheel or a glossy basket that the background model
   keeps flagging gets a real match every frame, so it looks like a perfectly
   tracked shopper with zero speed and was reported as a 15 s dwell. What a
   shopper always has and furniture never does is that **they walked here**.
   A track is only held, and only counts, if it has been measured a body width
   from where it appeared for several *sustained* frames -- cumulative distance
   accumulates jitter into a fake journey, and peak distance is spoofed by one
   frame where the basket's blob merged with a passer-by. Anything confirmed for
   ~2 s without that is retired as furniture, and its position is remembered so
   the same blob does not respawn a fresh track every 26 frames.

   The filter is **off for the sim** (`config/sim/tracker.yaml`) and on by
   default for real footage. The sim renders clean rectangles with no furniture
   noise, so there it can only misfire -- and it did: 32 real walkers retired
   over 1000 s and entries −20 % → −24 %. Those were not churn-reborn tracks but
   **orphans**: a dwelling walker whose track was stolen by a passer-by, leaving
   them to spawn a fresh never-moved track. Track stealing is the root cause and
   is the same mechanism behind the 1.8× id churn; it is the next thing to fix
   in the tracker, not something the furniture filter should be papering over.
8. **Foot points are buffered from track birth**, not from confirmation, and
   replayed once when the track confirms. People cross a tripwire at the frame
   edge within a frame or two of appearing, long before a 3-hit confirmation.

## Measured on `config/sim`

Over 1000 s of stream time against the sim's own ground truth:

| Metric | Result | Target |
|---|---|---|
| Exit count error | **+2 %** | ±10 % |
| Entry count error | **−20 %** | ±10 % |
| Mean tracked vs actual shoppers | **12.64 vs 12.56** | — |
| Id churn (tracks per walker) | **1.8×** | — |
| Blob merging (raw → prefiltered) | 12.56 → **12.14** | ≈ no loss |

The entry gap is characterised, not mysterious: 9 % of sim walkers spawn within
one body width of a neighbour and share a blob, and 6 % head for a first target
below the door line. Two people entering shoulder to shoulder are one blob to any
appearance-free tracker, and the fix is a camera angle, not an algorithm.

**These are sim numbers.** The plan's accuracy targets are against annotated
footage, which does not exist yet. The prefilter in particular cannot be
validated until real clips land: sim blobs are clean rectangles with no
fragmentation, shadows or noise, which is exactly what the prefilter exists for.

## Bugs found in shared files (all fixed, all flagged to Rohit)

- `config/sim/tripwires.json` had `in_dir: "down"`. The homography is
  `Y = 6 − v/80`, so walking *into* the store is `v` *decreasing*. Every entry was
  reported as an exit.
- `sources/sim.py` could not demonstrate entry counting at all. Walkers stepped
  once before their first blob was rendered, so a spawn at `Y=0.2` (v=464) was
  already past the door line at v=460 by the time it was visible. They then left
  through the wall from the checkout, missing the door's u-span of 40–240
  entirely, and loitered on the exit once routed through it.

## Trying a clip

```
tools/try_video.sh vid/arcade1.mp4                       # any video, one command
tools/try_video.sh vid/arcade1.mp4 --bg-lr 0.0005 --morph 2
```

Runs the real chain (video → OpenCV background subtraction → tracker) with no
door, zones or floor calibration, writes `out/<name>_tracked.mp4` and `.json`,
prints who was present, how long they stood still and how far they walked to
get there, paints a heatmap over the frame, and opens the result.

Read the `seen in N/M frames` line first. Then the knobs, in the order they are
usually needed on real footage:

| Symptom | Knob |
|---|---|
| Shoppers vanish when they stand still | `--bg-lr 0.0005` (default 0.005 is ~10× too fast for retail) |
| Boxes on shelves, reflections, produce | `--morph 2` |
| People visible but unboxed (small in frame) | `--high-area 600` |
| Boxes flicker between ids | camera is too low; get it higher |

The door line and zone labels only appear with a per-camera `--config`; without
one they are deliberately not drawn, because the sim's floor plan painted over
real footage is a lie.

## Performance

`ShopperPipeline.on_frame` costs **0.62 ms mean / 1.08 ms p95** per frame on a
laptop, over 2900 warm sim frames averaging 10.6 blobs. The budget is 8 ms on the
board's 650 MHz A9, which is roughly an order of magnitude slower, so this is in
the right ballpark but is **not** a substitute for measuring on hardware.

## Still to do

- Re-tune the prefilter and tracker on real footage when the clipsets land; the
  merge rule is currently tuned to be conservative because fusing two shoppers
  loses one permanently while splitting one is recoverable.
- Profile on the board against the ≤ 8 ms/frame budget (needs hardware).
- `evaluate.py` against hand-annotated ground truth once `footage/groundtruth/`
  exists. `shopper/metrics.py` already accepts both circulating GT schemas, and
  writes `results/<clip>.json` plus a rolled-up `results/summary.json` so the
  accuracy slide is generated rather than typed.
- Optional same-view track re-link (a track ending at floor (X,Y) and a new one
  starting within 1.5 m / 2 s is the same journey). Deliberately not built: the
  plan says ship it only if it improves dwell MAE, and dwell MAE cannot be
  measured until annotated clips exist.
- The shopper slide for the deck (W6).

## Note for whoever writes the annotation tool

Two ground-truth schemas are in circulation: `tools/evaluate.py` and its tests
use `entries: [{t, dir}]` with `occupancy` as `[t, count]` pairs, while plan 06
specifies separate `entries`/`exits` and `{t, count}` dicts plus `people[].zones`.
`shopper/metrics.py` normalises both, so neither side is blocked, but it is worth
pinning one before anyone spends two hours per clip annotating.
