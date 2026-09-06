# Demo script (~7 min)

One Jetson Orin Nano on the table. Rohit owns the box, dashboard lead drives
the screen, nobody else touches a keyboard.

**Before the judges arrive**
- Jetson booted, `systemctl is-active retail-web` → `active`, dashboard open on
  the wall at `http://<jetson>:8000`.
- One run already processed and selected, so the dashboard has history in it
  and the Live View is looping the clip rather than sitting idle.
- Clips for the live upload already on the laptop, in a folder, named.
- Mock shelf lit, USB camera framed, all six facings stocked. Spare product in
  a pocket.
- Ethernet unplugged from the start. Say so.
- Fallback typed and not pressed:
  `python3 main.py --source file --backend reference --config data/runs/<id>/config --db data/retail.db --run-id fallback`

| # | Time | What happens | Who | Says |
|---|---|---|---|---|
| 1 | 0:30 | Title. The box is on the table, the cable is not in it | Rohit | One edge device. No cloud. No faces. Retail analytics for a shop with an unreliable line. |
| 2 | 1:00 | **Upload.** Drag two clips in, mark overhead and shelf, drag the entry line onto the door, flip the IN arrow, drag the zones onto the aisles, Process | Dashboard lead | Nobody configures this with a JSON file. The shopkeeper points at their own doorway. |
| 3 | 1:30 | Processing at camera speed: the bar, and the Live View with detections, zones and the door line drawn on the frame | Shopper analytics lead | That is the detector running on this box, right now. Boxes, never identities. |
| 4 | 1:30 | Dashboard fills in as it runs: occupancy, entries, zone dwell, heatmap, funnel | Dashboard lead | Every tile is the same run. Nothing here is a mock. |
| 5 | 1:30 | **Live shelf.** Judge removes a product. Fill drops, stock-out fires, planogram alert, duration timer, lost-revenue tick | Shelf lead invites a judge | Ask a judge to take something off the shelf. Then stop talking and let the alert land. |
| 6 | 0:45 | Queue tile and the staffing alert | Backend lead | Erlang C on the measured arrival rate — it fires before the queue is long, not after. |
| 7 | 1:00 | Accuracy vs ground truth, the timing table, DPDP slide | Rohit + Khushwant | Why this box and not a Pi: this table. Why no privacy problem: this slide. |

**The numbers to have on the slide** (`python -m tools.perf --backends reference yolo`):
CUDA detector vs the CPU chain, ms/frame and fps on the identical clip, plus
`tools/compare_backends.py` for how closely they agree on the boxes themselves.

**The one rehearsed recovery.** If the upload or the run stalls in segments
2–4: switch the run picker to the pre-processed run, keep talking. Do not debug
on stage. Segment 5 is the only live one and it is the one worth risking.

**Questions we will get**
- *Does it store video?* No. Frames are processed and dropped; the Live View
  holds one frame in memory to draw it and nothing is written. Only counts and
  events are persisted. See `docs/DPDP.md`.
- *Does it recognise people?* No face detection, no re-identification. Track
  IDs are per-process integers that reset on restart.
- *What if the internet is down?* The cable is already out. Cloud sync is
  optional and only ever sends aggregates.
- *Why not a Raspberry Pi?* The timing table. The detector needs the GPU to
  hold camera rate on two streams.
- *Is the POS real?* No, and say so before you are asked. A sale is a shopper
  standing in the checkout zone for five seconds — an observed dwell, not an
  invented transaction. The basket value is simulated from the planogram.
- *How accurate is the floor geometry?* Zones are drawn on the frame and mapped
  through a flat scale unless someone ran `geometry/tools/calibrate.py`. Dwell
  and zone counts are right either way; metres are only right after calibration.
