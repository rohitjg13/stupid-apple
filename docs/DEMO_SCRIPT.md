# Demo script (~7 min)

Rohit is on the board. Dashboard lead drives the screen. Nobody else touches a keyboard.

**Before the judges arrive**
- Board booted, `systemctl is-active retail` → `active`, dashboard open on the wall.
- `cat /var/lib/retail/health.json` → `clock_unsynced: false`, frames rising.
- Mock shelf lit, camera framed, all six facings stocked. Spare product in a pocket.
- Laptop on the same table, fallback command already typed, not pressed:
  `python3 main.py --source file --backend reference --config config/demo`

| # | Time | What happens | Who | Says |
|---|---|---|---|---|
| 1 | 0:30 | Title slide | Rohit | One board. No cloud. No faces. Retail analytics that runs in a shop with no internet. |
| 2 | 2:00 | Overhead footage playing through the PL; occupancy tile, entry count, zone headcount, heatmap filling in | Dashboard lead narrates, Rohit on board | This is recorded footage going through the FPGA, not a video player. Blobs panel is the live detection. |
| 3 | 1:30 | **Live shelf.** Judge removes a product. Fill drops, stock-out fires, planogram alert, duration timer, lost-revenue tick | Shelf lead invites a judge | Ask a judge to take something off the shelf. Then stop talking and let the alert land. |
| 4 | 1:00 | Queue clip: lane count, predicted vs actual wait, "open counter 2" | Backend lead | The prediction is M/M/c on measured arrival and service rate, not a guess. |
| 5 | 1:30 | Accuracy vs ground truth, silicon slide (utilization, PL vs CPU on the same clip), DPDP slide | Rohit + Khushwant | Why an FPGA and not a Pi: this table. |

**The one rehearsed recovery.** If anything in segment 2 or 4 stalls: say
"that's the board, we have the same pipeline on the laptop", press enter on
the fallback, keep talking. Do not debug on stage. Segment 3 is the only live
one and it is the one worth risking.

**Questions we will get**
- *Does it store video?* No. Frames are processed and dropped; only counts and
  events are persisted. See `docs/DPDP.md`.
- *Does it recognise people?* No face detection, no re-identification. Track
  IDs are per-process integers that reset on restart.
- *What if the internet is down?* It never needed it. Cloud sync is optional
  and only ever sends aggregates.
- *Why not just a Raspberry Pi?* Silicon slide: PL vs CPU ms/frame on the
  identical clip, from `tools/perf.py`.
