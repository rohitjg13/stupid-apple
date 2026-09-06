# Data protection (DPDP Act 2023)

The system is designed so that most of the Act does not apply to it, because it
does not collect personal data in the first place.

## What is never collected
- **No frames or crops are stored.** Video is processed and the buffer is
  reused. Nothing is written to disk, not even for debugging, unless a developer
  explicitly runs `--stage-outputs`, which writes to a scratch directory and is
  never enabled on the demo path.
- **No faces.** There is no face detection, no embedding, no recognition. The
  detector emits a class label ("person") and a rectangle, nothing else. On the
  FPGA path it is a background-subtraction mask at 320×240; on the Jetson path a
  YOLO person detector. Either way the frame is discarded the moment boxes are
  extracted, and every stage after that sees only `x, y, w, h`.
- **No appearance-based re-identification.** Nothing about how a person *looks*
  is ever computed, compared or stored: no embeddings, no colour histograms, no
  gait or face signature. A shopper who walks behind a shelf and out the other
  side is recognised as the same person within a few seconds, but only from
  physics — where they were, how fast they were going, how tall the box is. That
  re-link expires in seconds and cannot span a visit, a camera or a day.
- **No identity across visits.** Track IDs are per-process integers starting at 0
  and reset on every restart. The same person on two visits is two unrelated
  tracks, and nothing links a track to an identity.
- **No audio, no wifi/BLE probing, no POS card data.** The POS feed is a stub
  carrying basket size and amount only.

## What is stored
Counts and events only: occupancy per second, tripwire crossings, zone visit
durations, 0.25 m heatmap tallies, shelf fill percentages, stock-out intervals,
queue estimates. None of it identifies a person, and none of it can be turned
back into an image.

## Where it is stored
On the board, in a local SQLite database. Cloud sync is **off by default**
(`cloud.enabled: false` in `store.yaml`), optional, and sends aggregates only —
never raw events, never anything per-person. The system is fully functional with
no network at all.

## Notice and purpose
A shop deploying this puts up a notice at the entrance stating that anonymous
counting is in use. Purpose is limited to store operations: staffing, queue
management, restocking. There is no profiling and no automated decision that
affects any individual.

## Retention
Raw per-event rows are kept 30 days, then rolled up into hourly aggregates and
deleted. Nothing needs a deletion request mechanism because nothing is linked to
a person, but the retention job is in the backend and runs daily.

## Security
Board runs on the shop's LAN with no inbound ports open. The dashboard is
served on the LAN only. There are no credentials in the repo.

## Why this is the right design, not a compromise
A camera system that stored faces would need consent, a grievance officer, breach
reporting and deletion workflows. Counting blobs needs none of that and answers
every question a shop manager actually asks. The privacy property comes from the
architecture, not from a policy promise.
