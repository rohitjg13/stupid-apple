# Plan 5 — Backend, data layer & queue intelligence (+ POS stub, conversion by zone)

Read `00-shared-context.md` first. You are the sink for every event on the bus and the source of every number on the dashboard. You own SQLite (WAL, batched writes: SD cards corrupt on power loss), KPI aggregation, the alerting rules, sync-when-available cloud push (aggregates only), queue intelligence (cell occupancy → headcount regression; M/M/c + exponential smoothing forecast; "open counter N" before the peak), the POS/ERP REST stub with synthetic timestamps, and conversion by zone.

## Ownership
| Area | Files |
|---|---|
| Storage | `backend/schema.sql`, `backend/db.py` (single writer thread, batched txns, retention, `run_id`) |
| Sink | `backend/sink.py` (bus subscriber `*` → inserts) |
| Aggregates | `backend/aggregates.py` (every query), `backend/api.py` (FastAPI router mounted by the dashboard server) |
| Alerts | `backend/alerts.py` (rules → `alert` events) |
| Queue | `backend/queue_model.py` (regression, M/M/c, smoothing), `backend/staffing.py` |
| POS | `backend/pos_stub.py` (REST adapter + synthetic generator), `backend/pos_csv.py` |
| Conversion | `backend/conversion.py` |
| Cloud | `backend/cloud_sync.py` (+ a 40-line receiver `tools/cloud_receiver.py` to demo it) |
| Eval | `tools/evaluate.py` queue half; `tools/backfill_history.py` |

## Schema (W1; review with Rohit and the dashboard lead)
```sql
PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
CREATE TABLE run (run_id TEXT PRIMARY KEY, store_id TEXT, started REAL, clipset TEXT);
CREATE TABLE tripwire (run_id, t REAL, dir TEXT);
CREATE TABLE occupancy (run_id, t REAL, count INTEGER);
CREATE TABLE visit (run_id, track_id INTEGER, zone TEXT, t_enter REAL, t_exit REAL, dwell_s REAL GENERATED ALWAYS AS (t_exit-t_enter) STORED);
CREATE TABLE heatmap (run_id, t_bucket INTEGER, gx INTEGER, gy INTEGER, count INTEGER, PRIMARY KEY(run_id,t_bucket,gx,gy));
CREATE TABLE shelf_fill (run_id, t REAL, roi TEXT, fill INTEGER);
CREATE TABLE stockout (run_id, roi TEXT, sku TEXT, t_start REAL, t_end REAL);
CREATE TABLE planogram_violation (run_id, t REAL, roi TEXT, expected TEXT, observed TEXT);
CREATE TABLE lane_occ (run_id, t REAL, lane INTEGER, cells BLOB);          -- 16 bytes
CREATE TABLE queue_estimate (run_id, t REAL, lane INTEGER, count REAL, pred_wait_s REAL);
CREATE TABLE queue_truth (clip TEXT, t REAL, lane INTEGER, count INTEGER, actual_wait_s REAL);
CREATE TABLE pos_txn (run_id, t REAL, lane INTEGER, items INTEGER, amount REAL, zone_hint TEXT);
CREATE TABLE alert (run_id, t REAL, severity TEXT, rule TEXT, message TEXT, acked INTEGER DEFAULT 0);
CREATE TABLE kpi_15m (run_id, t_bucket INTEGER, key TEXT, value REAL, PRIMARY KEY(run_id,t_bucket,key));  -- what cloud sync pushes
CREATE TABLE sync_state (key TEXT PRIMARY KEY, last_bucket INTEGER);
-- index every t column
```
Retention: raw `shelf_fill`/`lane_occ` rolled into 1-min means after 1 h, dropped after 7 d. One transaction per second. `run_id` per demo clip so each segment shows clean numbers (`tools/reset_run.py`).

## Week by week

### W1 (Sep 7–13) — storage, sink, stubs
1. `schema.sql`, `db.py` (writer thread fed by a queue, batched commits, PRAGMAs, no `VACUUM` at runtime), `sink.py` mapping every `event_type` in `00` §5 to an insert. Test: 10,000 events under `--source sim --fast`, zero drops, DB integrity check passes.
2. `aggregates.py` stubs with the exact signatures the dashboard lead needs, returning plausible fake data: `live_summary()`, `footfall_by_hour(day)`, `footfall_by_day(n)`, `dwell_by_zone(t0,t1)`, `conversion_by_zone(t0,t1)`, `heatmap(t0,t1)`, `stockout_durations(t0,t1)`, `replenishment_list()`, `queue_status()`, `wait_pred_vs_actual(t0,t1)`, `alerts(open_only)`, `stores()` (multi-store: returns one real + the backfilled sims).
3. `api.py`: FastAPI router `/api/...` over those functions; the dashboard lead mounts it.
4. `pos_stub.py` design: a REST adapter with the shape a real POS export would have (`GET /pos/transactions?since=`) backed by a synthetic generator: a transaction is emitted when a lane's estimated count drops by ≥ 1 (else on a timer when idle), `items ~ Poisson(3)`, amount from `planogram.json` prices, `zone_hint` sampled proportional to the last 10 min zone-dwell distribution (so conversion is correlated with dwell and the chart makes sense). Service rate μ is derived from stub timestamps exactly as it would be from a real POS. State clearly in the deck that POS is simulated.

### W2 (Sep 14–20) — Little's law, stock-out KPIs, alerts, backfill
1. Store-wide dwell by Little's law: `W = L/λ`, `L` = mean occupancy over the window, `λ` = entries/s over the same window; 15-min and daily. Unit test on a synthetic stream with a known answer.
2. Stock-out KPIs from the shelf lead's events: duration per facing, lost revenue `= hours_empty × expected_sales_per_hour × unit_price`, restock time stats. Replenishment list = open stock-outs sorted by revenue rate.
3. `alerts.py`: yaml rules → `alert` events with cooldowns: `stockout_start` (warning), stock-out > 5 min (critical), queue predicted wait > target (warning), occupancy > capacity (info), `clock_unsynced` (info). Ack endpoint.
4. `tools/backfill_history.py`: run the sim 7 virtual days in `--fast` with a daily footfall curve under `run_id='history'`, plus 2 extra fake stores (`store_id demo-02/03`) for the multi-store view.
5. Real aggregates replace stubs: footfall by hour/day, dwell by zone (median/p90), heatmap, stock-outs.

### W3 (Sep 21–27) — queue intelligence, conversion
1. **Queue count**: `count_lane = max(0, a·Σcells + b)` fitted by least squares on the queue-clip ground truth (from the dashboard/footage lead, Thu W3). Before that: `count ≈ 0.8 × cells with occ > 40`. EMA α=0.3. Publish `queue_estimate` every 2 s. Report R² and MAE.
2. **Forecast**: `λ` = arrivals to checkout/min = exponentially smoothed `tripwire out` rate (or lane-count increases if cleaner on the clip) over the last N=10 min; `μ` = 1 / mean inter-departure time from `pos_txn` per lane; `c` = open counters from `store.yaml`. Erlang C → `P(wait)`, `Wq = P(wait)/(c·μ−λ)`; unstable if `λ ≥ c·μ`. Blend with the naive `count/μ` (take the max) so the number is never absurdly low. Project λ 15 min ahead with Holt's linear smoothing to fire staffing alerts *before* the peak. Tests against textbook M/M/1 and M/M/2 values.
3. `staffing.py`: smallest `c` with `Wq ≤ target_wait_s` on the projected λ; message "open counter 2: λ=0.9/min, μ=0.7/min, projected wait 6.5 min"; 60 s cooldown.
4. **Conversion by zone**: visits with dwell ≥ 5 s; converted if followed within 15 min by a `pos_txn` whose `zone_hint` maps to the zone (real POS: SKU → planogram → zone) **or** by a `shelf_fill` drop ≥ 30 in a ROI mapped to that zone within 60 s of the visit. Conversion = converted/visits. Labels: high dwell + low conversion = "confused / review layout"; high dwell + high conversion = "engaged"; low dwell + low conversion = "ignored". The dashboard lead shows the labels.
5. `wait_pred_vs_actual`: join `queue_estimate` with `queue_truth` (loaded from `footage/groundtruth/<queue_clip>.json`); MAE.
6. Friday: every aggregate real on sim + reference.

### W4 (Sep 28–Oct 4) — board
1. Deploy with Rohit. DB growth/hour, write latency, power-pull ×3 with integrity check. Every aggregate < 200 ms on the board (`EXPLAIN QUERY PLAN`, pre-aggregate `kpi_15m`).
2. `reset_run.py` flow per demo segment.

### W5 (Oct 5–11) — cloud sync, calibration, numbers
1. `cloud_sync.py`: every 15 min, if network + synced clock, POST `kpi_15m` rows since `sync_state` to `cloud.url`; retry with backoff; never blocks the pipeline; **never sends raw events**. `tools/cloud_receiver.py` on the laptop prints what arrives; unplug the cable mid-demo to show nothing breaks and the backlog syncs on reconnect. (Cut candidate: if short on time, ship the receiver demo only.)
2. Fit the queue regression on the annotated clip; `evaluate.py` queue half: count MAE, wait MAE, alert lead time vs the annotated peak.
3. Business-impact numbers for the deck with sensitivities stated honestly (POS simulated, metres approximate).

### W6 — demo
Own the "queue & business logic" slide and the answers: "conversion without knowing what they bought" (POS line items → planogram → zone; simulated here, correlated with dwell), "why M/M/c not an LSTM" (three parameters, calibrates in a day, predicted-vs-actual chart proves it), "what leaves the store" (15-min KPIs only, on the cloud slide).

## Acceptance criteria
- No dropped events under `busy_evening --fast`; integrity OK after 3 power pulls.
- All aggregates < 200 ms on the board; DB < 50 MB after a demo day.
- Tests: Little's law, Erlang C, regression fit, conversion labels, alert cooldowns.
- Queue count MAE ≤ 1, wait MAE reported, staffing alert fires ≥ 2 min before the annotated peak.
- Backfilled 7-day history + 2 fake stores present at boot.

## How to use your Claude agent
- Opener: "Read docs/SHARED.md and docs/plans/05-backend-queue.md. Schema in backend/schema.sql. Task: <module>. pytest first with hand-computed expected values."
- Strong delegations: db layer + batched writer; every SQL aggregate (paste the schema, ask for SQL + a fixture-DB test + `EXPLAIN QUERY PLAN`); Erlang C and Holt smoothing with textbook tests; the POS stub REST adapter; the conversion classifier; backfill; cloud sync client/receiver.
- Do yourself: the conversion definition and labels, alert thresholds, planogram-price sanity, the honesty notes in the deck.
