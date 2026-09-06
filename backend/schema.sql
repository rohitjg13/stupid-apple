-- Backend storage schema (docs/plans/05-backend-queue.md).
-- SQLite WAL + synchronous=NORMAL are set in db.py; this file is just DDL.
-- One transaction per second, never VACUUM at runtime.

CREATE TABLE IF NOT EXISTS run (
  run_id TEXT PRIMARY KEY,
  store_id TEXT,
  started REAL,
  clipset TEXT
);

CREATE TABLE IF NOT EXISTS tripwire (
  run_id TEXT,
  t REAL,
  dir TEXT
);

CREATE TABLE IF NOT EXISTS occupancy (
  run_id TEXT,
  t REAL,
  count INTEGER
);

CREATE TABLE IF NOT EXISTS visit (
  run_id TEXT,
  track_id INTEGER,
  zone TEXT,
  t_enter REAL,
  t_exit REAL,
  dwell_s REAL GENERATED ALWAYS AS (t_exit - t_enter) STORED
);

CREATE TABLE IF NOT EXISTS heatmap (
  run_id TEXT,
  t_bucket INTEGER,
  gx INTEGER,
  gy INTEGER,
  count INTEGER,
  PRIMARY KEY (run_id, t_bucket, gx, gy)
);

CREATE TABLE IF NOT EXISTS shelf_fill (
  run_id TEXT,
  t REAL,
  roi TEXT,
  fill INTEGER
);

CREATE TABLE IF NOT EXISTS stockout (
  run_id TEXT,
  roi TEXT,
  sku TEXT,
  t_start REAL,
  t_end REAL
);

CREATE TABLE IF NOT EXISTS planogram_violation (
  run_id TEXT,
  t REAL,
  roi TEXT,
  expected TEXT,
  observed TEXT
);

CREATE TABLE IF NOT EXISTS lane_occ (
  run_id TEXT,
  t REAL,
  lane INTEGER,
  cells BLOB          -- 16 bytes
);

CREATE TABLE IF NOT EXISTS queue_estimate (
  run_id TEXT,
  t REAL,
  lane INTEGER,
  count REAL,
  pred_wait_s REAL
);

CREATE TABLE IF NOT EXISTS queue_truth (
  clip TEXT,
  t REAL,
  lane INTEGER,
  count INTEGER,
  actual_wait_s REAL
);

CREATE TABLE IF NOT EXISTS pos_txn (
  run_id TEXT,
  t REAL,
  lane INTEGER,
  items INTEGER,
  amount REAL,
  zone_hint TEXT
);

CREATE TABLE IF NOT EXISTS alert (
  run_id TEXT,
  t REAL,
  severity TEXT,
  rule TEXT,
  message TEXT,
  acked INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kpi_15m (
  run_id TEXT,
  t_bucket INTEGER,
  key TEXT,
  value REAL,
  PRIMARY KEY (run_id, t_bucket, key)
);

CREATE TABLE IF NOT EXISTS sync_state (
  key TEXT PRIMARY KEY,
  last_bucket INTEGER
);

-- Indexing follows the shape of every aggregate: `WHERE run_id = ? AND t
-- BETWEEN ? AND ?`. A bare index on `t` is not enough once the 7-day backfill
-- and the demo run share a time range in one file — SQLite then walks every
-- store's rows and filters run_id per row. The composites below are what keep
-- the aggregates inside the 200 ms board budget.
--
-- The superseded single-column indexes are dropped rather than left in place:
-- on an SD card every redundant index is write amplification on the hot path.
DROP INDEX IF EXISTS idx_tripwire_t;
DROP INDEX IF EXISTS idx_occupancy_t;
DROP INDEX IF EXISTS idx_visit_enter;
DROP INDEX IF EXISTS idx_visit_exit;
DROP INDEX IF EXISTS idx_heatmap_bucket;
DROP INDEX IF EXISTS idx_stockout_start;
DROP INDEX IF EXISTS idx_stockout_end;
DROP INDEX IF EXISTS idx_planogram_t;
DROP INDEX IF EXISTS idx_queue_estimate_t;
DROP INDEX IF EXISTS idx_pos_txn_t;
DROP INDEX IF EXISTS idx_alert_t;

CREATE INDEX IF NOT EXISTS idx_occupancy_run_t   ON occupancy(run_id, t);
CREATE INDEX IF NOT EXISTS idx_tripwire_run_t    ON tripwire(run_id, dir, t);
CREATE INDEX IF NOT EXISTS idx_visit_run_enter   ON visit(run_id, t_enter);
CREATE INDEX IF NOT EXISTS idx_stockout_run      ON stockout(run_id, t_start);
CREATE INDEX IF NOT EXISTS idx_planogram_run_t   ON planogram_violation(run_id, t);
CREATE INDEX IF NOT EXISTS idx_queue_est_run     ON queue_estimate(run_id, lane, t);
CREATE INDEX IF NOT EXISTS idx_pos_txn_run_t     ON pos_txn(run_id, t);
CREATE INDEX IF NOT EXISTS idx_alert_run         ON alert(run_id, acked, t);
CREATE INDEX IF NOT EXISTS idx_shelf_fill_run_t  ON shelf_fill(run_id, t);
CREATE INDEX IF NOT EXISTS idx_lane_occ_run_t    ON lane_occ(run_id, t);

-- heatmap and kpi_15m are covered by their primary keys. A (run_id, gx, gy)
-- index was tried for the per-cell GROUP BY and measured no faster: SQLite
-- still builds a temp B-tree, so it was pure write cost. What actually makes
-- that query cheap is rolling the 10-second buckets up (see db.rollup).
-- Retention and rollup delete by time across every run, so shelf_fill and
-- lane_occ keep a bare `t` index as well; queue_truth is keyed by clip, not run.
CREATE INDEX IF NOT EXISTS idx_shelf_fill_t ON shelf_fill(t);
CREATE INDEX IF NOT EXISTS idx_lane_occ_t ON lane_occ(t);
CREATE INDEX IF NOT EXISTS idx_queue_truth_t ON queue_truth(clip, t);
