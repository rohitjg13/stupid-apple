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

-- Every t column is indexed; every aggregate filters on time.
CREATE INDEX IF NOT EXISTS idx_tripwire_t ON tripwire(t);
CREATE INDEX IF NOT EXISTS idx_occupancy_t ON occupancy(t);
CREATE INDEX IF NOT EXISTS idx_visit_enter ON visit(t_enter);
CREATE INDEX IF NOT EXISTS idx_visit_exit ON visit(t_exit);
CREATE INDEX IF NOT EXISTS idx_heatmap_bucket ON heatmap(t_bucket);
CREATE INDEX IF NOT EXISTS idx_shelf_fill_t ON shelf_fill(t);
CREATE INDEX IF NOT EXISTS idx_stockout_start ON stockout(t_start);
CREATE INDEX IF NOT EXISTS idx_stockout_end ON stockout(t_end);
CREATE INDEX IF NOT EXISTS idx_planogram_t ON planogram_violation(t);
CREATE INDEX IF NOT EXISTS idx_lane_occ_t ON lane_occ(t);
CREATE INDEX IF NOT EXISTS idx_queue_estimate_t ON queue_estimate(t);
CREATE INDEX IF NOT EXISTS idx_queue_truth_t ON queue_truth(t);
CREATE INDEX IF NOT EXISTS idx_pos_txn_t ON pos_txn(t);
CREATE INDEX IF NOT EXISTS idx_alert_t ON alert(t);
