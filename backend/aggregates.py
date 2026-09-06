"""Every dashboard query, expressed in SQL over the backend schema.

The dashboard lead only ever calls these functions; they never touch the DB
directly. Signatures match the plan (05-backend-queue.md W1 §2), so the API
router and dashboard can be wired in parallel.

Two things are computed in Python rather than SQL on purpose:

- percentiles, because SQLite has no percentile function on the board build
- Little's law and lost revenue, because both need a planogram/parameter the
  database does not hold

`kpi_rollup` pre-aggregates a 15-minute bucket into `kpi_15m`; it is what cloud
sync uploads and what keeps the historical charts under the 200 ms budget when
the raw tables have a demo day in them.
"""
from __future__ import annotations

import datetime
import math
import time

KPI_BUCKET_S = 900


class Aggregates:
    def __init__(self, db, run_id=None, planogram=None, zone_of=None):
        self.db = db
        self.run_id = run_id
        self.planogram = list(planogram or [])
        self.zone_of = zone_of

    def _rid(self, run_id):
        return run_id or self.run_id

    def _by_sku(self):
        return {p["sku"]: p for p in self.planogram}

    # ---- live -------------------------------------------------------------
    def live_summary(self, run_id=None):
        """The tiles at the top of the dashboard, in one round trip."""
        rid = self._rid(run_id)
        occ = self.db.query_one(
            "SELECT t, count FROM occupancy WHERE run_id = ? "
            "ORDER BY t DESC LIMIT 1", (rid,)) or {}
        now = occ.get("t") or time.time()
        t0, _ = _day_bounds(now)
        entries = self.db.query_one(
            "SELECT COUNT(*) AS n FROM tripwire WHERE run_id = ? AND dir = 'in' "
            "AND t >= ?", (rid, t0)) or {}
        exits = self.db.query_one(
            "SELECT COUNT(*) AS n FROM tripwire WHERE run_id = ? AND dir = 'out' "
            "AND t >= ?", (rid, t0)) or {}
        open_alerts = self.db.query_one(
            "SELECT COUNT(*) AS n FROM alert WHERE run_id = ? AND acked = 0",
            (rid,)) or {}
        open_stockouts = self.db.query_one(
            "SELECT COUNT(*) AS n FROM stockout WHERE run_id = ? AND t_end IS NULL",
            (rid,)) or {}
        queue = self.queue_status(rid)
        sales = self.db.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS amount FROM pos_txn "
            "WHERE run_id = ? AND t >= ?", (rid, t0)) or {}
        return {
            "run_id": rid,
            "t": occ.get("t"),
            "occupancy": occ.get("count", 0),
            "occupancy_t": occ.get("t"),
            "entries_today": entries.get("n", 0),
            "exits_today": exits.get("n", 0),
            "open_alerts": open_alerts.get("n", 0),
            "open_stockouts": open_stockouts.get("n", 0),
            "lanes": queue,
            "queue_total": sum(r["count"] or 0 for r in queue),
            "max_pred_wait_s": max([r["pred_wait_s"] or 0 for r in queue], default=0.0),
            "txns_today": sales.get("n", 0),
            "sales_today": sales.get("amount", 0.0),
        }

    # ---- footfall ---------------------------------------------------------
    def footfall_by_hour(self, day, run_id=None):
        """Entries per hour on a given unix day (local midnight..midnight)."""
        t0, t1 = _day_bounds(day)
        rows = self.db.query(
            "SELECT CAST((t - ?) / 3600 AS INTEGER) AS h, COUNT(*) AS n "
            "FROM tripwire WHERE run_id = ? AND dir = 'in' AND t >= ? AND t < ? "
            "GROUP BY h ORDER BY h", (t0, self._rid(run_id), t0, t1))
        return [{"hour": r["h"], "entries": r["n"]} for r in rows]

    def footfall_by_day(self, n, run_id=None, now=None):
        """Entries per day for the last n local days, oldest first.

        Local days, not `t / 86400`: the charts are labelled in store-local
        dates and a UTC bucket would split an Indian evening across two bars.
        Days with no entries are returned as zeroes so the chart has no holes.
        """
        now = time.time() if now is None else now
        t0, _ = _day_bounds(now)
        starts = [t0 - i * 86400 for i in range(int(n) - 1, -1, -1)]
        rid = self._rid(run_id)
        # One indexed COUNT per day rather than pulling a week of rows into
        # Python to bucket them. Each day's bounds are recomputed rather than
        # assuming 86400 s, so a DST change does not shift a whole bar.
        out = []
        for start in starts:
            lo, hi = _day_bounds(start)
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM tripwire WHERE run_id = ? "
                "AND dir = 'in' AND t >= ? AND t < ?", (rid, lo, hi)) or {}
            out.append({"day": _iso_day(lo), "day_start": lo,
                        "entries": row.get("n", 0)})
        return out

    # ---- dwell ------------------------------------------------------------
    def dwell_by_zone(self, t0, t1, run_id=None):
        rows = self.db.query(
            "SELECT zone, dwell_s FROM visit WHERE run_id = ? "
            "AND t_enter >= ? AND t_enter < ? ORDER BY zone",
            (self._rid(run_id), t0, t1))
        groups = {}
        for r in rows:
            groups.setdefault(r["zone"], []).append(r["dwell_s"])
        out = {}
        for zone, dwells in groups.items():
            dwells.sort()
            out[zone] = {"visits": len(dwells),
                         "median_dwell_s": _percentile(dwells, 0.5),
                         "p90_dwell_s": _percentile(dwells, 0.9)}
        return out

    def dwell_littles_law(self, t0, t1, run_id=None):
        """Store-wide mean dwell by Little's law: W = L / lambda.

        L is the mean occupancy over the window and lambda the entry rate over
        the same window, so this needs no tracker and no per-visitor identity —
        which is exactly why it survives the "store nothing personal" rule.
        Returns W in seconds, or None when nobody entered (lambda = 0).
        """
        rid = self._rid(run_id)
        occ = self.db.query_one(
            "SELECT AVG(count) AS l, COUNT(*) AS n FROM occupancy "
            "WHERE run_id = ? AND t >= ? AND t < ?", (rid, t0, t1)) or {}
        entries = self.db.query_one(
            "SELECT COUNT(*) AS n FROM tripwire WHERE run_id = ? AND dir = 'in' "
            "AND t >= ? AND t < ?", (rid, t0, t1)) or {}
        window = float(t1) - float(t0)
        L = occ.get("l")
        n_in = entries.get("n", 0)
        lam = (n_in / window) if window > 0 else 0.0
        W = (L / lam) if (L is not None and lam > 0) else None
        return {"mean_occupancy": L, "entries": n_in, "window_s": window,
                "lambda_per_s": lam, "mean_dwell_s": W,
                "samples": occ.get("n", 0)}

    # ---- heatmap ----------------------------------------------------------
    def heatmap(self, t0, t1, run_id=None):
        rows = self.db.query(
            "SELECT gx, gy, SUM(count) AS count FROM heatmap "
            "WHERE run_id = ? AND t_bucket >= ? AND t_bucket < ? "
            "GROUP BY gx, gy", (self._rid(run_id), t0, t1))
        return [{"gx": r["gx"], "gy": r["gy"], "count": r["count"]} for r in rows]

    # ---- stock-outs -------------------------------------------------------
    def stockout_durations(self, t0, t1, run_id=None):
        """Closed stock-outs in the window, with the revenue each one cost.

        lost_revenue = hours_empty * expected_sales_per_hour * unit_price, the
        plan's W2 §2 formula. Unknown SKUs price at zero rather than guessing.
        """
        rows = self.db.query(
            "SELECT roi, sku, t_start, t_end, (t_end - t_start) AS duration_s "
            "FROM stockout WHERE run_id = ? AND t_start >= ? AND t_start < ? "
            "AND t_end IS NOT NULL ORDER BY t_start",
            (self._rid(run_id), t0, t1))
        by_sku = self._by_sku()
        out = []
        for r in rows:
            row = dict(r)
            row["lost_revenue"] = _lost_revenue(row["duration_s"], by_sku.get(r["sku"]))
            out.append(row)
        return out

    def restock_stats(self, t0, t1, run_id=None):
        """How long it takes staff to refill: the W2 §2 restock-time stats."""
        rows = self.stockout_durations(t0, t1, run_id)
        if not rows:
            return {"n": 0, "median_s": None, "p90_s": None, "mean_s": None,
                    "total_lost_revenue": 0.0}
        durations = sorted(r["duration_s"] for r in rows)
        return {
            "n": len(durations),
            "median_s": _percentile(durations, 0.5),
            "p90_s": _percentile(durations, 0.9),
            "mean_s": sum(durations) / len(durations),
            "total_lost_revenue": round(sum(r["lost_revenue"] for r in rows), 2),
        }

    def replenishment_list(self, run_id=None, planogram=None, now=None):
        """Open stock-outs sorted by revenue rate (sales/h * unit price).

        Revenue rate is the sort key the plan asks for — it is what the shelf
        is losing per hour *right now*, so the highest one is what staff should
        walk to first. `lost_revenue_so_far` is shown alongside it as the cost
        already incurred.
        """
        rid = self._rid(run_id)
        rows = self.db.query(
            "SELECT rowid AS id, roi, sku, t_start FROM stockout WHERE run_id = ? "
            "AND t_end IS NULL ORDER BY t_start", (rid,))
        by_sku = {p["sku"]: p for p in (planogram if planogram is not None
                                        else self.planogram)}
        now = self._latest_t(rid) if now is None else now
        out = []
        for r in rows:
            p = by_sku.get(r["sku"], {})
            rate = (p.get("expected_sales_per_hour", 0) or 0) * (p.get("unit_price", 0) or 0)
            empty_s = max(0.0, now - r["t_start"]) if now is not None else 0.0
            out.append({"id": r["id"], "roi": r["roi"], "sku": r["sku"],
                        "name": p.get("name"), "t_start": r["t_start"],
                        "empty_s": empty_s, "revenue_rate": rate,
                        "lost_revenue_so_far": round(empty_s / 3600.0 * rate, 2)})
        out.sort(key=lambda x: (x["revenue_rate"], x["empty_s"]), reverse=True)
        return out

    def _latest_t(self, run_id):
        """Newest timestamp we have seen for this run: 'now' in clip time.

        Wall-clock `time.time()` would be wrong for a replayed clip or the
        backfill, both of which live in their own time base.
        """
        row = self.db.query_one(
            "SELECT MAX(t) AS t FROM (SELECT MAX(t) AS t FROM occupancy WHERE run_id = ? "
            "UNION ALL SELECT MAX(t) FROM shelf_fill WHERE run_id = ? "
            "UNION ALL SELECT MAX(t) FROM tripwire WHERE run_id = ?)",
            (run_id, run_id, run_id))
        return row["t"] if row and row["t"] is not None else None

    # ---- queue ------------------------------------------------------------
    def queue_status(self, run_id=None):
        """Latest queue estimate per lane."""
        rid = self._rid(run_id)
        # One index seek to the end of each lane rather than a GROUP BY over
        # the run: a demo day is ~40k rows per lane and the dashboard asks for
        # this every second. There are at most 16 lanes (pl/regs.py MAX_LANES).
        lanes = self.db.query(
            "SELECT DISTINCT lane FROM queue_estimate WHERE run_id = ? "
            "ORDER BY lane", (rid,))
        out = []
        for row in lanes:
            latest = self.db.query_one(
                "SELECT lane, t, count, pred_wait_s FROM queue_estimate "
                "WHERE run_id = ? AND lane = ? ORDER BY t DESC LIMIT 1",
                (rid, row["lane"]))
            if latest is not None:
                out.append(latest)
        return out

    def wait_pred_vs_actual(self, t0, t1, run_id=None, tolerance_s=None):
        """Join predictions with ground truth per lane. MAE is the W5 target.

        Matching is per lane and nearest-in-time; a truth sample with no
        prediction for its lane is skipped rather than matched to another
        lane's number, which would quietly flatter the MAE.
        """
        truth = self.db.query(
            "SELECT t, lane, count, actual_wait_s FROM queue_truth "
            "WHERE t >= ? AND t < ? ORDER BY t", (t0, t1))
        if not truth:
            # No ground truth loaded for this window — the normal case on the
            # live dashboard. Return before pulling a day of predictions that
            # would have nothing to be scored against.
            return {"mae_s": None, "count_mae": None, "n": 0, "rows": []}
        # Predictions only need to span the labelled range, which is a clip,
        # not the whole window the caller asked about.
        pad = tolerance_s if tolerance_s is not None else 60.0
        lo = max(t0, truth[0]["t"] - pad)
        hi = min(t1, truth[-1]["t"] + pad)
        pred = self.db.query(
            "SELECT t, lane, count, pred_wait_s FROM queue_estimate "
            "WHERE run_id = ? AND t >= ? AND t < ? ORDER BY t",
            (self._rid(run_id), lo, hi))
        by_lane = {}
        for p in pred:
            by_lane.setdefault(p["lane"], []).append(p)

        rows = []
        for tr in truth:
            candidates = by_lane.get(tr["lane"])
            if not candidates:
                continue
            best = min(candidates, key=lambda p: abs(p["t"] - tr["t"]))
            if tolerance_s is not None and abs(best["t"] - tr["t"]) > tolerance_s:
                continue
            rows.append({"t": tr["t"], "lane": tr["lane"],
                         "pred_count": best["count"], "actual_count": tr["count"],
                         "pred_wait_s": best["pred_wait_s"],
                         "actual_wait_s": tr["actual_wait_s"],
                         "error_s": best["pred_wait_s"] - tr["actual_wait_s"]})
        if not rows:
            return {"mae_s": None, "count_mae": None, "n": 0, "rows": []}
        mae = sum(abs(r["error_s"]) for r in rows) / len(rows)
        count_mae = sum(abs(r["pred_count"] - r["actual_count"]) for r in rows) / len(rows)
        return {"mae_s": mae, "count_mae": count_mae, "n": len(rows), "rows": rows}

    # ---- conversion -------------------------------------------------------
    def conversion_by_zone(self, t0, t1, run_id=None, zone_of=None):
        """Visits, conversions, rate and label per zone (see conversion.py).

        Both conversion signals are supplied: pos_txn `zone_hint` and the
        shelf_fill drop, so a zone with no checkout attribution can still
        convert off the shelf.
        """
        from backend.conversion import classify_visits, label_zone

        rid = self._rid(run_id)
        zone_of = zone_of if zone_of is not None else self.zone_of

        visits = [_Row(r["zone"], {"track_id": r["track_id"],
                                   "t_enter": r["t_enter"], "t_exit": r["t_exit"]},
                       r["t_enter"])
                  for r in self.db.query(
                      "SELECT track_id, zone, t_enter, t_exit FROM visit "
                      "WHERE run_id = ? AND t_enter >= ? AND t_enter < ?",
                      (rid, t0, t1))]
        txns = [_Row(None, {"zone_hint": r["zone_hint"]}, r["t"])
                for r in self.db.query(
                    "SELECT t, zone_hint FROM pos_txn WHERE run_id = ? "
                    "AND t >= ? AND t < ?", (rid, t0, t1 + 900.0))]
        # Shelf samples are pulled a minute wide either side: the drop that
        # converts a visit can land just outside the visit window.
        fills = [_Row(r["roi"], {"fill": r["fill"]}, r["t"])
                 for r in self.db.query(
                     "SELECT t, roi, fill FROM shelf_fill WHERE run_id = ? "
                     "AND t >= ? AND t < ? ORDER BY t", (rid, t0 - 60.0, t1 + 60.0))]

        rows = classify_visits(visits, txns, fills, zone_of=zone_of)
        out = {}
        for row in rows:
            d = out.setdefault(row["zone"], {"visits": 0, "conversions": 0,
                                             "rate": 0.0, "dwells": []})
            d["visits"] += 1
            d["conversions"] += int(row["converted"])
            d["dwells"].append(row["dwell_s"])
        for d in out.values():
            d["rate"] = d["conversions"] / d["visits"] if d["visits"] else 0.0
            dwells = sorted(d.pop("dwells"))
            d["median_dwell_s"] = _percentile(dwells, 0.5)
            d["label"] = label_zone(d["median_dwell_s"], d["rate"])
        return out

    # ---- alerts -----------------------------------------------------------
    def alerts(self, open_only=True, run_id=None, limit=200):
        q = ("SELECT rowid AS id, run_id, t, severity, rule, message, acked "
             "FROM alert WHERE run_id = ?")
        params = [self._rid(run_id)]
        if open_only:
            q += " AND acked = 0"
        q += " ORDER BY t DESC LIMIT ?"
        params.append(int(limit))
        return self.db.query(q, params)

    def ack_alert(self, alert_id, run_id=None):
        """Acknowledge one alert. Idempotent; returns the row, or None."""
        rid = self._rid(run_id)
        self.db.enqueue("UPDATE alert SET acked = 1 WHERE rowid = ? AND run_id = ?",
                        (int(alert_id), rid)).drain()
        return self.db.query_one(
            "SELECT rowid AS id, t, severity, rule, message, acked FROM alert "
            "WHERE rowid = ? AND run_id = ?", (int(alert_id), rid))

    def ack_all(self, run_id=None):
        rid = self._rid(run_id)
        self.db.enqueue("UPDATE alert SET acked = 1 WHERE run_id = ? AND acked = 0",
                        (rid,)).drain()
        return self.db.query_one(
            "SELECT COUNT(*) AS n FROM alert WHERE run_id = ? AND acked = 0", (rid,))

    # ---- multi-store ------------------------------------------------------
    def stores(self):
        """One real store plus any backfilled history stores."""
        rows = self.db.query("SELECT DISTINCT store_id FROM run ORDER BY store_id")
        return [r["store_id"] for r in rows] if rows else []

    def store_details(self):
        """Store list with the runs behind it, for the multi-store switcher."""
        rows = self.db.query(
            "SELECT store_id, run_id, started, clipset FROM run "
            "ORDER BY store_id, started")
        out = {}
        for r in rows:
            d = out.setdefault(r["store_id"], {"store_id": r["store_id"], "runs": []})
            d["runs"].append({"run_id": r["run_id"], "started": r["started"],
                              "clipset": r["clipset"]})
        return list(out.values())

    def runs(self):
        return self.db.query(
            "SELECT run_id, store_id, started, clipset FROM run ORDER BY started DESC")

    # ---- 15-minute pre-aggregation ----------------------------------------
    def kpi_rollup(self, t0, t1, run_id=None):
        """Fold [t0, t1) into `kpi_15m` rows, one bucket at a time.

        This is the only table cloud sync uploads, and the table the historical
        charts read on the board, so it is deliberately a small fixed set of
        scalar keys rather than anything shaped like a raw event.
        """
        rid = self._rid(run_id)
        b0, b1 = kpi_bucket(t0), kpi_bucket(t1 - 1e-9) if t1 > t0 else kpi_bucket(t0)
        written = 0
        for bucket in range(int(b0), int(b1) + 1, KPI_BUCKET_S):
            for key, value in self._kpis(rid, bucket, bucket + KPI_BUCKET_S).items():
                if value is None:
                    continue
                self.db.enqueue(
                    "INSERT INTO kpi_15m (run_id, t_bucket, key, value) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(run_id, t_bucket, key) "
                    "DO UPDATE SET value = excluded.value",
                    (rid, bucket, key, float(value)))
                written += 1
        self.db.drain()
        return written

    def _kpis(self, rid, t0, t1):
        one = self.db.query_one
        occ = one("SELECT AVG(count) AS mean, MAX(count) AS peak FROM occupancy "
                  "WHERE run_id = ? AND t >= ? AND t < ?", (rid, t0, t1)) or {}
        trip = one("SELECT SUM(dir = 'in') AS ins, SUM(dir = 'out') AS outs "
                   "FROM tripwire WHERE run_id = ? AND t >= ? AND t < ?",
                   (rid, t0, t1)) or {}
        pos = one("SELECT COUNT(*) AS n, SUM(amount) AS amount, SUM(items) AS items "
                  "FROM pos_txn WHERE run_id = ? AND t >= ? AND t < ?",
                  (rid, t0, t1)) or {}
        q = one("SELECT AVG(count) AS count, AVG(pred_wait_s) AS wait "
                "FROM queue_estimate WHERE run_id = ? AND t >= ? AND t < ?",
                (rid, t0, t1)) or {}
        alerts = one("SELECT COUNT(*) AS n FROM alert WHERE run_id = ? "
                     "AND t >= ? AND t < ?", (rid, t0, t1)) or {}
        visits = one("SELECT COUNT(*) AS n, AVG(dwell_s) AS dwell FROM visit "
                     "WHERE run_id = ? AND t_enter >= ? AND t_enter < ?",
                     (rid, t0, t1)) or {}
        little = self.dwell_littles_law(t0, t1, rid)
        return {
            "occupancy_mean": occ.get("mean"),
            "occupancy_peak": occ.get("peak"),
            "footfall_in": trip.get("ins") or 0,
            "footfall_out": trip.get("outs") or 0,
            "visits": visits.get("n") or 0,
            "dwell_mean_s": visits.get("dwell"),
            "dwell_littles_law_s": little["mean_dwell_s"],
            "pos_txns": pos.get("n") or 0,
            "pos_items": pos.get("items") or 0,
            "pos_amount": pos.get("amount") or 0.0,
            "queue_count_mean": q.get("count"),
            "queue_wait_mean_s": q.get("wait"),
            "alerts": alerts.get("n") or 0,
        }

    def kpi_series(self, t0, t1, key=None, run_id=None):
        """Read back the pre-aggregated buckets; what the history charts plot."""
        q = ("SELECT t_bucket, key, value FROM kpi_15m WHERE run_id = ? "
             "AND t_bucket >= ? AND t_bucket < ?")
        params = [self._rid(run_id), kpi_bucket(t0), t1]
        if key:
            q += " AND key = ?"
            params.append(key)
        return self.db.query(q + " ORDER BY t_bucket", params)


class _Row:
    """Event-shaped view of a DB row for conversion.py, which speaks Events."""

    __slots__ = ("zone_id", "payload", "t")

    def __init__(self, zone_id, payload, t):
        self.zone_id = zone_id
        self.payload = payload
        self.t = t


def kpi_bucket(t):
    """15-minute bucket (unix seconds) for a timestamp."""
    return int(math.floor(float(t) / KPI_BUCKET_S)) * KPI_BUCKET_S


def _lost_revenue(duration_s, item):
    if not item or duration_s is None:
        return 0.0
    rate = (item.get("expected_sales_per_hour", 0) or 0) * (item.get("unit_price", 0) or 0)
    return round(duration_s / 3600.0 * rate, 2)


def _percentile(sorted_values, q):
    """Nearest-rank percentile on an already sorted list. 0.0 when empty."""
    if not sorted_values:
        return 0.0
    rank = max(1, int(math.ceil(q * len(sorted_values))))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _day_bounds(t):
    """Unix second -> (local midnight, next local midnight)."""
    lt = time.localtime(t)
    midnight = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday)
    t0 = time.mktime(midnight.timetuple())
    return t0, t0 + 86400


def _iso_day(t):
    lt = time.localtime(t)
    return f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}"
