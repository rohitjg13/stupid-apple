"""Every dashboard query, expressed in SQL over the backend schema.

The dashboard lead only ever calls these functions; they never touch the DB
directly. Signatures match the plan (05-backend-queue.md W1 §2), so the API
router and dashboard can be wired in parallel.
"""
from __future__ import annotations

import math


class Aggregates:
    def __init__(self, db, run_id=None):
        self.db = db
        self.run_id = run_id

    def _runs(self, run_id):
        return (run_id or self.run_id,)

    # ---- live -------------------------------------------------------------
    def live_summary(self, run_id=None):
        occ = self.db.query_one(
            "SELECT t, count FROM occupancy WHERE run_id = ? "
            "ORDER BY t DESC LIMIT 1", self._runs(run_id)) or {}
        return {
            "occupancy": occ.get("count", 0),
            "occupancy_t": occ.get("t"),
        }

    # ---- footfall ---------------------------------------------------------
    def footfall_by_hour(self, day, run_id=None):
        """Entries per hour on a given unix day (local midnight..midnight)."""
        t0, t1 = _day_bounds(day)
        rows = self.db.query(
            "SELECT CAST((t - ?) / 3600 AS INTEGER) AS h, COUNT(*) AS n "
            "FROM tripwire WHERE run_id = ? AND dir = 'in' AND t >= ? AND t < ? "
            "GROUP BY h ORDER BY h", (t0, run_id or self.run_id, t0, t1))
        return [{"hour": r["h"], "entries": r["n"]} for r in rows]

    def footfall_by_day(self, n, run_id=None):
        """Entries per day for the last n days."""
        rows = self.db.query(
            "SELECT CAST(t / 86400 AS INTEGER) AS d, COUNT(*) AS n "
            "FROM tripwire WHERE run_id = ? AND dir = 'in' "
            "GROUP BY d ORDER BY d DESC LIMIT ?", (run_id or self.run_id, n))
        return [{"day": r["d"], "entries": r["n"]} for r in rows]

    # ---- dwell ------------------------------------------------------------
    def dwell_by_zone(self, t0, t1, run_id=None):
        rows = self.db.query(
            "SELECT zone, dwell_s FROM visit WHERE run_id = ? "
            "AND t_enter >= ? AND t_enter < ? ORDER BY zone",
            (run_id or self.run_id, t0, t1))
        groups = {}
        for r in rows:
            groups.setdefault(r["zone"], []).append(r["dwell_s"])
        out = {}
        for zone, dwells in groups.items():
            dwells.sort()
            median = dwells[len(dwells) // 2] if dwells else 0.0
            p90 = dwells[min(len(dwells) - 1, int(math.ceil(0.9 * len(dwells)) - 1))] if dwells else 0.0
            out[zone] = {"visits": len(dwells), "median_dwell_s": median,
                         "p90_dwell_s": p90}
        return out

    # ---- heatmap ----------------------------------------------------------
    def heatmap(self, t0, t1, run_id=None):
        rows = self.db.query(
            "SELECT gx, gy, SUM(count) AS count FROM heatmap "
            "WHERE run_id = ? AND t_bucket >= ? AND t_bucket < ? "
            "GROUP BY gx, gy", (run_id or self.run_id, t0, t1))
        return [{"gx": r["gx"], "gy": r["gy"], "count": r["count"]} for r in rows]

    # ---- stock-outs -------------------------------------------------------
    def stockout_durations(self, t0, t1, run_id=None):
        rows = self.db.query(
            "SELECT roi, sku, t_start, t_end, (t_end - t_start) AS duration_s "
            "FROM stockout WHERE run_id = ? AND t_start >= ? AND t_start < ? "
            "AND t_end IS NOT NULL ORDER BY t_start",
            (run_id or self.run_id, t0, t1))
        return [dict(r) for r in rows]

    def replenishment_list(self, run_id=None, planogram=None):
        """Open stock-outs sorted by revenue rate (hours_empty * sales/h * price).

        planogram: list of planogram dicts; when None, price is unknown (0).
        """
        rows = self.db.query(
            "SELECT roi, sku, t_start FROM stockout WHERE run_id = ? "
            "AND t_end IS NULL ORDER BY t_start", (run_id or self.run_id,))
        prices = {} if not planogram else {p["sku"]: p for p in planogram}
        now = self.db.query_one("SELECT MAX(t) AS t FROM occupancy") or {}
        out = []
        for r in rows:
            sku = r["sku"]
            p = prices.get(sku, {})
            rate = (p.get("expected_sales_per_hour", 0) * p.get("unit_price", 0))
            out.append({"roi": r["roi"], "sku": sku, "t_start": r["t_start"],
                        "revenue_rate": rate})
        out.sort(key=lambda x: x["revenue_rate"], reverse=True)
        return out

    # ---- queue ------------------------------------------------------------
    def queue_status(self, run_id=None):
        """Latest queue estimate per lane."""
        rows = self.db.query(
            "SELECT q.lane, q.count, q.pred_wait_s FROM queue_estimate q "
            "WHERE q.run_id = ? AND q.t = (SELECT MAX(t) FROM queue_estimate "
            "WHERE run_id = ? AND lane = q.lane) ORDER BY q.lane",
            (run_id or self.run_id, run_id or self.run_id))
        return rows

    def wait_pred_vs_actual(self, t0, t1, run_id=None):
        """Join predictions with ground truth by nearest timestamp. MAE is the W5 target."""
        pred = self.db.query(
            "SELECT t, lane, count, pred_wait_s FROM queue_estimate "
            "WHERE run_id = ? AND t >= ? AND t < ? ORDER BY t",
            (run_id or self.run_id, t0, t1))
        truth = self.db.query(
            "SELECT t, lane, count, actual_wait_s FROM queue_truth "
            "WHERE t >= ? AND t < ? ORDER BY t", (t0, t1))
        if not pred or not truth:
            return {"mae_s": None, "n": 0, "rows": []}
        rows = []
        for t in truth:
            best = min(pred, key=lambda p: abs(p["t"] - t["t"]))
            rows.append({"t": t["t"], "lane": t["lane"],
                         "pred_wait_s": best["pred_wait_s"],
                         "actual_wait_s": t["actual_wait_s"],
                         "error_s": best["pred_wait_s"] - t["actual_wait_s"]})
        mae = sum(abs(r["error_s"]) for r in rows) / len(rows)
        return {"mae_s": mae, "n": len(rows), "rows": rows}

    # ---- conversion -------------------------------------------------------
    def conversion_by_zone(self, t0, t1, run_id=None, zone_of=None):
        """Visits, conversions, and conversion rate per zone (see conversion.py)."""
        from backend.conversion import classify_visits

        visits = [dict(r) for r in self.db.query(
            "SELECT track_id, zone, t_enter, t_exit FROM visit "
            "WHERE run_id = ? AND t_enter >= ? AND t_enter < ?",
            (run_id or self.run_id, t0, t1))]
        # Rebuild Event-like objects for classify_visits; it only needs
        # zone_id and payload keys, so a lightweight namespace suffices.
        class V:
            def __init__(self, r):
                self.zone_id = r["zone"]
                self.payload = {"track_id": r["track_id"], "t_enter": r["t_enter"],
                                "t_exit": r["t_exit"]}
        txns = [dict(r) for r in self.db.query(
            "SELECT t, lane, zone_hint FROM pos_txn WHERE run_id = ? "
            "AND t >= ? AND t < ?", (run_id or self.run_id, t0, t1))]
        class T:
            def __init__(self, r):
                self.t = r["t"]
                self.payload = {"zone_hint": r["zone_hint"]}

        rows = classify_visits([V(r) for r in visits], [T(r) for r in txns], [],
                               zone_of=zone_of)
        out = {}
        for row in rows:
            z = row["zone"]
            d = out.setdefault(z, {"visits": 0, "conversions": 0, "rate": 0.0})
            d["visits"] += 1
            d["conversions"] += int(row["converted"])
        for d in out.values():
            d["rate"] = d["conversions"] / d["visits"] if d["visits"] else 0.0
        return out

    # ---- alerts -----------------------------------------------------------
    def alerts(self, open_only=True, run_id=None):
        q = "SELECT * FROM alert WHERE run_id = ?"
        params = [run_id or self.run_id]
        if open_only:
            q += " AND acked = 0"
        return self.db.query(q + " ORDER BY t DESC", params)

    def stores(self):
        """One real store plus any backfilled history stores."""
        rows = self.db.query("SELECT DISTINCT store_id FROM run")
        return [r["store_id"] for r in rows] if rows else []


def _day_bounds(t):
    """Unix second -> (midnight, next midnight) in local time."""
    import datetime
    import time
    lt = time.localtime(t)
    midnight = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday)
    t0 = time.mktime(midnight.timetuple())
    return t0, t0 + 86400
