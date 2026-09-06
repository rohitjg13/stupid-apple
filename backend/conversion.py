"""Conversion by zone: visits -> converted-or-not, then labels.

The plan's definition (05-backend-queue.md W3 §4):

- a visit counts only if dwell >= 5 s
- it is "converted" if, within 15 min of t_exit, a pos_txn's zone_hint maps to
  the zone, OR within 60 s of the visit a shelf_fill drops >= 30 in a ROI
  mapped to that zone
- labels: high dwell + low conversion = "confused / review layout";
  high dwell + high conversion = "engaged";
  low dwell + low conversion = "ignored"

"Drops >= 30" is a *change* between consecutive samples in the same ROI, not a
low absolute reading: a facing that is simply empty all afternoon is a
stock-out, not somebody buying something. `shelf_drops()` does that differencing
so the rule means what the plan says it means.

Nothing here needs to know who the shopper was — a visit is a zone, two
timestamps and a per-process track id that is thrown away at exit.
"""
from __future__ import annotations

CONVERT_WINDOW_S = 900.0
SHELF_WINDOW_S = 60.0
SHELF_DROP = 30.0
DWELL_THRESHOLD_S = 5.0
HIGH_DWELL_S = 120.0
HIGH_CONVERSION = 0.5


def shelf_drops(fills, min_drop=SHELF_DROP):
    """[(t, roi, drop)] for each consecutive fall of >= min_drop within a ROI.

    `fills` is any iterable of shelf_fill events; they are sorted per ROI here
    so callers can hand over a whole window unordered.
    """
    by_roi = {}
    for e in fills:
        by_roi.setdefault(e.zone_id, []).append(e)
    out = []
    for roi, samples in by_roi.items():
        samples.sort(key=lambda e: e.t)
        for prev, cur in zip(samples, samples[1:]):
            drop = prev.payload["fill"] - cur.payload["fill"]
            if drop >= min_drop:
                out.append((cur.t, roi, drop))
    out.sort()
    return out


def classify_visits(visits, txns, fills, zone_of=None):
    """Return a list of {visit, converted, dwell_s, zone, label, via}.

    visits: iterable of visit events (zone_id = zone, payload has t_enter/t_exit)
    txns:   iterable of pos_txn events
    fills:  iterable of shelf_fill events
    zone_of: optional fn(zone_hint / roi) -> zone. The real mapping is
             SKU -> planogram -> zone; the stub already emits zone ids, so the
             default is the identity.
    """
    txns = sorted(txns, key=lambda e: e.t)
    drops = shelf_drops(fills)
    out = []
    for v in visits:
        t_enter, t_exit = v.payload["t_enter"], v.payload["t_exit"]
        dwell = t_exit - t_enter
        if dwell < DWELL_THRESHOLD_S:
            continue                      # too short to count as a browse

        zone = v.zone_id
        via = None

        # (a) a txn whose zone maps to this zone, within 15 min of exit
        for e in txns:
            if e.t > t_exit + CONVERT_WINDOW_S:
                break
            if e.t < t_enter:
                continue
            tzone = e.payload["zone_hint"]
            if zone_of is not None:
                tzone = zone_of(tzone)
            if tzone == zone:
                via = "pos"
                break

        # (b) a shelf_fill drop >= 30 in a ROI mapped to this zone, within 60 s
        if via is None:
            for t, roi, _drop in drops:
                if t > t_exit + SHELF_WINDOW_S:
                    break
                if t < t_enter - SHELF_WINDOW_S:
                    continue
                rzone = zone_of(roi) if zone_of is not None else roi
                if rzone == zone:
                    via = "shelf"
                    break

        converted = via is not None
        out.append({"visit": v, "converted": converted, "dwell_s": dwell,
                    "zone": zone, "label": _label(dwell, converted), "via": via})
    return out


def _label(dwell, converted):
    """Per-visit label. Zone-level labels are `label_zone`."""
    if dwell >= HIGH_DWELL_S and not converted:
        return "confused"
    if dwell >= HIGH_DWELL_S and converted:
        return "engaged"
    if not converted:
        return "ignored"
    return "engaged"


def label_zone(median_dwell_s, rate):
    """The label the dashboard shows per zone (plan W3 §4).

    A single visit converts or it does not, so the plan's "high dwell + low
    conversion" only means anything once the visits in a zone are pooled — that
    pooling is what this labels. The fourth quadrant the plan does not name
    (short dwell, high conversion) is "efficient": people find it and buy it.
    """
    high_dwell = median_dwell_s >= HIGH_DWELL_S
    high_conv = rate >= HIGH_CONVERSION
    if high_dwell and not high_conv:
        return "confused"          # long browse, no sale: review the layout
    if high_dwell and high_conv:
        return "engaged"
    if not high_dwell and high_conv:
        return "efficient"
    return "ignored"


def conversion_rate(rows):
    """Fraction of counted visits that converted."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["converted"]) / len(rows)
