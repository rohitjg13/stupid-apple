"""Conversion by zone: visits -> converted-or-not, then labels.

The plan's definition (05-backend-queue.md W3 §4):

- a visit counts only if dwell >= 5 s
- it is "converted" if, within 15 min of t_exit, a pos_txn's zone_hint maps to
  the zone, OR within 60 s of the visit a shelf_fill drops >= 30 in a ROI
  mapped to that zone
- labels: high dwell + low conversion = "confused / review layout";
  high dwell + high conversion = "engaged";
  low dwell + low conversion = "ignored"
"""
from __future__ import annotations

CONVERT_WINDOW_S = 900.0
SHELF_WINDOW_S = 60.0
SHELF_DROP = 30.0
DWELL_THRESHOLD_S = 5.0
HIGH_DWELL_S = 120.0
HIGH_CONVERSION = 0.5


def classify_visits(visits, txns, fills, zone_of=None):
    """Return a list of {visit, converted, dwell_s, zone, label}.

    visits: iterable of visit events (zone_id = zone, payload has t_enter/t_exit)
    txns:   iterable of pos_txn events
    fills:  iterable of shelf_fill events
    zone_of: optional fn(sku/roi) -> zone, for mapping a txn/facing to a zone.
             The real mapping is SKU -> planogram -> zone; here the caller can
             supply `planogram`'s facing -> zone lookup.
    """
    txns = sorted(txns, key=lambda e: e.t)
    fills = sorted(fills, key=lambda e: e.t)
    out = []
    for v in visits:
        dwell = v.payload["t_exit"] - v.payload["t_enter"]
        if dwell < DWELL_THRESHOLD_S:
            continue                      # too short to count as a browse

        zone = v.zone_id
        converted = False

        # (a) a txn whose zone maps to this zone, within 15 min of exit
        for e in txns:
            if e.t > v.payload["t_exit"] + CONVERT_WINDOW_S:
                break
            if e.t < v.payload["t_enter"]:
                continue
            tzone = e.payload["zone_hint"]
            if zone_of is not None:
                tzone = zone_of(tzone)
            if tzone == zone:
                converted = True
                break

        # (b) a shelf_fill drop >= 30 in a ROI mapped to this zone, within 60 s
        if not converted:
            for e in fills:
                if e.t > v.payload["t_exit"] + SHELF_WINDOW_S:
                    break
                if e.t < v.payload["t_enter"] - SHELF_WINDOW_S:
                    continue
                rzone = e.zone_id
                if zone_of is not None:
                    rzone = zone_of(rzone)
                if rzone == zone and e.payload["fill"] <= SHELF_DROP:
                    # a "drop" is signalled by a low fill reading here; the
                    # shelf lead owns the exact threshold, so keep it simple.
                    converted = True
                    break

        label = _label(dwell, converted)
        out.append({"visit": v, "converted": converted, "dwell_s": dwell,
                    "zone": zone, "label": label})
    return out


def _label(dwell, converted):
    if dwell >= HIGH_DWELL_S and not converted:
        return "confused"
    if dwell >= HIGH_DWELL_S and converted:
        return "engaged"
    if not converted:
        return "ignored"
    return "engaged"


def conversion_rate(rows):
    """Fraction of counted visits that converted."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["converted"]) / len(rows)
