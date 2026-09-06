"""backend/conversion.py — visit -> converted, and the engagement labels."""
from core.events import Event


def ev(t, et, payload, zone_id=None, stream="overhead"):
    return Event(t, "demo-01", stream, zone_id, et, payload)


def visit(t_enter, t_exit, zone):
    return ev(t_enter, "visit", {"track_id": 1, "t_enter": t_enter, "t_exit": t_exit},
              zone_id=zone)


def test_short_visits_are_not_counted():
    from backend.conversion import classify_visits
    rows = classify_visits([visit(0, 3, "aisle_a")], [], [])
    assert rows == []


def test_txn_in_zone_converts():
    from backend.conversion import classify_visits
    rows = classify_visits([visit(0, 60, "checkout")],
                           [ev(61, "pos_txn", {"lane": 1, "items": 1,
                                                "amount": 10.0, "zone_hint": "checkout"},
                               stream="-")], [])
    assert rows[0]["converted"] is True
    assert rows[0]["label"] == "engaged"


def test_txn_after_window_does_not_convert():
    from backend.conversion import classify_visits
    rows = classify_visits([visit(0, 10, "aisle_a")],
                           [ev(1000, "pos_txn", {"lane": 1, "items": 1,
                                                  "amount": 10.0, "zone_hint": "aisle_a"},
                               stream="-")], [])
    assert rows[0]["converted"] is False


def test_shelf_drop_converts():
    from backend.conversion import classify_visits
    # A *drop* of >= 30 between consecutive samples, per the plan: 180 -> 20.
    fills = [ev(10, "shelf_fill", {"fill": 180}, zone_id="aisle_a", stream="shelf"),
             ev(15, "shelf_fill", {"fill": 20}, zone_id="aisle_a", stream="shelf")]
    rows = classify_visits([visit(0, 14, "aisle_a")], [], fills)
    assert rows[0]["converted"] is True
    assert rows[0]["via"] == "shelf"


def test_shelf_that_is_merely_empty_does_not_convert():
    """A facing empty all afternoon is a stock-out, not a sale.

    The absolute fill level says nothing about whether anyone took anything;
    only the change between two samples does.
    """
    from backend.conversion import classify_visits
    fills = [ev(10, "shelf_fill", {"fill": 20}, zone_id="aisle_a", stream="shelf"),
             ev(15, "shelf_fill", {"fill": 19}, zone_id="aisle_a", stream="shelf")]
    rows = classify_visits([visit(0, 14, "aisle_a")], [], fills)
    assert rows[0]["converted"] is False


def test_small_shelf_drop_is_below_threshold():
    from backend.conversion import classify_visits, SHELF_DROP
    fills = [ev(10, "shelf_fill", {"fill": 100}, zone_id="aisle_a", stream="shelf"),
             ev(15, "shelf_fill", {"fill": 100 - SHELF_DROP + 1}, zone_id="aisle_a",
                stream="shelf")]
    rows = classify_visits([visit(0, 14, "aisle_a")], [], fills)
    assert rows[0]["converted"] is False


def test_shelf_drop_in_another_roi_does_not_convert():
    from backend.conversion import classify_visits
    fills = [ev(10, "shelf_fill", {"fill": 180}, zone_id="aisle_b", stream="shelf"),
             ev(15, "shelf_fill", {"fill": 20}, zone_id="aisle_b", stream="shelf")]
    rows = classify_visits([visit(0, 14, "aisle_a")], [], fills)
    assert rows[0]["converted"] is False


def test_shelf_drops_differences_per_roi():
    from backend.conversion import shelf_drops
    fills = [ev(30, "shelf_fill", {"fill": 40}, zone_id="A1", stream="shelf"),
             ev(10, "shelf_fill", {"fill": 200}, zone_id="A1", stream="shelf"),
             ev(20, "shelf_fill", {"fill": 190}, zone_id="A1", stream="shelf"),
             ev(10, "shelf_fill", {"fill": 100}, zone_id="A2", stream="shelf")]
    # Unsorted input, two ROIs: only A1's 190 -> 40 clears the threshold, and
    # a single sample in A2 has nothing to difference against.
    assert shelf_drops(fills) == [(30.0, "A1", 150)]


def test_labels_cover_all_cases():
    from backend.conversion import classify_visits, HIGH_DWELL_S
    # low dwell + no conversion -> ignored
    rows = classify_visits([visit(0, 6, "aisle_a")], [], [])
    assert rows[0]["label"] == "ignored"
    # high dwell + no conversion -> confused
    rows = classify_visits([visit(0, HIGH_DWELL_S + 10, "aisle_a")], [], [])
    assert rows[0]["label"] == "confused"
    # high dwell + conversion -> engaged
    rows = classify_visits([visit(0, HIGH_DWELL_S + 10, "aisle_a")],
                           [ev(HIGH_DWELL_S + 11, "pos_txn",
                               {"lane": 1, "items": 1, "amount": 10.0,
                                "zone_hint": "aisle_a"}, stream="-")], [])
    assert rows[0]["label"] == "engaged"


def test_zone_mapping_via_callable():
    from backend.conversion import classify_visits

    def zone_of(facing):
        return {"A1": "aisle_a"}.get(facing)

    txns = [ev(60, "pos_txn", {"lane": 1, "items": 1, "amount": 10.0, "zone_hint": "A1"},
               stream="-")]
    rows = classify_visits([visit(0, 59, "aisle_a")], txns, [], zone_of=zone_of)
    assert rows[0]["converted"] is True


def test_conversion_rate():
    from backend.conversion import classify_visits, conversion_rate
    rows = classify_visits([visit(0, 10, "aisle_a"), visit(20, 30, "aisle_b")],
                           [ev(31, "pos_txn", {"lane": 1, "items": 1,
                                               "amount": 10.0, "zone_hint": "aisle_b"},
                               stream="-")], [])
    assert conversion_rate(rows) == 0.5
    assert conversion_rate([]) == 0.0


# ---- zone-level labels (the quadrant the dashboard shows) -------------------

def test_label_zone_quadrants():
    from backend.conversion import HIGH_DWELL_S, label_zone
    lo, hi = HIGH_DWELL_S - 1, HIGH_DWELL_S + 1
    assert label_zone(hi, 0.1) == "confused"     # long browse, no sale
    assert label_zone(hi, 0.9) == "engaged"
    assert label_zone(lo, 0.1) == "ignored"
    assert label_zone(lo, 0.9) == "efficient"    # found it and bought it


def test_label_zone_is_inclusive_at_the_thresholds():
    from backend.conversion import HIGH_CONVERSION, HIGH_DWELL_S, label_zone
    assert label_zone(HIGH_DWELL_S, HIGH_CONVERSION) == "engaged"


def test_via_records_which_signal_converted():
    from backend.conversion import classify_visits
    txns = [ev(20, "pos_txn", {"lane": 1, "items": 1, "amount": 5.0,
                               "zone_hint": "aisle_a"}, stream="-")]
    rows = classify_visits([visit(0, 10, "aisle_a")], txns, [])
    assert rows[0]["via"] == "pos"
    rows = classify_visits([visit(0, 10, "aisle_a")], [], [])
    assert rows[0]["via"] is None
