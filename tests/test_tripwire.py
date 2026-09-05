"""shopper/tripwire.py -- crossing geometry, direction, debounce."""
import pytest

from shopper.params import TrackerParams
from shopper.tracker import Tracker
from shopper.tripwire import TripwireCounter, segments_intersect, side_of
from tests.fixtures import scenarios as sc

P = TrackerParams()
HORIZONTAL = {"id": "door", "stream": "overhead", "p1": [40, 460], "p2": [240, 460],
              "in_dir": "up"}
VERTICAL = {"id": "side", "stream": "overhead", "p1": [300, 100], "p2": [300, 400],
            "in_dir": "right"}


class FakeTrack:
    """The tripwire only needs an id, a trail and a foot point."""

    def __init__(self, tid, points, t=0.0):
        self.id = tid
        self.trail = [(t, x, y) for x, y in points]
        self.last_t = t

    @property
    def foot(self):
        return self.trail[-1][1], self.trail[-1][2]


def fire(wire, points, params=None, t=10.0):
    c = TripwireCounter([wire], params or P)
    return c.update([FakeTrack(0, points)], t)


# --- geometry -------------------------------------------------------------
def test_segments_that_cross_are_detected():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))


def test_parallel_segments_never_intersect():
    assert not segments_intersect((0, 0), (10, 0), (0, 5), (10, 5))


def test_segments_that_stop_short_do_not_intersect():
    assert not segments_intersect((0, 0), (4, 0), (5, -5), (5, 5))


def test_side_of_is_signed_and_zero_on_the_line():
    line = ((0, 0), (10, 0))
    assert side_of(line, (5, 5)) == 1
    assert side_of(line, (5, -5)) == -1
    assert side_of(line, (5, 0)) == 0


# --- direction, all four in_dir values ------------------------------------
def test_walking_up_through_an_up_wire_is_an_entry():
    assert fire(HORIZONTAL, [(120, 480), (120, 440)]) == [("door", "in", 0)]


def test_walking_down_through_an_up_wire_is_an_exit():
    assert fire(HORIZONTAL, [(120, 440), (120, 480)]) == [("door", "out", 0)]


def test_walking_down_through_a_down_wire_is_an_entry():
    wire = dict(HORIZONTAL, in_dir="down")
    assert fire(wire, [(120, 440), (120, 480)]) == [("door", "in", 0)]


def test_walking_up_through_a_down_wire_is_an_exit():
    wire = dict(HORIZONTAL, in_dir="down")
    assert fire(wire, [(120, 480), (120, 440)]) == [("door", "out", 0)]


def test_walking_right_through_a_right_wire_is_an_entry():
    assert fire(VERTICAL, [(280, 200), (320, 200)]) == [("side", "in", 0)]


def test_walking_left_through_a_right_wire_is_an_exit():
    assert fire(VERTICAL, [(320, 200), (280, 200)]) == [("side", "out", 0)]


def test_walking_left_through_a_left_wire_is_an_entry():
    wire = dict(VERTICAL, in_dir="left")
    assert fire(wire, [(320, 200), (280, 200)]) == [("side", "in", 0)]


def test_an_unknown_in_dir_is_rejected_at_construction():
    with pytest.raises(ValueError, match="in_dir"):
        TripwireCounter([dict(HORIZONTAL, in_dir="sideways")], P)


# --- what must not fire ---------------------------------------------------
def test_walking_parallel_to_the_wire_never_fires():
    assert fire(HORIZONTAL, [(60, 470), (200, 470)]) == []


def test_crossing_beyond_the_end_of_the_segment_never_fires():
    """The wire is a segment, not an infinite line: past x=240 is not the door."""
    assert fire(HORIZONTAL, [(500, 470), (500, 450)]) == []


def test_a_track_that_never_moves_never_fires():
    assert fire(HORIZONTAL, [(120, 480), (120, 480)]) == []


def test_wires_for_another_stream_are_ignored():
    c = TripwireCounter([dict(HORIZONTAL, stream="shelf")], P)
    assert c.wires == []
    assert c.update([FakeTrack(0, [(120, 480), (120, 440)])], 10.0) == []


# --- debounce and counting ------------------------------------------------
def test_one_track_cannot_fire_twice_inside_the_debounce():
    c = TripwireCounter([HORIZONTAL], TrackerParams(tripwire_debounce_s=2.0))
    a = c.update([FakeTrack(0, [(120, 480), (120, 440)])], 10.0)
    b = c.update([FakeTrack(0, [(120, 440), (120, 480)])], 10.5)
    assert len(a) == 1 and b == [], "debounce did not hold"


def test_the_same_track_may_fire_again_after_the_debounce():
    c = TripwireCounter([HORIZONTAL], TrackerParams(tripwire_debounce_s=2.0))
    c.update([FakeTrack(0, [(120, 480), (120, 440)])], 10.0)
    later = c.update([FakeTrack(0, [(120, 440), (120, 480)])], 13.0)
    assert later == [("door", "out", 0)]


def test_counts_accumulate_by_direction():
    c = TripwireCounter([HORIZONTAL], P)
    c.update([FakeTrack(0, [(120, 480), (120, 440)])], 10.0)
    c.update([FakeTrack(1, [(120, 440), (120, 480)])], 10.0)
    assert c.counts == {"in": 1, "out": 1}


def test_forget_drops_a_deleted_tracks_state():
    c = TripwireCounter([HORIZONTAL], P)
    c.update([FakeTrack(0, [(120, 480), (120, 440)])], 10.0)
    c.forget(0)
    assert not any(k[0] == 0 for k in c._last)


# --- the sim's four-pixel spawn ------------------------------------------
def test_an_entry_is_caught_even_though_it_happens_before_confirmation():
    """The config/sim case: spawn at v=464, door at v=460, confirmed at hit 3.

    The crossing happens on the track's second frame. If the tripwire only ever
    saw confirmed tracks' current positions it would miss every single entry.
    """
    tk = Tracker(P)
    counter = TripwireCounter([HORIZONTAL], P)
    events = []
    for i, boxes in enumerate(sc.walk_across_line(n=8, y_line=460, x=90)):
        active = tk.update(sc.frame_from_boxes(i, boxes).blobs, i / 15.0)
        events += counter.update(active, i / 15.0)
    assert events == [("door", "in", 0)], f"expected one entry, got {events}"


def test_the_replayed_birth_buffer_is_consumed_only_once():
    tk = Tracker(P)
    counter = TripwireCounter([HORIZONTAL], P)
    events = []
    for i, boxes in enumerate(sc.walk_across_line(n=20, y_line=460, x=90)):
        active = tk.update(sc.frame_from_boxes(i, boxes).blobs, i / 15.0)
        events += counter.update(active, i / 15.0)
    assert len(events) == 1, f"replayed the buffer more than once: {events}"
