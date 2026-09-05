"""shopper/visits.py -- the zone state machine and dwell."""
from shopper.params import TrackerParams
from shopper.visits import VisitTracker, Visit

FPS = 15.0
P = TrackerParams(visit_debounce_s=1.0, min_visit_s=2.0)


def play(zones_per_frame, params=None, track_id=0, fps=FPS):
    """Feed one track a scripted path of zones; return completed visits."""
    vt = VisitTracker(params or P)
    out = []
    for i, z in enumerate(zones_per_frame):
        out += vt.update([(track_id, z)], i / fps)
    return vt, out


def steady(zone, seconds, fps=FPS):
    return [zone] * int(seconds * fps)


def test_a_short_pass_through_is_not_a_visit():
    _, visits = play(steady("aisle_a", 1.5) + steady("checkout", 5))
    assert [v.zone_id for v in visits] == [], "1.5 s of walking through is not a visit"


def test_a_real_stay_becomes_a_visit_when_they_leave():
    _, visits = play(steady("aisle_a", 6) + steady("checkout", 4))
    assert len(visits) == 1
    assert visits[0].zone_id == "aisle_a"
    assert visits[0].dwell_s > TrackerParams().min_visit_s


def test_dwell_measures_presence_not_the_debounce_delay():
    """t_exit is when they were first seen elsewhere, not when we believed it."""
    _, visits = play(steady("aisle_a", 6) + steady("checkout", 4))
    # In aisle_a from t=0 until first seen in checkout at t=6, so dwell is 6 s
    # exactly -- the 1 s of debounce delays the decision, not the measurement.
    assert visits[0].dwell_s == 6.0


def test_flicker_on_a_boundary_does_not_split_the_visit():
    """The failure this whole module exists to prevent."""
    path = steady("aisle_a", 3)
    for _ in range(10):                       # 10 wobbles across the boundary
        path += ["aisle_b"] * 3 + ["aisle_a"] * 3
    path += steady("checkout", 4)
    _, visits = play(path)
    assert len(visits) == 1, f"boundary flicker split the visit into {len(visits)}"
    assert visits[0].zone_id == "aisle_a"


def test_a_genuine_zone_change_is_believed_after_the_debounce():
    _, visits = play(steady("aisle_a", 4) + steady("aisle_b", 4) + steady("checkout", 4))
    assert [v.zone_id for v in visits] == ["aisle_a", "aisle_b"]


def test_leaving_the_zoned_area_entirely_closes_the_visit():
    _, visits = play(steady("aisle_a", 5) + [None] * int(4 * FPS))
    assert [v.zone_id for v in visits] == ["aisle_a"]


def test_time_outside_every_zone_is_never_itself_a_visit():
    _, visits = play([None] * int(10 * FPS))
    assert visits == []


def test_a_deleted_track_flushes_its_open_visit():
    vt, visits = play(steady("aisle_a", 5))
    assert visits == [], "visit closed early"
    v = vt.close_track(0, 5.0)
    assert isinstance(v, Visit) and v.zone_id == "aisle_a"


def test_a_deleted_track_with_too_short_a_visit_flushes_nothing():
    vt, _ = play(steady("aisle_a", 1.2))
    assert vt.close_track(0, 1.2) is None


def test_shutdown_flushes_every_open_visit():
    vt = VisitTracker(P)
    for i in range(int(5 * FPS)):
        vt.update([(0, "aisle_a"), (1, "aisle_b")], i / FPS)
    flushed = vt.close(5.0)
    assert sorted(v.zone_id for v in flushed) == ["aisle_a", "aisle_b"]


def test_closing_an_unknown_track_is_harmless():
    assert VisitTracker(P).close_track(99, 1.0) is None


def test_two_tracks_keep_independent_state():
    vt = VisitTracker(P)
    out = []
    for i in range(int(5 * FPS)):
        out += vt.update([(0, "aisle_a"), (1, "aisle_b")], i / FPS)
    for i in range(int(5 * FPS), int(9 * FPS)):
        out += vt.update([(0, "checkout"), (1, "checkout")], i / FPS)
    assert sorted((v.track_id, v.zone_id) for v in out) == [(0, "aisle_a"), (1, "aisle_b")]
