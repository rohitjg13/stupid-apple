"""shopper/analytics.py -- the questions a per-frame detector cannot answer."""
import pytest

from shopper.analytics import AnalyticsParams, Journeys

FPS = 10.0


def grid_region(u, v):
    """Three columns: left < 200 <= centre < 400 <= right."""
    return "left" if u < 200 else ("centre" if u < 400 else "right")


def make(**kw):
    return Journeys(FPS, grid_region, AnalyticsParams(**kw))


def walk(js, tid, xs, standing=False, start=0, y=300, width=40):
    """Feed one person a scripted sequence of x positions."""
    for i, x in enumerate(xs):
        still = standing(i) if callable(standing) else standing
        js.observe(start + i, [(tid, (x, y), width, still)])
    return js


# --- 1. engagement -------------------------------------------------------------
def test_someone_who_walks_past_counts_as_passed_but_not_stopped():
    js = make()
    walk(js, 0, list(range(100, 200, 5)))                 # crosses "left", never stops
    rows = {r["region"]: r for r in js.engagement_rows()}
    assert rows["left"]["passed"] == 1
    assert rows["left"]["stopped"] == 0
    assert rows["left"]["stop_rate_pct"] == 0.0


def test_someone_who_stops_counts_in_both():
    js = make()
    walk(js, 0, [150] * 40, standing=True)                # 4 s standing in "left"
    row = js.engagement_rows()[0]
    assert row["region"] == "left"
    assert (row["passed"], row["stopped"]) == (1, 1)
    assert row["stop_rate_pct"] == 100.0
    assert row["mean_stop_s"] == pytest.approx(4.0)


def test_stop_rate_is_stoppers_over_passers():
    js = make()
    for tid in range(4):                                   # 4 people walk past
        walk(js, tid, list(range(100, 190, 5)))
    walk(js, 99, [150] * 40, standing=True)                # a 5th stops
    row = {r["region"]: r for r in js.engagement_rows()}["left"]
    assert (row["passed"], row["stopped"]) == (5, 1)
    assert row["stop_rate_pct"] == 20.0


def test_a_pause_shorter_than_stop_s_is_not_a_stop():
    js = make(stop_s=2.0)
    walk(js, 0, [150] * 10, standing=True)                 # 1 s only
    assert js.engagement_rows()[0]["stopped"] == 0


def test_clipping_a_corner_does_not_count_as_passing():
    js = make(min_region_s=0.3, region_debounce_s=0.1)
    walk(js, 0, [150, 155] + [250] * 30)                   # 0.2 s in "left"
    regions = {r["region"] for r in js.engagement_rows()}
    assert regions == {"centre"}


# --- 2. flow -------------------------------------------------------------------
def test_a_path_is_the_sequence_of_regions():
    js = make(region_debounce_s=0.2)
    walk(js, 0, [100] * 10 + [300] * 10 + [500] * 10)
    assert js.people[0].path == ["left", "centre", "right"]


def test_boundary_flicker_does_not_become_a_path():
    """Without debouncing this reads left>centre>left>centre>left..."""
    js = make(region_debounce_s=0.5)
    xs = [190] * 10
    for _ in range(8):
        xs += [205, 195]                                   # wobble across the line
    xs += [190] * 10
    walk(js, 0, xs)
    assert js.people[0].path == ["left"], js.people[0].path


def test_transitions_are_counted_across_people():
    js = make(region_debounce_s=0.2)
    for tid in range(3):
        walk(js, tid, [100] * 10 + [300] * 10)             # left -> centre
    walk(js, 9, [500] * 10 + [300] * 10)                   # right -> centre
    rows = js.transition_rows()
    assert rows[0] == {"from_region": "left", "to_region": "centre",
                       "moves": 3, "share_pct": 75.0}


def test_no_transitions_when_nobody_moved_between_regions():
    js = make()
    walk(js, 0, [150] * 30)
    assert js.transition_rows() == []


# --- 3. shopper type -----------------------------------------------------------
def test_a_brief_walk_through_is_passing():
    js = make(passing_s=3.0)
    walk(js, 0, list(range(100, 200, 10)))                 # 1 s, no stop
    assert js.person_rows()[0]["shopper_type"] == "passing"


def test_one_stop_is_considered():
    js = make()
    walk(js, 0, list(range(100, 200, 5)) + [200] * 30,
         standing=lambda i: i >= 20)
    assert js.person_rows()[0]["shopper_type"] == "considered"


def test_two_stops_is_browsing():
    js = make(stop_s=1.0)
    xs = [150] * 15 + list(range(150, 250, 10)) + [250] * 15
    walk(js, 0, xs, standing=lambda i: i < 15 or i >= 25)
    assert js.person_rows()[0]["shopper_type"] == "browsing"
    assert js.people[0].stops == 2


def test_one_long_stand_is_considered_not_browsing():
    """Studying one display for a minute is a different shopper from sampling
    four of them, and lumping both under "browsing" loses that."""
    js = make()
    walk(js, 0, [150] * 60, standing=True)
    assert js.person_rows()[0]["shopper_type"] == "considered"
    assert js.people[0].stops == 1


def test_a_long_walk_without_stopping_is_direct():
    js = make(passing_s=3.0)
    walk(js, 0, list(range(100, 600, 5)))                  # 10 s, always moving
    assert js.person_rows()[0]["shopper_type"] == "direct"


def test_jitter_while_standing_is_not_counted_as_walking():
    """A stationary shopper's box breathes; summing it invents a journey."""
    js = make()
    xs = [150 + (i % 2) for i in range(60)]                # 1 px wobble, standing
    walk(js, 0, xs, standing=True)
    assert js.person_rows()[0]["walked_px"] == 0


def test_real_walking_is_measured():
    js = make()
    walk(js, 0, list(range(100, 300, 10)))                 # 190 px of travel
    assert js.person_rows()[0]["walked_px"] == pytest.approx(190, abs=15)


# --- 4. groups -----------------------------------------------------------------
def test_two_people_walking_together_are_a_group():
    js = make(group_min_s=1.0)
    for f in range(40):
        js.observe(f, [(0, (100 + f * 5, 300), 40, False),
                       (1, (130 + f * 5, 300), 40, False)])     # 30 px apart
    rows = js.group_rows()
    assert len(rows) == 1
    assert rows[0]["size"] == 2 and rows[0]["members"] == "#0 #1"
    assert rows[0]["together_s"] == pytest.approx(4.0)


def test_two_strangers_far_apart_are_not_a_group():
    js = make(group_min_s=1.0)
    for f in range(40):
        js.observe(f, [(0, (100, 300), 40, True), (1, (500, 300), 40, True)])
    assert js.group_rows() == []


def test_briefly_passing_close_is_not_a_group():
    js = make(group_min_s=3.0, group_min_frac=0.5)
    for f in range(40):
        a = (100 + f * 10, 300)
        b = (500 - f * 10, 300)                                 # cross in the middle
        js.observe(f, [(0, a, 40, False), (1, b, 40, False)])
    assert js.group_rows() == []


def test_a_group_of_three_is_merged_transitively():
    js = make(group_min_s=1.0)
    for f in range(40):
        js.observe(f, [(0, (100, 300), 40, True), (1, (140, 300), 40, True),
                       (2, (180, 300), 40, True)])              # A-B and B-C close
    rows = js.group_rows()
    assert len(rows) == 1 and rows[0]["size"] == 3


def test_group_distance_scales_with_body_size():
    """40 px apart is together for a big near box, apart for a small far one."""
    near = make(group_min_s=1.0)
    far = make(group_min_s=1.0)
    for f in range(40):
        near.observe(f, [(0, (100, 300), 60, True), (1, (160, 300), 60, True)])
        far.observe(f, [(0, (100, 300), 10, True), (1, (160, 300), 10, True)])
    assert len(near.group_rows()) == 1
    assert far.group_rows() == []


def test_type_counts_totals_everyone():
    js = make()
    walk(js, 0, [150] * 40, standing=True)
    walk(js, 1, list(range(100, 200, 10)), start=100)
    assert sum(js.type_counts().values()) == 2


def test_engagement_leads_with_where_people_stopped_not_where_they_walked():
    """The busiest area is usually the doorway; the useful one is the display
    people actually stayed at."""
    js = make()
    for tid in range(6):                                   # a busy thoroughfare
        walk(js, tid, list(range(100, 190, 5)))
    walk(js, 90, [300] * 40, standing=True)                # one quiet display
    walk(js, 91, [305] * 40, standing=True, start=100)
    rows = js.engagement_rows()
    assert rows[0]["region"] == "centre", [r["region"] for r in rows]
    assert rows[0]["stopped"] == 2 and rows[1]["passed"] == 6
