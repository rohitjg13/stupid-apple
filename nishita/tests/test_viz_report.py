"""tools/viz_tracks.py report helpers -- the pure parts."""
import numpy as np

from tools.viz_tracks import (Report, bucket_of, content_rect, near_edge, region_of)


def test_content_rect_for_a_16_9_clip_has_bars_top_and_bottom():
    x0, y0, x1, y1 = content_rect(1920, 1080)
    assert (x0, x1) == (0, 640)
    assert y0 == 60 and y1 == 420


def test_content_rect_for_a_portrait_clip_has_bars_left_and_right():
    x0, y0, x1, y1 = content_rect(1080, 1920)
    assert (y0, y1) == (0, 480)
    assert x0 > 0 and x1 < 640


def test_near_edge_is_judged_against_content_not_the_black_bars():
    rect = content_rect(1920, 1080)                    # content y 60..420
    assert near_edge(320, 70, rect)                    # just inside the top of the picture
    assert not near_edge(320, 240, rect)               # middle of the picture
    assert near_edge(5, 240, rect)                     # left edge


def test_region_names_cover_the_nine_cells():
    rect = (0, 0, 640, 480)
    assert region_of(10, 10, rect) == "top-left"
    assert region_of(320, 240, rect) == "middle-centre"
    assert region_of(630, 470, rect) == "bottom-right"


def test_buckets_are_whole_seconds_of_frames():
    assert bucket_of(0, 30.0, 10.0) == 0
    assert bucket_of(299, 30.0, 10.0) == 0
    assert bucket_of(300, 30.0, 10.0) == 1


class FakeTrack:
    def __init__(self, tid, foot, travelled=0.0):
        self.id, self.foot, self.travelled, self.held, self.seen_speed = tid, foot, travelled, 0, 0.0


def test_footfall_counts_arrivals_and_departures_per_bucket():
    rep = Report(fps=10.0, rect=(0, 0, 640, 480), bucket_s=1.0, has_door=False, has_zones=False)
    still = lambda tr: True
    none = lambda tr: None
    for f in range(25):                                   # 2.5 s
        tracks = [FakeTrack(0, (100, 400))]
        if 10 <= f < 20:                                  # second person for 1 s
            tracks.append(FakeTrack(1, (300, 400)))
        rep.frame(f, tracks, still, none, None)
    rows = rep.footfall_rows()
    assert [r["arrivals"] for r in rows] == [1, 1, 0]
    assert [r["departures"] for r in rows] == [0, 1, 0]   # #1 left in bucket 1; #0 still there
    assert rows[1]["people_present_max"] == 2


def test_dwell_reports_seconds_and_the_region_they_stood_in():
    rep = Report(fps=10.0, rect=(0, 0, 640, 480), bucket_s=1.0, has_door=False, has_zones=False)
    for f in range(30):
        rep.frame(f, [FakeTrack(7, (600, 460), travelled=55)], lambda tr: f >= 10, lambda tr: None, None)
    row = rep.dwell_rows()[0]
    assert row["present_s"] == 3.0 and row["standing_still_s"] == 2.0
    assert row["mostly_in_region"] == "bottom-right" and row["walked_px"] == 55


def test_edge_proxy_counts_a_walk_through_as_one_entry_and_one_exit():
    rep = Report(fps=10.0, rect=(0, 0, 640, 480), bucket_s=1.0, has_door=False, has_zones=False)
    for f in range(30):
        u = 5 + f * 22                                    # left edge -> past the right edge band
        rep.frame(f, [FakeTrack(0, (u, 300))] if f < 29 else [], lambda tr: False, lambda tr: None, None)
    rows = rep.entries_exits_rows()
    assert rows[0]["entered_view_from_edge"] == 1
    assert rows[0]["left_view_via_edge"] == 1
    assert "no door" in rows[0]["basis"]


def test_someone_still_present_at_the_end_has_not_left():
    rep = Report(fps=10.0, rect=(0, 0, 640, 480), bucket_s=1.0, has_door=False, has_zones=False)
    for f in range(30):
        rep.frame(f, [FakeTrack(0, (5 + f, 300))], lambda tr: False, lambda tr: None, None)
    assert rep.entries_exits_rows()[0]["left_view_via_edge"] == 0


def test_a_configured_door_reports_crossings_not_the_proxy():
    rep = Report(fps=10.0, rect=(0, 0, 640, 480), bucket_s=1.0, has_door=True, has_zones=False)
    for f in range(20):
        rep.frame(f, [], lambda tr: False, lambda tr: None, None)
    rep.crossings = [(3, "in"), (15, "out")]
    rows = rep.entries_exits_rows()
    assert rows[0]["entered"] == 1 and rows[1]["exited"] == 1
    assert rows[0]["basis"] == "configured door line"


def test_summary_is_honest_about_what_needs_a_config():
    rep = Report(fps=10.0, rect=(0, 0, 640, 480), bucket_s=1.0, has_door=False, has_zones=False)
    rep.frame(0, [FakeTrack(0, (100, 100))], lambda tr: False, lambda tr: None, None)
    s = rep.summary("x.mp4", "yolo")
    assert "proxy" in s["1_entries_exits"]["basis"]
    assert "zones.json" in s["2_footfall"]["by_zone"]
    assert s["4_heatmap"]["hottest"]["foot_samples"] == 1
