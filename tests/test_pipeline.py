"""shopper/pipeline.py -- the glue, and the guarantees main.py depends on."""
import numpy as np
import pytest

from core.config import load_clipset
from core.events import EVENT_TYPES
from pl.contract import FLAG_BG_WARM
from shopper.params import TrackerParams
from shopper.pipeline import ShopperPipeline
from tests.fixtures import scenarios as sc

FPS = 15.0
T0 = 1_000_000.0
CFG = load_clipset("config/sim")


class Recorder:
    """A synchronous stand-in for core.bus.Bus, so tests are not thread-timed."""

    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def of(self, event_type):
        return [e for e in self.events if e.event_type == event_type]


def run(box_lists, warm_from=0, params=None, cfg=CFG, fps=FPS):
    bus = Recorder()
    pipe = ShopperPipeline(cfg, bus, params=params or TrackerParams())
    for i, boxes in enumerate(box_lists):
        frame = sc.frame_from_boxes(i, boxes, warm=i >= warm_from, fps=fps)
        pipe.on_frame(frame, T0 + i / fps)
    return pipe, bus


def walk_through_store(n=150):
    """In through the door and up into an aisle, the way config/sim's walkers go.

    Foot point starts at v=476, below the door tripwire at v=460, with u=120 so it
    is inside the wire's x-span of 40..240. Walking up means Y increasing on the
    floor (the sim homography is Y = 6 - v/80), so this crosses entrance into
    aisle_a and then stands still.
    """
    out, v = [], 476.0
    for _ in range(n):
        v = max(v - 2.0, 300.0)
        out.append([(100, int(v) - 136, 40, 136)])
    return out


# --- the occupancy heartbeat ---------------------------------------------
def test_occupancy_is_published_about_once_a_second():
    _, bus = run(walk_through_store(150))
    assert 8 <= len(bus.of("occupancy")) <= 12


def test_occupancy_still_ticks_while_the_background_is_cold():
    """A tile that stops updating for 7 s looks broken, and test_main.py counts these."""
    _, bus = run(walk_through_store(150), warm_from=100)
    occ = bus.of("occupancy")
    assert 8 <= len(occ) <= 12, f"heartbeat stopped during warm-up: {len(occ)}"
    assert occ[0].payload["count"] == 0, "counted people from a cold background model"


def test_nothing_is_tracked_before_the_background_is_warm():
    pipe, bus = run(walk_through_store(90), warm_from=100)
    assert pipe.tracker.tracks == []
    assert bus.of("visit") == [] and bus.of("tripwire") == []


def test_shoppers_in_the_entrance_are_not_counted_as_shopping():
    """They are arriving, not browsing. config/sim's entrance is X 0-3 m."""
    entrance_only = [[(60, 300, 40, 136)] for _ in range(60)]
    _, bus = run(entrance_only)
    assert {e.payload["count"] for e in bus.of("occupancy")} == {0}


def test_a_shopper_in_an_aisle_is_counted():
    aisle = [[(400, 100, 40, 136)] for _ in range(60)]
    _, bus = run(aisle)
    assert max(e.payload["count"] for e in bus.of("occupancy")) == 1


# --- the flood guard ------------------------------------------------------
def test_a_flooded_frame_pauses_counting():
    _, bus = run(sc.flood(n=60))
    assert {e.payload["count"] for e in bus.of("occupancy")} == {0}


def test_counting_resumes_after_the_flood_clears():
    seq = sc.flood(n=15) + [[(400, 100, 40, 136)] for _ in range(120)]
    _, bus = run(seq)
    assert max(e.payload["count"] for e in bus.of("occupancy")) == 1


# --- events match the frozen contract ------------------------------------
@pytest.mark.parametrize("event_type", ["occupancy", "visit", "tripwire", "heatmap"])
def test_every_emitted_event_carries_exactly_the_frozen_payload_keys(event_type):
    """core.events.Event validates this too; this proves we actually emit them."""
    _, bus = run(walk_through_store(400))
    got = bus.of(event_type)
    assert got, f"no {event_type} events were emitted at all"
    for e in got:
        assert set(e.payload) == EVENT_TYPES[event_type]


def test_events_carry_the_store_id_and_stream():
    _, bus = run(walk_through_store(200))
    assert {e.store_id for e in bus.events} == {"demo-01"}
    assert {e.stream for e in bus.events} == {"overhead"}


def test_a_visit_reports_its_zone_on_the_envelope_not_in_the_payload():
    _, bus = run(walk_through_store(400))
    visits = bus.of("visit")
    assert visits and all(v.zone_id for v in visits)
    assert all("zone" not in v.payload for v in visits)


def test_track_ids_appear_only_inside_visit_payloads():
    """docs/SHARED.md and CLAUDE.md: ids are ephemeral and must not leak."""
    _, bus = run(walk_through_store(400))
    for e in bus.events:
        if e.event_type != "visit":
            assert "track_id" not in e.payload


def test_the_heatmap_is_published_on_its_own_period():
    _, bus = run(walk_through_store(400), params=TrackerParams(heatmap_period_s=10.0))
    hm = bus.of("heatmap")
    # 400 frames at 15 fps is ~26.7 s of stream time.
    assert 1 <= len(hm) <= 3, f"expected a couple of heatmaps, got {len(hm)}"
    assert hm[-1].payload["tiles"], "the heatmap published no cells"


# --- lifecycle ------------------------------------------------------------
def test_close_flushes_open_visits():
    pipe, bus = run([[(400, 100, 40, 136)] for _ in range(120)])
    before = len(bus.of("visit"))
    pipe.close(T0 + 8.0)
    assert len(bus.of("visit")) > before, "an open visit was lost at shutdown"


def test_close_on_an_idle_pipeline_is_harmless():
    pipe, _ = run(sc.empty(10))
    pipe.close()


def test_an_empty_store_emits_only_the_heartbeat():
    _, bus = run(sc.empty(60))
    assert {e.event_type for e in bus.events} <= {"occupancy"}
    assert {e.payload["count"] for e in bus.of("occupancy")} == {0}


def test_a_frame_from_another_stream_is_ignored():
    bus = Recorder()
    pipe = ShopperPipeline(CFG, bus)
    pipe.on_frame(sc.frame_from_boxes(0, [(100, 100, 40, 136)], stream="shelf"), T0)
    assert bus.events == []


def test_sixty_four_blobs_do_not_crash_the_pipeline():
    _, bus = run(sc.too_many_blobs(n=40, count=70))
    assert bus.of("occupancy")
