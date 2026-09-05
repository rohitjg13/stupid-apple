"""shopper/tracker.py -- association, lifecycle, and the failures that matter."""
import numpy as np
import pytest

from shopper.kalman import KalmanBox
from shopper.params import TrackerParams
from shopper.tracker import CONFIRMED, DELETED, LOST, Tracker, iou_matrix
from tests.fixtures import scenarios as sc

P = TrackerParams()


def run(box_lists, params=None, fps=15.0):
    """Feed a scenario through a tracker; return (tracker, per-frame active ids)."""
    tk = Tracker(params or TrackerParams())
    ids = []
    for i, boxes in enumerate(box_lists):
        frame = sc.frame_from_boxes(i, boxes, fps=fps)
        active = tk.update(frame.blobs, i / fps)
        ids.append([tr.id for tr in active])
    return tk, ids


# --- IoU ------------------------------------------------------------------
def test_iou_of_identical_boxes_is_one():
    assert iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10)])[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou_matrix([(0, 0, 10, 10)], [(50, 50, 10, 10)])[0, 0] == 0.0


def test_iou_half_overlap():
    # 10x10 boxes sharing exactly half their area: inter 50, union 150.
    got = iou_matrix([(0, 0, 10, 10)], [(5, 0, 10, 10)])[0, 0]
    assert got == pytest.approx(50 / 150)


def test_iou_matrix_handles_empty_sides():
    assert iou_matrix([], [(0, 0, 4, 4)]).shape == (0, 1)
    assert iou_matrix([(0, 0, 4, 4)], []).shape == (1, 0)


# --- lifecycle ------------------------------------------------------------
def test_a_track_confirms_only_after_n_init_hits():
    _, ids = run(sc.linear_walk(n=6))
    assert ids[0] == [] and ids[1] == [], "confirmed before n_init hits"
    assert ids[2] == [0], "not confirmed on the third hit"


def test_a_single_walker_keeps_one_id_throughout():
    _, ids = run(sc.linear_walk(n=30))
    seen = {i for frame in ids for i in frame}
    assert seen == {0}, f"expected one id, saw {seen}"


def test_ids_start_at_zero_and_increment():
    tk, _ = run(sc.linear_walk(n=5))
    assert [tr.id for tr in tk.tracks] == [0]


# --- the two failures the plan calls out ----------------------------------
def test_crossing_tracks_keep_their_ids():
    """Two people walking through each other must not swap ids."""
    _, ids = run(sc.crossing(n=40))
    settled = [frame for frame in ids if len(frame) == 2]
    assert settled, "never tracked both people at once"
    assert all(set(frame) == {0, 1} for frame in settled)
    # And both must survive to the end rather than one absorbing the other.
    assert set(ids[-1]) == {0, 1}


def test_a_ten_frame_disappearance_keeps_the_same_id():
    _, ids = run(sc.occlusion(n=40, gap=(15, 25)))
    before = ids[14]
    after = next(frame for frame in ids[25:] if frame)
    assert before == [0] and after == [0], f"id changed across the gap: {before} -> {after}"


def test_a_track_lost_for_longer_than_max_age_is_deleted():
    p = TrackerParams(max_age=5)
    _, ids = run(sc.occlusion(n=40, gap=(10, 30)), params=p)
    assert ids[-1] == [1], "expected a fresh id after the track aged out"


def test_a_lost_track_is_predicted_forward_not_frozen():
    tk, _ = run(sc.linear_walk(n=10))
    tr = tk.tracks[0]
    x_before = tr.box[0]
    for i in range(10, 13):
        tk.update(sc.frame_from_boxes(i, []).blobs, i / 15.0)
    assert tk.tracks[0].state == LOST
    assert tk.tracks[0].box[0] > x_before, "a lost track should coast, not freeze"


# --- contract edge cases --------------------------------------------------
def test_no_blobs_is_fine():
    tk, ids = run(sc.empty(10))
    assert ids == [[]] * 10 and tk.tracks == []


def test_sixty_four_blobs_and_the_truncation_flag_do_not_crash():
    frames = sc.sequence(sc.too_many_blobs(n=6, count=70))
    assert int(frames[0].result["num_blobs"]) == 64
    tk = Tracker()
    for i, f in enumerate(frames):
        tk.update(f.blobs, i / 15.0)
    assert len(tk.tracks) <= 64


def test_small_blobs_never_start_a_track():
    """Below the area pseudo-score they can only rescue an existing track."""
    tiny = [[(100 + 4 * i, 200, 12, 12)] for i in range(10)]     # area 144
    tk, ids = run(tiny)
    assert all(frame == [] for frame in ids)
    assert tk.tracks == []


def test_a_small_blob_can_still_rescue_an_established_track():
    """A partly-occluded person drops below the area score but keeps their id.

    Note the geometry: because area *is* the score here, a low-score blob is a
    small blob, and a small blob cannot have much IoU with a large one. So the
    second pass only rescues people whose blob sits near the threshold -- which
    is exactly who it is for.
    """
    walk = [[(100 + 6 * i, 200, 45, 40)] for i in range(8)]        # area 1800, high
    shrunk = [[(100 + 6 * i, 202, 38, 31)] for i in range(8, 14)]  # area 1178, low
    tk, ids = run(walk + shrunk)
    assert ids[-1] == [0], "the second pass should have held the track"


def test_deleted_tracks_are_reported_once():
    p = TrackerParams(max_age=2)
    tk = Tracker(p)
    for i, boxes in enumerate(sc.linear_walk(n=6)):
        tk.update(sc.frame_from_boxes(i, boxes).blobs, i / 15.0)
    seen = []
    for i in range(6, 12):
        tk.update(sc.frame_from_boxes(i, []).blobs, i / 15.0)
        seen += [tr.id for tr in tk.just_deleted]
    assert seen == [0], f"expected exactly one deletion, got {seen}"


def test_close_flushes_every_surviving_track():
    tk, _ = run(sc.linear_walk(n=10))
    assert [tr.id for tr in tk.close()] == [0]
    assert tk.tracks == []


# --- overlap grace --------------------------------------------------------
def test_both_people_survive_a_merge_and_come_out_with_the_same_ids():
    tk, ids = run(sc.merge_and_split(n=24, merge=(9, 15)))
    assert set(ids[8]) == {0, 1}, "not tracking both before the merge"
    assert set(ids[-1]) == {0, 1}, f"ids did not survive the merge: {ids[-1]}"
    assert len(tk.tracks) == 2, "a merge must not delete the occluded person"


def test_occupancy_does_not_dip_while_two_people_are_merged():
    """The whole point of overlap grace: one blob, still two shoppers."""
    _, ids = run(sc.merge_and_split(n=24, merge=(9, 15)))
    during = ids[9:15]
    assert all(len(frame) == 2 for frame in during), \
        f"count dropped during the merge: {[len(f) for f in during]}"


def test_a_shared_track_is_not_dragged_onto_the_other_person():
    """It coasts on its own velocity; overwriting it would fuse the two tracks."""
    tk, _ = run(sc.merge_and_split(n=24, merge=(9, 15)))
    a, b = sorted(tk.tracks, key=lambda t: t.id)
    assert abs(a.box[0] - b.box[0]) > 20, \
        f"tracks collapsed onto each other: {a.box[0]:.0f} vs {b.box[0]:.0f}"


def test_sharing_is_bounded_by_overlap_grace():
    p = TrackerParams(overlap_grace=3)
    _, ids = run(sc.merge_and_split(n=30, merge=(9, 25)), params=p)
    assert len(ids[12]) == 1, "sharing should have lapsed after overlap_grace"


# --- kalman ---------------------------------------------------------------
def test_kalman_converges_on_constant_velocity():
    kf = KalmanBox((100, 200, 40, 136))
    for i in range(1, 15):
        kf.predict()
        kf.update((100 + 6 * i, 200, 40, 136))
    assert kf.velocity[0] == pytest.approx(6.0, abs=0.5)
    assert kf.bbox[0] == pytest.approx(184, abs=2)


def test_kalman_covariance_stays_well_conditioned():
    kf = KalmanBox((100, 200, 40, 136))
    for i in range(200):
        kf.predict()
        kf.update((100 + 6 * i, 200, 40, 136))
    assert np.allclose(kf.P, kf.P.T), "covariance lost symmetry"
    assert (np.linalg.eigvalsh(kf.P) > 0).all(), "covariance is not positive definite"


def test_kalman_predicted_box_never_collapses():
    kf = KalmanBox((100, 200, 40, 136))
    for _ in range(500):
        kf.predict()
    assert kf.bbox[2] >= 1.0 and kf.bbox[3] >= 1.0


# --- standing still -------------------------------------------------------
# A background subtractor sees change, so it goes blind to anyone who stops
# moving -- exactly the shopper whose dwell time the product is built to
# measure. These cover holding them anyway.

def walk_then_stand(walk=10, stand=6, vanish=60, x0=100, y0=200, dx=6):
    """Walk in, stand still for a moment, then the blob is absorbed away."""
    boxes = [[(x0 + dx * i, y0, 40, 136)] for i in range(walk)]
    x = x0 + dx * (walk - 1)
    boxes += [[(x, y0, 40, 136)] for _ in range(stand)]
    return boxes + [[] for _ in range(vanish)]


def test_a_browser_absorbed_into_the_background_is_still_counted():
    _, ids = run(walk_then_stand())
    assert ids[-1] == [0], "lost the shopper the moment they stopped moving"


def test_a_held_browser_does_not_drift_away_from_where_they_stand():
    tk, _ = run(walk_then_stand())
    x_standing = 100 + 6 * 9
    assert abs(tk.tracks[0].box[0] - x_standing) < 10, \
        f"held track drifted to {tk.tracks[0].box[0]:.0f}, person is at {x_standing}"


def test_holding_is_bounded():
    p = TrackerParams(static_max_age=20)
    _, ids = run(walk_then_stand(vanish=80), params=p)
    assert ids[-1] == [], "a held track must eventually be given up on"


def test_holding_can_be_switched_off():
    _, ids = run(walk_then_stand(), params=TrackerParams(static_max_age=0))
    assert ids[-1] == []


def test_someone_who_walks_out_is_not_held():
    """Leaving requires moving, so a fast track that vanishes really has gone."""
    _, ids = run(sc.linear_walk(n=12, dx=9) + [[] for _ in range(60)])
    assert ids[-1] == [], "held a track that walked out of frame"


def test_a_one_frame_speed_blip_does_not_condemn_a_browser():
    """The hold must not be a one-way door; lost tracks can be reclaimed."""
    boxes = walk_then_stand(stand=3, vanish=60)
    tk, ids = run(boxes)
    assert ids[-1] == [0]
    assert tk.tracks[0].held > 0


def test_a_held_track_reattaches_when_the_person_moves_again():
    boxes = walk_then_stand(stand=4, vanish=30)
    boxes += [[(154 + 6 * i, 200, 40, 136)] for i in range(10)]
    tk, ids = run(boxes)
    assert ids[-1] == [0], "a browser who starts walking again should keep their id"


def test_a_blob_that_flickered_in_place_is_not_held_as_a_browser():
    """A basket catching the light appears, sits, vanishes. It never walked
    anywhere, so it is not a shopper who stopped -- do not hold it for 30 s."""
    flicker = [[(300, 300, 40, 136)] for _ in range(6)] + [[] for _ in range(40)]
    _, ids = run(flicker)
    assert ids[-1] == [], "held a stationary noise blob as if it were a shopper"


def test_a_real_browser_walked_to_the_shelf_and_is_held():
    """Same shape as the flicker, but preceded by a walk: that is a person."""
    _, ids = run(walk_then_stand(walk=10, stand=6, vanish=40))
    assert ids[-1] == [0]


def test_the_travel_requirement_is_configurable():
    flicker = [[(300, 300, 40, 136)] for _ in range(6)] + [[] for _ in range(40)]
    _, ids = run(flicker, params=TrackerParams(static_min_travel_px=0.0))
    assert ids[-1] == [0], "with no travel requirement the flicker should be held"


# --- furniture ------------------------------------------------------------
# A trolley wheel or a glossy basket that the background model keeps flagging
# gets a real match every frame, so it looks like a perfectly tracked person with
# zero speed. The hold logic never sees it. It has to be caught by the one thing
# a shopper always has and furniture never does: it walked here.

def test_a_static_object_flagged_every_frame_is_retired_as_furniture():
    static = [[(300, 300, 40, 136)] for _ in range(60)]
    tk, ids = run(static)
    assert ids[-1] == [], "counted a never-moving blob as a shopper"
    assert tk.tracks == []


def test_furniture_is_flagged_so_the_pipeline_can_skip_its_visit():
    tk = Tracker(TrackerParams())
    flags = []
    for i in range(40):
        tk.update(sc.frame_from_boxes(i, [(300, 300, 40, 136)]).blobs, i / 15.0)
        flags += [t.furniture for t in tk.just_deleted]
    assert flags == [True]


def test_a_newly_confirmed_walker_counts_before_walking_a_body_width():
    """The filter must not delay real people: at 3 hits they have moved 18 px."""
    _, ids = run(sc.linear_walk(n=6))
    assert ids[2] == [0]


def test_a_shopper_who_walked_in_and_stopped_is_never_mistaken_for_furniture():
    boxes = [[(100 + 6 * i, 200, 40, 136)] for i in range(10)]
    boxes += [[(154, 200, 40, 136)] for _ in range(90)]        # 6 s standing still
    _, ids = run(boxes)
    assert ids[-1] == [0], "retired a real browser as furniture"
    assert all(frame == [0] for frame in ids[3:]), "occupancy dipped while they browsed"


def test_furniture_retirement_age_is_configurable():
    static = [[(300, 300, 40, 136)] for _ in range(60)]
    _, ids = run(static, params=TrackerParams(furniture_age=10_000))
    assert ids[-1] == [0]


def test_retired_furniture_does_not_respawn_as_a_new_track_every_two_seconds():
    """Retiring is not enough: the blob is still there next frame. Without a
    memory of where furniture was, it respawns and is counted 23 frames in 26."""
    static = [[(300, 300, 40, 136)] for _ in range(200)]
    _, ids = run(static)
    after_first_retirement = ids[40:]
    assert all(frame == [] for frame in after_first_retirement), \
        f"furniture respawned: {sorted({i for f in after_first_retirement for i in f})}"


def test_furniture_memory_expires_once_the_object_stops_being_flagged():
    """While the basket keeps getting flagged the memory is refreshed on purpose.
    Once it is gone for longer than the memory, a blob there is a new arrival."""
    box = (300, 300, 40, 136)
    boxes = [[box] for _ in range(40)]          # retired as furniture around frame 26
    boxes += [[] for _ in range(40)]            # gone; memory (30 frames) lapses
    boxes += [[box] for _ in range(20)]         # something appears there again
    _, ids = run(boxes, params=TrackerParams(furniture_memory=30))
    assert ids[-1] != [], "memory never expired"


def test_a_real_person_walking_past_remembered_furniture_is_still_tracked():
    """Spawn suppression must only stop *new* tracks on the furniture, never an
    existing walker who passes through that spot."""
    static = (300, 300, 40, 136)
    boxes = [[static] for _ in range(40)]                          # furniture retires
    boxes += [[static, (100 + 8 * i, 300, 40, 136)] for i in range(40)]  # walker passes
    _, ids = run(boxes)
    walker_frames = [f for f in ids[45:]]
    assert all(len(f) == 1 for f in walker_frames), \
        f"lost the walker near furniture: {[len(f) for f in walker_frames]}"


def test_one_merge_blip_does_not_make_a_basket_look_like_it_walked():
    """A single frame where the basket's blob merged with a passer-by and the
    measurement jumped 150 px must not count as a journey."""
    static = (300, 300, 40, 136)
    boxes = [[static] for _ in range(10)] + [[(450, 300, 40, 136)]] + [[static] for _ in range(60)]
    tk, ids = run(boxes)
    assert ids[-1] == [], "one outlier measurement earned a basket a 30 s hold"


# --- id churn vs furniture ------------------------------------------------
def test_a_shopper_whose_track_was_recycled_while_standing_is_not_furniture():
    """Walk in, stand. The track dies (a bad frame), a fresh one is born on the
    same spot. That person walked here; the new track must know it."""
    p = TrackerParams(max_age=3)
    boxes = [[(100 + 6 * i, 200, 40, 136)] for i in range(10)]      # walks 54 px
    boxes += [[(154, 200, 40, 136)] for _ in range(5)]              # stands
    boxes += [[] for _ in range(6)]                                 # blob gone: track dies
    boxes += [[(154, 200, 40, 136)] for _ in range(90)]             # still standing
    tk, ids = run(boxes, params=p)
    assert ids[-1] != [], "the recycled track was retired as furniture"
    assert not any(t.furniture for t in tk.just_deleted)


def test_true_furniture_has_no_walking_predecessor_and_is_still_retired():
    static = [[(300, 300, 40, 136)] for _ in range(80)]
    _, ids = run(static, params=TrackerParams(max_age=3))
    assert ids[-1] == []
