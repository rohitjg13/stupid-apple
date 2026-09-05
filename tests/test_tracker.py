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
