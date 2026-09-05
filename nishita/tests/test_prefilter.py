"""shopper/blob_prefilter.py -- fragments back together, noise thrown away."""
import numpy as np

from pl.contract import BLOB_DT
from shopper.blob_prefilter import prefilter
from shopper.params import TrackerParams
from tests.fixtures import scenarios as sc


def boxes_of(arr):
    return [tuple(int(b[k]) for k in ("x", "y", "w", "h")) for b in arr]


def blobs(box_list):
    return sc.frame_from_boxes(0, box_list).blobs


def test_a_person_split_in_two_becomes_one_blob():
    got = prefilter(blobs(sc.split_person(n=1, split=(0, 1))[0]))
    assert boxes_of(got) == [(100, 200, 40, 136)]


def test_the_merged_area_is_the_whole_person_not_the_pieces():
    """The tracker scores on area, so a reassembled person must score as one."""
    got = prefilter(blobs(sc.split_person(n=1, split=(0, 1))[0]))
    assert int(got[0]["area"]) == 40 * 136


def test_a_person_split_into_three_collapses_to_one():
    three = [(100, 200, 40, 40), (100, 245, 40, 40), (100, 290, 40, 46)]
    assert len(prefilter(blobs(three))) == 1


def test_two_people_standing_apart_are_not_merged():
    two = [(100, 200, 40, 136), (300, 200, 40, 136)]
    assert len(prefilter(blobs(two))) == 2


def test_two_people_side_by_side_are_not_merged():
    """Adjacent horizontally, not stacked: no vertical-fragment relationship."""
    two = [(100, 200, 40, 136), (145, 200, 40, 136)]
    assert len(prefilter(blobs(two))) == 2


def test_a_wide_flat_shadow_is_dropped():
    assert len(prefilter(blobs(sc.noise_slivers(n=1)[0]))) == 0


def test_aspect_ratio_cutoff_is_configurable():
    sliver = [(300, 300, 120, 20)]                       # 6:1
    assert len(prefilter(blobs(sliver), TrackerParams(max_aspect=10.0))) == 1


def test_no_blobs_in_no_blobs_out():
    got = prefilter(blobs([]))
    assert len(got) == 0 and got.dtype == BLOB_DT


def test_output_keeps_the_contract_dtype_and_area_ordering():
    got = prefilter(blobs([(0, 0, 20, 30), (100, 100, 60, 90), (300, 0, 30, 40)]))
    assert got.dtype == BLOB_DT
    areas = [int(b["area"]) for b in got]
    assert areas == sorted(areas, reverse=True), "contract requires area-desc order"


def test_a_full_frame_of_blobs_does_not_hang():
    many = [((i * 9) % 600, (i * 7) % 440, 20, 25) for i in range(64)]
    assert len(prefilter(blobs(many))) <= 64


def test_merging_is_idempotent():
    once = prefilter(blobs(sc.split_person(n=1, split=(0, 1))[0]))
    assert boxes_of(prefilter(once)) == boxes_of(once)
