"""tools/autoconfig.py — a clipset generated from footage must pass load_clipset."""
import numpy as np
import pytest

from core.config import load_clipset
from geometry.homography import image_to_floor
from pl.regs import MAX_LANES, MAX_ROIS
from tools import autoconfig


@pytest.fixture
def clip(tmp_path):
    """A tiny two-frame mp4; enough for cv2 to report fps and frame count."""
    import cv2
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(4):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    if not path.exists():
        pytest.skip("no mp4 encoder in this OpenCV build")
    return path


def test_generated_clipset_loads(tmp_path, clip):
    dest = autoconfig.make_clipset(tmp_path / "cs", overhead=[clip], shelf=[clip])
    cfg = load_clipset(dest)
    assert cfg.streams["overhead"]["paths"] == [str(clip)]
    assert cfg.streams["overhead"]["fps"] == pytest.approx(10.0)
    assert len(cfg.rois) == 6 and len(cfg.planogram) == 6
    assert {p["facing"] for p in cfg.planogram} == {r["id"] for r in cfg.rois}


def test_several_overhead_clips_become_one_playlist(tmp_path, clip):
    dest = autoconfig.make_clipset(tmp_path / "cs", overhead=[clip, clip, clip])
    assert len(load_clipset(dest).streams["overhead"]["paths"]) == 3


def test_no_shelf_video_binds_the_shelf_stream_to_the_camera(tmp_path, clip):
    cfg = load_clipset(autoconfig.make_clipset(tmp_path / "cs", overhead=[clip]))
    assert cfg.streams["shelf"]["source"] == "camera"


def test_shelf_grid_is_configurable_and_bounded(tmp_path, clip):
    cfg = load_clipset(autoconfig.make_clipset(tmp_path / "cs", overhead=[clip],
                                               shelf=[clip], shelf_grid=(4, 8)))
    assert len(cfg.rois) == 32
    with pytest.raises(ValueError):
        autoconfig.rois(9, 8)                      # 72 > the 64-entry ROI table


def test_tables_fit_the_hardware(tmp_path, clip):
    cfg = load_clipset(autoconfig.make_clipset(tmp_path / "cs", overhead=[clip],
                                               shelf=[clip], shelf_grid=(8, 8)))
    assert len(cfg.rois) == MAX_ROIS and len(cfg.lanes) == MAX_LANES
    with pytest.raises(ValueError):
        autoconfig.lanes(4, 8)                     # 32 > the 16-entry lane table


def test_door_line_comes_from_the_wizard(tmp_path, clip):
    dest = autoconfig.make_clipset(tmp_path / "cs", overhead=[clip],
                                   door_line=[[10, 300], [600, 310]])
    wire = load_clipset(dest).tripwires[0]
    assert wire["p1"] == [10, 300] and wire["p2"] == [600, 310]
    assert wire["in_dir"] == "up"


def test_a_vertical_door_line_enters_sideways(tmp_path, clip):
    dest = autoconfig.make_clipset(tmp_path / "cs", overhead=[clip],
                                   door_line=[[100, 20], [110, 460]])
    assert load_clipset(dest).tripwires[0]["in_dir"] == "right"


def test_homography_maps_the_frame_onto_the_floor():
    H = autoconfig.homography((8.0, 6.0))
    assert image_to_floor(H, (0, 480)) == pytest.approx([0.0, 0.0])       # near-left
    assert image_to_floor(H, (640, 0)) == pytest.approx([8.0, 6.0])       # far-right


def test_zones_cover_the_floor_and_name_the_entrance():
    ids = [z["id"] for z in autoconfig.zones()]
    assert "entrance" in ids                       # shopper/params.py excludes it
    assert "checkout" in ids


def test_a_clipset_with_no_videos_at_all_still_validates(tmp_path):
    cfg = load_clipset(autoconfig.make_clipset(tmp_path / "cs"))
    assert cfg.streams["overhead"]["source"] == "camera"


def test_undecodable_video_fails_loudly(tmp_path):
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="cannot decode"):
        autoconfig.make_clipset(tmp_path / "cs", overhead=[tmp_path / "junk.mp4"])


def test_a_degenerate_grid_from_the_wizard_is_rejected(tmp_path, clip):
    with pytest.raises(ValueError, match="at least 1x1"):
        autoconfig.make_clipset(tmp_path / "cs", overhead=[clip], shelf_grid=(0, 3))
    with pytest.raises(ValueError, match="at least 1x1"):
        autoconfig.lanes(2, 0)


def test_every_generated_box_sits_inside_the_pl_frame(tmp_path, clip):
    # load_clipset is the enforcer; a grid it rejects would strand a demo.
    for grid in ((1, 1), (2, 3), (5, 7), (8, 8)):
        dest = autoconfig.make_clipset(tmp_path / f"cs{grid}", overhead=[clip],
                                       shelf=[clip], shelf_grid=grid)
        load_clipset(dest)
