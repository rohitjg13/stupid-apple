"""shopper/params.py -- tracker.yaml loading."""
import pytest

from shopper.params import MIN_BLOB_AREA_FULL_RES, TrackerParams


def test_defaults_load_with_no_clipset():
    assert TrackerParams.load() == TrackerParams()


def test_a_clipset_without_tracker_yaml_uses_defaults(tmp_path):
    assert TrackerParams.load(tmp_path) == TrackerParams()


def test_the_sim_clipset_overrides_are_picked_up():
    p = TrackerParams.load("config/sim")
    assert p.merge_x_tol == 0.95
    assert p.tripwire_deadband_px == 12.0


def test_unlisted_values_keep_their_defaults():
    assert TrackerParams.load("config/sim").n_init == TrackerParams().n_init


def test_a_typo_is_rejected_rather_than_silently_ignored(tmp_path):
    (tmp_path / "tracker.yaml").write_text("iou_hihg: 0.7\n")
    with pytest.raises(ValueError, match="unknown tracker.yaml keys"):
        TrackerParams.load(tmp_path)


def test_a_non_mapping_is_rejected(tmp_path):
    (tmp_path / "tracker.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        TrackerParams.load(tmp_path)


def test_an_empty_file_is_treated_as_no_overrides(tmp_path):
    (tmp_path / "tracker.yaml").write_text("")
    assert TrackerParams.load(tmp_path) == TrackerParams()


def test_the_area_pseudo_score_accounts_for_the_pl_scale_factor():
    """pl/reference.py reports area * SCALE^2, so the 150 register default is 600."""
    assert MIN_BLOB_AREA_FULL_RES == 600
    assert TrackerParams().high_area == 1200


def test_tracker_yaml_does_not_break_rohits_config_validation():
    """An extra file in the clipset folder must not upset load_clipset."""
    from core.config import load_clipset
    assert load_clipset("config/sim").store_id == "demo-01"
