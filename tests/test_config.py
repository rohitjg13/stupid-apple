"""core/config.py — validate a clipset folder so typos die at startup, not on stage."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from core.config import ConfigError, load_clipset

FIXTURE = Path(__file__).parent.parent / "config" / "sim"


@pytest.fixture
def clipset(tmp_path):
    dst = tmp_path / "cs"
    shutil.copytree(FIXTURE, dst)
    return dst


def rewrite(clipset, name, mutate):
    p = clipset / name
    data = yaml.safe_load(p.read_text()) if name.endswith(".yaml") else json.loads(p.read_text())
    data = mutate(data)
    p.write_text(yaml.safe_dump(data) if name.endswith(".yaml") else json.dumps(data))


def test_shipped_sim_clipset_is_valid():
    cfg = load_clipset(FIXTURE)
    assert cfg.store_id == "demo-01"
    assert set(cfg.streams) == {"overhead", "shelf"}
    assert cfg.rois and cfg.lanes and cfg.zones and cfg.tripwires and cfg.planogram


def test_roi_and_lane_ids_are_unique(clipset):
    rewrite(clipset, "rois.json", lambda r: r + [dict(r[0])])
    with pytest.raises(ConfigError, match="duplicate"):
        load_clipset(clipset)


def test_roi_outside_pl_frame_rejected(clipset):
    """The classic: authoring rois.json at 640x480 instead of 320x240."""
    rewrite(clipset, "rois.json", lambda r: [dict(r[0], w=640, h=480)] + r[1:])
    with pytest.raises(ConfigError):
        load_clipset(clipset)


def test_too_many_rois_rejected(clipset):
    rewrite(clipset, "rois.json", lambda r: [dict(r[0], id=f"R{i}") for i in range(65)])
    with pytest.raises(ConfigError, match="64"):
        load_clipset(clipset)


def test_too_many_lane_cells_rejected(clipset):
    rewrite(clipset, "lanes.json", lambda l: [dict(l[0], id=f"L{i}") for i in range(17)])
    with pytest.raises(ConfigError, match="16"):
        load_clipset(clipset)


def test_planogram_facing_must_be_a_real_roi(clipset):
    rewrite(clipset, "planogram.json", lambda p: [dict(p[0], facing="NOPE")])
    with pytest.raises(ConfigError, match="NOPE"):
        load_clipset(clipset)


def test_thresholds_must_be_ordered(clipset):
    rewrite(clipset, "rois.json", lambda r: [dict(r[0], empty_below=200, low_below=100)] + r[1:])
    with pytest.raises(ConfigError, match="empty_below"):
        load_clipset(clipset)


def test_missing_required_key_names_the_file_and_the_entry(clipset):
    def drop(r):
        r[0].pop("empty_below")
        return r
    rewrite(clipset, "rois.json", drop)
    with pytest.raises(ConfigError) as e:
        load_clipset(clipset)
    assert "rois.json" in str(e.value) and "empty_below" in str(e.value)


def test_tripwire_stream_must_exist(clipset):
    rewrite(clipset, "tripwires.json", lambda t: [dict(t[0], stream="basement")])
    with pytest.raises(ConfigError, match="basement"):
        load_clipset(clipset)


def test_missing_file_is_a_clear_error(clipset):
    (clipset / "lanes.json").unlink()
    with pytest.raises(ConfigError, match="lanes.json"):
        load_clipset(clipset)


def test_roi_index_matches_pl_table_order():
    """roi_fill[i] belongs to rois[i]; the shelf lead depends on this."""
    cfg = load_clipset(FIXTURE)
    assert cfg.roi_index[cfg.rois[0]["id"]] == 0
    assert list(cfg.roi_index.values()) == list(range(len(cfg.rois)))
