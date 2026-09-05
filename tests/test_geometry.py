"""geometry/ — image px <-> floor metres, zone lookup, heatmap cells. SHARED.md §6."""
import numpy as np
import pytest

from core.config import load_clipset
from geometry.homography import floor_to_image, image_to_floor
from geometry.zones import HEATMAP_CELL_M, ZoneMap

CFG = load_clipset("config/sim")
H = CFG.homography["overhead"]


def test_round_trip_single_point():
    uv = (320.0, 400.0)
    assert np.allclose(floor_to_image(H, image_to_floor(H, uv)), uv)


def test_known_scale_of_the_sim_floor():
    """config/sim is exactly 80 px/m with floor Y=0 at the bottom of the image."""
    assert np.allclose(image_to_floor(H, (0.0, 480.0)), (0.0, 0.0))
    assert np.allclose(image_to_floor(H, (80.0, 480.0)), (1.0, 0.0))
    assert np.allclose(image_to_floor(H, (0.0, 400.0)), (0.0, 1.0))


def test_batch_matches_loop():
    uv = np.array([[10.0, 20.0], [300.0, 450.0], [639.0, 479.0]])
    batch = image_to_floor(H, uv)
    assert batch.shape == (3, 2)
    for i, p in enumerate(uv):
        assert np.allclose(batch[i], image_to_floor(H, p))


def test_batch_round_trip():
    uv = np.random.default_rng(0).uniform([0, 0], [640, 480], size=(50, 2))
    assert np.allclose(floor_to_image(H, image_to_floor(H, uv)), uv)


def test_rejects_a_non_3x3():
    with pytest.raises(ValueError):
        image_to_floor(np.eye(2), (0.0, 0.0))


# ---- zones ---------------------------------------------------------------

def test_zone_of_finds_the_right_zone():
    zm = ZoneMap(CFG.zones)
    assert zm.zone_of(1.0, 1.0) == "entrance"      # [[0,0],[3,0],[3,2],[0,2]]
    assert zm.zone_of(1.0, 3.0) == "aisle_a"
    assert zm.zone_of(6.0, 3.0) == "aisle_b"
    assert zm.zone_of(6.0, 1.0) == "checkout"


def test_point_outside_every_zone_is_none():
    zm = ZoneMap(CFG.zones)
    assert zm.zone_of(-5.0, -5.0) is None
    assert zm.zone_of(100.0, 100.0) is None


def test_raster_agrees_with_true_polygon_containment():
    """The 0.25 m raster is an approximation; it must not be a wrong one.

    Only points more than one cell from a boundary are asserted -- inside that
    band the raster is allowed to round either way.
    """
    import cv2
    zm = ZoneMap(CFG.zones)
    rng = np.random.default_rng(1)
    pts = rng.uniform([0, 0], [8, 6], size=(1500, 2))
    checked = 0
    for zone in CFG.zones:
        contour = np.asarray(zone["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        for X, Y in pts:
            d = cv2.pointPolygonTest(contour, (float(X), float(Y)), True)
            if abs(d) <= HEATMAP_CELL_M:
                continue
            checked += 1
            assert (d > 0) == (zm.zone_of(X, Y) == zone["id"]), (X, Y, zone["id"], d)
    assert checked > 4000, "test barely exercised the raster"


def test_zone_lookup_is_vectorised():
    zm = ZoneMap(CFG.zones)
    ids = zm.zone_of_many(np.array([[1.0, 1.0], [1.0, 3.0], [-9.0, -9.0]]))
    assert ids == ["entrance", "aisle_a", None]


def test_heatmap_index_is_quarter_metre():
    zm = ZoneMap(CFG.zones)
    assert zm.heatmap_index(0.0, 0.0) == (0, 0)
    assert zm.heatmap_index(0.24, 0.0) == (0, 0)
    assert zm.heatmap_index(0.26, 0.0) == (1, 0)
    assert zm.heatmap_index(1.0, 2.0) == (4, 8)


def test_heatmap_index_handles_negatives_without_aliasing():
    """A tracker glitch outside the floor must not land in cell (0,0)."""
    zm = ZoneMap(CFG.zones)
    assert zm.heatmap_index(-0.1, -0.1) == (-1, -1)


def test_no_shapely_on_the_board():
    """Board deps are fixed (CLAUDE.md); zone lookup must stay numpy + cv2."""
    import ast
    import geometry.zones as z
    tree = ast.parse(open(z.__file__).read())
    imported = {n.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"__future__", "numpy", "cv2"}, imported
