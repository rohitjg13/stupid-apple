"""shopper/heatmap.py -- accumulation, the 10 s publish, and live decay."""
from shopper.heatmap import Heatmap
from shopper.params import TrackerParams

P = TrackerParams(heatmap_period_s=10.0, heatmap_decay_per_s=0.98)


def test_cells_accumulate():
    h = Heatmap(P)
    for _ in range(3):
        h.add((4, 5))
    h.add((6, 7))
    assert h.counts == {(4, 5): 3, (6, 7): 1}


def test_nothing_is_published_before_the_period_elapses():
    h = Heatmap(P)
    h.tick(0.0)
    h.add((1, 1))
    assert h.tick(5.0) is None


def test_a_payload_arrives_once_the_period_elapses():
    h = Heatmap(P)
    h.tick(0.0)
    h.add((1, 1))
    out = h.tick(10.5)
    assert out is not None
    assert out["tiles"] == [[1, 1, 1]]
    assert out["t_bucket"] == 0


def test_the_payload_matches_the_frozen_event_keys():
    """docs/SHARED.md §5: heatmap payload is exactly {t_bucket, tiles}."""
    h = Heatmap(P)
    assert set(h.payload(0.0)) == {"t_bucket", "tiles"}


def test_t_bucket_is_the_minute():
    h = Heatmap(P)
    assert h.payload(125.0)["t_bucket"] == 2


def test_the_published_grid_does_not_decay():
    """Historical totals must stay totals, or the numbers stop meaning anything."""
    h = Heatmap(P)
    h.tick(0.0)
    h.add((1, 1))
    for t in range(1, 40):
        h.tick(float(t))
    assert h.counts[(1, 1)] == 1


def test_the_live_grid_decays():
    h = Heatmap(P)
    h.tick(0.0)
    h.add((1, 1))
    h.tick(1.0)
    before = dict(h.live)
    h.tick(30.0)
    assert h.live.get((1, 1), 0) < before[(1, 1)]


def test_the_live_grid_forgets_cold_cells_entirely():
    h = Heatmap(P)
    h.tick(0.0)
    h.add((1, 1))
    h.tick(1.0)
    h.tick(400.0)
    assert h.live == {}, "a decayed grid must not grow without bound"


def test_live_tiles_are_serialisable():
    h = Heatmap(P)
    h.tick(0.0)
    h.add((2, 3))
    h.tick(1.0)
    tiles = h.live_tiles()
    assert tiles and all(len(t) == 3 for t in tiles)


def test_an_empty_heatmap_publishes_an_empty_tile_list():
    h = Heatmap(P)
    h.tick(0.0)
    assert h.tick(11.0)["tiles"] == []
