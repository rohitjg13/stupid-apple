"""Contract A: FrameResult, 856 bytes, frozen. docs/SHARED.md §4."""
import numpy as np
import pytest

from pl.contract import BLOB_DT, FRAME_RESULT_DT, Frame, blob_foot_point


def test_sizes_are_frozen():
    assert BLOB_DT.itemsize == 12
    assert FRAME_RESULT_DT.itemsize == 856


def test_field_layout_and_types():
    f = FRAME_RESULT_DT.fields
    assert f["frame_id"][0] == np.dtype("<u4") and f["frame_id"][1] == 0
    assert f["num_blobs"][0] == np.dtype("<u2") and f["num_blobs"][1] == 4
    assert f["flags"][0] == np.dtype("<u2") and f["flags"][1] == 6
    assert f["blobs"][1] == 8
    assert FRAME_RESULT_DT["blobs"].shape == (64,)
    assert FRAME_RESULT_DT["roi_fill"].shape == (64,)
    assert FRAME_RESULT_DT["lane_occupancy"].shape == (16,)


def test_bytes_round_trip():
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    r["frame_id"] = 4242
    r["num_blobs"] = 2
    r["flags"] = 0b11
    r["blobs"][0] = (100, 200, 40, 80, 3200)
    r["blobs"][1] = (10, 20, 4, 8, 32)
    r["roi_fill"][3] = 255
    r["lane_occupancy"][15] = 100

    raw = r.tobytes()
    assert len(raw) == 856
    back = np.frombuffer(raw, dtype=FRAME_RESULT_DT)[0]
    assert back["frame_id"] == 4242
    assert back["num_blobs"] == 2
    assert back["flags"] == 0b11
    assert tuple(back["blobs"][0]) == (100, 200, 40, 80, 3200)
    assert back["roi_fill"][3] == 255
    assert back["lane_occupancy"][15] == 100


def test_foot_point_is_bottom_centre():
    b = np.zeros(1, dtype=BLOB_DT)[0]
    b["x"], b["y"], b["w"], b["h"] = 100, 200, 40, 80
    assert blob_foot_point(b) == (120, 280)


def test_frame_wrapper_and_to_dict():
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    r["frame_id"] = 7
    r["num_blobs"] = 1
    r["blobs"][0] = (1, 2, 3, 4, 12)
    fr = Frame(stream="overhead", frame_id=7, t_ns=123, result=r, image=None)

    d = fr.to_dict()
    assert d["stream"] == "overhead"
    assert d["frame_id"] == 7
    assert d["t_ns"] == 123
    assert len(d["blobs"]) == 1                      # truncated to num_blobs
    assert d["blobs"][0] == {"x": 1, "y": 2, "w": 3, "h": 4, "area": 12}
    assert "image" not in d                          # never serialise pixels
    import json; json.dumps(d)                       # must be JSON-safe


def test_frame_id_echoed_from_result():
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    r["frame_id"] = 9
    with pytest.raises(ValueError):
        Frame(stream="shelf", frame_id=8, t_ns=0, result=r, image=None)


def test_stream_must_be_known():
    r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
    with pytest.raises(ValueError):
        Frame(stream="kitchen", frame_id=0, t_ns=0, result=r, image=None)
