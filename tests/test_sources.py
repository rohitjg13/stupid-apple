"""sources/file.py and sources/camera.py."""
import numpy as np
import pytest

import cv2

from core.config import load_clipset
from sources.camera import CameraSource
from sources.file import FileSource

CFG = load_clipset("config/sim")


def make_clip(path, n=12, size=(640, 480), fourcc="MJPG"):
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 15, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 20, np.uint8)
        cv2.rectangle(f, (i * 20, 100), (i * 20 + 60, 240), (240, 240, 240), -1)
        vw.write(f)
    vw.release()
    return path


@pytest.fixture
def clip(tmp_path):
    return make_clip(tmp_path / "c.avi")


def cfg_with(tmp_path, clip, **over):
    cfg = load_clipset("config/sim")
    cfg.streams["overhead"] = {"source": "file", "path": str(clip), "fps": 15, **over}
    return cfg


def test_reads_frames_at_full_res(tmp_path, clip):
    src = FileSource(cfg_with(tmp_path, clip), "overhead", realtime=False)
    got = [f for _, f in zip(range(5), src.frames())]
    src.close()
    assert len(got) == 5
    for f in got:
        assert f.image.shape == (480, 640, 3)
        assert f.stream == "overhead"


def test_frame_ids_increment_from_zero(tmp_path, clip):
    src = FileSource(cfg_with(tmp_path, clip), "overhead", realtime=False)
    ids = [f.frame_id for _, f in zip(range(6), src.frames())]
    src.close()
    assert ids == list(range(6))


def test_letterboxes_odd_resolutions(tmp_path):
    clip = make_clip(tmp_path / "wide.avi", size=(1280, 720))
    src = FileSource(cfg_with(tmp_path, clip), "overhead", realtime=False)
    f = next(iter(src.frames()))
    src.close()
    assert f.image.shape == (480, 640, 3)
    assert not f.image[:60].any(), "expected black letterbox bars top and bottom"


def test_loop_true_runs_past_the_end(tmp_path, clip):
    src = FileSource(cfg_with(tmp_path, clip, loop=True), "overhead", realtime=False)
    got = [f for _, f in zip(range(30), src.frames())]      # clip is only 12 frames
    src.close()
    assert len(got) == 30
    assert got[-1].frame_id == 29, "frame_id keeps counting across loops"


def test_loop_false_stops_at_the_end(tmp_path, clip):
    src = FileSource(cfg_with(tmp_path, clip, loop=False), "overhead", realtime=False)
    got = list(src.frames())
    src.close()
    assert 10 <= len(got) <= 12


def test_missing_file_is_a_clear_error(tmp_path):
    cfg = load_clipset("config/sim")
    cfg.streams["overhead"] = {"source": "file", "path": str(tmp_path / "nope.avi"), "fps": 15}
    with pytest.raises(FileNotFoundError):
        FileSource(cfg, "overhead")


def test_t_ns_is_monotonic(tmp_path, clip):
    src = FileSource(cfg_with(tmp_path, clip, loop=True), "overhead", realtime=False)
    ts = [f.t_ns for _, f in zip(range(20), src.frames())]
    src.close()
    assert ts == sorted(ts)


# ---- camera --------------------------------------------------------------

class FakeCapture:
    """Stands in for cv2.VideoCapture; unplugs after `fail_after` reads."""

    def __init__(self, fail_after=None):
        self.fail_after, self.reads, self.released = fail_after, 0, False

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.full((480, 640, 3), 40, np.uint8)

    def set(self, *a):
        return True

    def release(self):
        self.released = True


def test_camera_reconnects_when_it_is_unplugged(monkeypatch):
    """The live shelf camera will get bumped on stage."""
    opened = []

    def fake_open(*a, **kw):
        cap = FakeCapture(fail_after=3 if not opened else None)
        opened.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", fake_open)
    src = CameraSource(CFG, "shelf", realtime=False, reconnect_delay=0.0)
    got = [f for _, f in zip(range(8), src.frames())]
    src.close()
    assert len(got) == 8
    assert len(opened) >= 2, "did not reopen the device after it went away"


def test_camera_gives_up_loudly_after_repeated_failures(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: FakeCapture(fail_after=0))
    src = CameraSource(CFG, "shelf", realtime=False, reconnect_delay=0.0, max_reconnects=2)
    with pytest.raises(RuntimeError, match="camera"):
        list(src.frames())
    src.close()
