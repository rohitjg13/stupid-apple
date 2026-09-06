"""core/clock.py — the board has no RTC and may boot with no network."""
import json
import time

import pytest

from core.clock import BUILD_EPOCH, Clock


def test_sane_system_clock_is_trusted(tmp_path):
    c = Clock(tmp_path / "clock.json")
    assert c.synced
    assert abs(c.now() - time.time()) < 1.0
    assert c.health()["clock_unsynced"] is False


def test_clock_behind_last_seen_is_unsynced_and_continues_forward(tmp_path):
    state = tmp_path / "clock.json"
    future = time.time() + 86400
    state.write_text(json.dumps({"last_seen": future}))
    c = Clock(state, _system_time=lambda: time.time())
    assert not c.synced
    assert c.now() > future, "must continue from the last known time, not go backwards"
    assert c.health()["clock_unsynced"] is True


def test_epoch_zero_boot_is_unsynced(tmp_path):
    """RTC-less board with no NTP boots somewhere around 1970."""
    c = Clock(tmp_path / "clock.json", _system_time=lambda: 1000.0)
    assert not c.synced
    assert c.now() >= BUILD_EPOCH


def test_now_never_goes_backwards_even_if_the_system_clock_jumps(tmp_path):
    t = [time.time()]
    c = Clock(tmp_path / "clock.json", _system_time=lambda: t[0])
    first = c.now()
    t[0] -= 3600                       # NTP steps the clock backwards mid-run
    assert c.now() >= first


def test_state_is_persisted_and_reloaded(tmp_path):
    state = tmp_path / "clock.json"
    c = Clock(state)
    c.persist()
    assert json.loads(state.read_text())["last_seen"] > BUILD_EPOCH
    assert Clock(state).synced


def test_persist_survives_a_read_only_filesystem(tmp_path):
    """A full SD card must not take the demo down."""
    c = Clock(tmp_path / "nope" / "deep" / "clock.json")
    c.persist()                        # must not raise
    assert c.now() > 0


def test_adopting_a_real_time_marks_the_clock_synced(tmp_path):
    """Backend lead's cloud sync waits for this before uploading."""
    c = Clock(tmp_path / "clock.json", _system_time=lambda: 1000.0)
    assert not c.synced
    c.adopt(time.time())
    assert c.synced
    assert abs(c.now() - time.time()) < 1.0


def test_clock_does_no_networking():
    import ast

    import core.clock as m
    tree = ast.parse(open(m.__file__).read())
    names = {n.name.split(".")[0] for node in ast.walk(tree)
             if isinstance(node, ast.Import) for n in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module}
    assert not names & {"socket", "urllib", "requests", "http", "ntplib"}, names
