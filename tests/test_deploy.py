"""The board scripts and the driver hand-off. Cheap checks that catch real breakage."""
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core.config import load_clipset
from pl.contract import FRAME_RESULT_DT

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "tools").glob("*.sh")))
def test_shell_scripts_parse(script):
    assert subprocess.run(["bash", "-n", ROOT / "tools" / script]).returncode == 0


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "tools").glob("*.sh")))
def test_shell_scripts_are_executable_and_strict(script):
    p = ROOT / "tools" / script
    assert p.stat().st_mode & 0o111, "not executable"
    assert "set -euo pipefail" in p.read_text()


def test_service_starts_main_with_a_state_dir_setup_creates():
    svc = (ROOT / "retail.service").read_text()
    setup = (ROOT / "tools" / "setup_board.sh").read_text()
    state_dirs = set(re.findall(r"(/var/lib/retail)/\w+\.json", svc))
    assert state_dirs, "service writes no state"
    for d in state_dirs:
        assert d in setup, f"{d} is never created by setup_board.sh"


def test_service_does_not_wait_for_the_network():
    """Fully functional with no network (SHARED.md §1)."""
    svc = (ROOT / "retail.service").read_text()
    directives = [l for l in svc.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not any("network-online.target" in l for l in directives)
    assert "Restart=always" in svc


def test_service_flags_all_exist_in_main():
    svc = (ROOT / "retail.service").read_text()
    flags = set(re.findall(r"(--[a-z-]+)", svc.split("ExecStart=")[1]))
    out = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"],
                         capture_output=True, text=True).stdout
    for f in flags:
        assert f in out, f"retail.service passes {f}, which main.py does not accept"


# ---- driver hand-off (Khushwant owns pl/driver.py; this pins what main.py needs) ----

def fake_driver_module(available=True, class_style=True):
    m = types.ModuleType("pl.driver")

    def process(image, stream_id=0, frame_id=0):
        r = np.zeros(1, dtype=FRAME_RESULT_DT)[0]
        r["frame_id"] = frame_id
        return r

    m.available = lambda: available
    if class_style:
        class Driver:
            def __init__(self, cfg=None):
                pass

            def read_latency_cycles(self):
                return 123
        Driver.process = staticmethod(process)
        m.Driver = Driver
    else:
        m.process = process
        m.read_latency_cycles = lambda: 123
    return m


@pytest.mark.parametrize("class_style", [True, False])
def test_main_accepts_either_driver_shape(monkeypatch, class_style):
    """His plan says driver.process(); either a module or a Driver class must work."""
    import main
    monkeypatch.setitem(sys.modules, "pl.driver", fake_driver_module(class_style=class_style))
    be = main.build_backend("pl", load_clipset("config/sim"))
    r = be.process(np.zeros((480, 640, 3), np.uint8), 0, 9)
    assert int(r["frame_id"]) == 9
    assert be.read_latency_cycles() == 123


def test_pl_backend_refused_when_driver_reports_unavailable(monkeypatch):
    import main
    monkeypatch.setitem(sys.modules, "pl.driver", fake_driver_module(available=False))
    with pytest.raises(SystemExit):
        main.check_backend("pl")
