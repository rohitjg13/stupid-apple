"""Load and validate a clipset folder. Typos die here, not on stage."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from pl.regs import MAX_LANES, MAX_ROIS, PL_H, PL_W


class ConfigError(Exception):
    """Bad clipset. The message names the file and the offending entry."""


REQUIRED = {
    "rois.json":      {"id", "stream", "x", "y", "w", "h", "empty_below", "low_below"},
    "lanes.json":     {"id", "lane", "cell", "x", "y", "w", "h"},
    "zones.json":     {"id", "polygon"},
    "tripwires.json": {"id", "stream", "p1", "p2", "in_dir"},
    "planogram.json": {"facing", "sku", "name", "unit_price", "expected_sales_per_hour"},
}
STREAMS = ("overhead", "shelf")


@dataclass
class Clipset:
    path: Path
    store_id: str
    streams: dict
    checkout: dict
    pos: dict
    cloud: dict
    rois: list
    lanes: list
    zones: list
    tripwires: list
    planogram: list
    homography: dict = field(default_factory=dict)

    @property
    def roi_index(self) -> dict:
        """roi id -> its index in roi_fill[] / the PL ROI_TABLE."""
        return {r["id"]: i for i, r in enumerate(self.rois)}

    @property
    def lane_index(self) -> dict:
        return {l["id"]: i for i, l in enumerate(self.lanes)}


def _fail(name, msg, entry=None):
    where = f" ({entry})" if entry is not None else ""
    raise ConfigError(f"{name}: {msg}{where}")


def _load_json(root: Path, name: str) -> list:
    p = root / name
    if not p.exists():
        _fail(name, f"missing from {root}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        _fail(name, f"is not valid JSON: {e}")
    if not isinstance(data, list):
        _fail(name, "must be a JSON list")
    for entry in data:
        missing = REQUIRED[name] - set(entry)
        if missing:
            _fail(name, f"entry is missing {sorted(missing)}", entry.get("id") or entry.get("facing"))
    return data


def _check_unique(name, entries, key="id"):
    seen = set()
    for e in entries:
        if e[key] in seen:
            _fail(name, f"duplicate {key}", e[key])
        seen.add(e[key])


def _check_box(name, e):
    x, y, w, h = (int(e[k]) for k in ("x", "y", "w", "h"))
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > PL_W or y + h > PL_H:
        _fail(name, f"box ({x},{y},{w},{h}) is outside the {PL_W}x{PL_H} PL frame "
                    f"— author tables at PL scale, not 640x480", e["id"])


def load_clipset(path) -> Clipset:
    root = Path(path)
    if not root.is_dir():
        raise ConfigError(f"clipset folder not found: {root}")

    store_p = root / "store.yaml"
    if not store_p.exists():
        _fail("store.yaml", f"missing from {root}")
    store = yaml.safe_load(store_p.read_text()) or {}
    for k in ("store_id", "streams"):
        if k not in store:
            _fail("store.yaml", f"missing {k!r}")
    for s in STREAMS:
        if s not in store["streams"]:
            _fail("store.yaml", f"streams is missing {s!r}")

    rois, lanes = _load_json(root, "rois.json"), _load_json(root, "lanes.json")
    zones, tripwires = _load_json(root, "zones.json"), _load_json(root, "tripwires.json")
    planogram = _load_json(root, "planogram.json")

    if len(rois) > MAX_ROIS:
        _fail("rois.json", f"{len(rois)} ROIs exceeds the hardware table of {MAX_ROIS}")
    if len(lanes) > MAX_LANES:
        _fail("lanes.json", f"{len(lanes)} lane cells exceeds the hardware table of {MAX_LANES}")

    _check_unique("rois.json", rois)
    _check_unique("lanes.json", lanes)
    _check_unique("zones.json", zones)
    _check_unique("tripwires.json", tripwires)
    _check_unique("planogram.json", planogram, key="facing")

    for r in rois:
        _check_box("rois.json", r)
        if r["stream"] not in STREAMS:
            _fail("rois.json", f"unknown stream {r['stream']!r}", r["id"])
        if not 0 <= r["empty_below"] < r["low_below"] <= 255:
            _fail("rois.json", "needs 0 <= empty_below < low_below <= 255", r["id"])
    for l in lanes:
        _check_box("lanes.json", l)
    for z in zones:
        if len(z["polygon"]) < 3:
            _fail("zones.json", "polygon needs at least 3 points", z["id"])
    for t in tripwires:
        if t["stream"] not in STREAMS:
            _fail("tripwires.json", f"unknown stream {t['stream']!r}", t["id"])
        if t["in_dir"] not in ("up", "down", "left", "right"):
            _fail("tripwires.json", f"bad in_dir {t['in_dir']!r}", t["id"])

    roi_ids = {r["id"] for r in rois}
    for p in planogram:
        if p["facing"] not in roi_ids:
            _fail("planogram.json", f"facing {p['facing']!r} is not an id in rois.json", p["sku"])

    homography = {s: np.load(root / f"homography_{s}.npy")
                  for s in STREAMS if (root / f"homography_{s}.npy").exists()}

    return Clipset(root, store["store_id"], store["streams"], store.get("checkout", {}),
                   store.get("pos", {}), store.get("cloud", {}),
                   rois, lanes, zones, tripwires, planogram, homography)
