"""server.py — upload a clip, press start, read the dashboard back.

The whole demo path in one file. Everything runs against a temp data dir, the
reference backend, and a 60-frame synthetic clip, so it needs no GPU and no
network.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("RETAIL_DATA", "")        # replaced by the fixture below


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    data = tmp_path_factory.mktemp("data")
    os.environ["RETAIL_DATA"] = str(data)
    import server                                # imported after RETAIL_DATA is set
    assert server.DATA == data, "server must honour RETAIL_DATA"
    with TestClient(server.app) as c:
        c.server = server
        yield c


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A walker crossing the frame bottom to top, so tripwires have something to see."""
    import cv2
    path = tmp_path_factory.mktemp("clips") / "walk.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
    if not w.isOpened():
        pytest.skip("no mp4 encoder in this OpenCV build")
    for i in range(120):
        img = np.full((480, 640, 3), 30, np.uint8)
        y = 460 - i * 3
        cv2.rectangle(img, (300, y - 90), (350, y), (220, 220, 220), -1)
        w.write(img)
    w.release()
    return path


def upload(client, clip, roles=("overhead",), n=1):
    files = [("files", (f"clip{i}.mp4", clip.read_bytes(), "video/mp4")) for i in range(n)]
    return client.post("/api/runs", files=files,
                       data={"roles": list(roles)}).json()


def test_upload_probes_each_clip(client, clip):
    run = upload(client, clip)
    assert run["state"] == "uploaded"
    assert run["videos"][0]["role"] == "overhead"
    assert run["videos"][0]["frames"] == 120
    assert run["videos"][0]["fps"] == pytest.approx(15.0)


def test_upload_rejects_a_non_video(client, tmp_path):
    r = client.post("/api/runs", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400 and "not a video" in r.text


def test_upload_rejects_an_unknown_role(client, clip):
    r = client.post("/api/runs", files=[("files", ("a.mp4", clip.read_bytes(), "video/mp4"))],
                    data={"roles": ["ceiling"]})
    assert r.status_code == 400


def test_upload_sanitises_the_filename(client, clip):
    r = client.post("/api/runs",
                    files=[("files", ("../../etc/evil name.mp4", clip.read_bytes(),
                                      "video/mp4"))])
    name = r.json()["videos"][0]["name"]
    assert "/" not in name and ".." not in name


def test_preview_returns_the_first_frame(client, clip):
    run = upload(client, clip)
    r = client.get(f"/api/runs/{run['run_id']}/preview", params={"role": "overhead"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"          # JPEG magic


def test_preview_of_a_role_with_no_clip_is_404(client, clip):
    run = upload(client, clip)
    assert client.get(f"/api/runs/{run['run_id']}/preview",
                      params={"role": "shelf"}).status_code == 404


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_full_pass_populates_the_dashboard(client, clip):
    run = upload(client, clip, roles=("overhead", "shelf"), n=2)
    rid = run["run_id"]
    started = client.post(f"/api/runs/{rid}/start",
                          json={"backend": "reference", "shelf_rows": 2,
                                "shelf_cols": 3, "door_line": [[0, 300], [639, 300]]})
    assert started.status_code == 200, started.text
    assert started.json()["backend"] == "reference"

    for _ in range(600):                          # ~30 s ceiling; it takes about 2
        status = client.get(f"/api/runs/{rid}").json()
        if status["state"] in ("done", "error"):
            break
        import time as _t
        _t.sleep(0.05)
    assert status["state"] == "done", status.get("error")
    assert status["processed"] == 240             # both clips, every frame

    d = client.get("/api/dashboard", params={"run_id": rid}).json()
    assert d["run_id"] == rid
    assert d["backend"] == "reference"
    assert len(d["shelf"]) == 6                   # the 2x3 grid we asked for
    assert len(d["heatmap"]) == 16 * 12
    assert len(d["footfall_spark"]) == 12
    assert [f["label"] for f in d["funnel"]] == ["Footfall", "Browsed", "Engaged",
                                                 "Purchased"]
    assert d["queue"], "queue estimates should exist for both lanes"
    assert d["t1"] > d["t0"]


def test_dashboard_with_no_runs_at_all_is_empty(client):
    assert client.get("/api/dashboard", params={"run_id": "ghost"}).json()["shelf"] == []


def test_starting_a_run_twice_conflicts(client, clip, monkeypatch):
    run = upload(client, clip)
    client.server.runs[run["run_id"]]["state"] = "running"
    r = client.post(f"/api/runs/{run['run_id']}/start", json={"backend": "reference"})
    assert r.status_code == 409


def test_a_bad_shelf_grid_is_rejected_before_anything_runs(client, clip):
    run = upload(client, clip)
    r = client.post(f"/api/runs/{run['run_id']}/start",
                    json={"backend": "reference", "shelf_rows": 0})
    assert r.status_code == 400 and "1x1" in r.text


def test_delete_removes_the_media_and_the_rows(client, clip):
    run = upload(client, clip)
    rid = run["run_id"]
    media = client.server.DATA / "runs" / rid
    assert media.is_dir()
    assert client.delete(f"/api/runs/{rid}").status_code == 200
    assert not media.exists()
    assert client.get(f"/api/runs/{rid}").status_code == 404


def test_runs_are_listed(client, clip):
    upload(client, clip)
    assert client.get("/api/runs").json()
