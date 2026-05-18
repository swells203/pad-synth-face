import numpy as np
from fastapi.testclient import TestClient

from defid_demo.app import app


def _rep(rng, speed, dwell):
    ptr, ts, x, y = [], 1000.0, 50.0, 50.0
    for i in range(45):
        x += speed + rng.normal(0, 0.4)
        y += speed * 0.4 + rng.normal(0, 0.4)
        ts += 16.0 + rng.normal(0, 2.0)
        ptr.append({"x": x, "y": y, "ts": ts})
    keys, kt = [], 3000.0
    for i in range(8):
        keys.append({"code": f"K{i}", "phase": "down", "ts": kt})
        kt += dwell * 1000.0
        keys.append({"code": f"K{i}", "phase": "up", "ts": kt})
        kt += 180.0 + rng.normal(0, 20.0)
    return {"pointer": ptr, "keys": keys}


def test_index_and_spectator_served():
    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert "DefinitiveID" in c.get("/").text
    assert c.get("/spectator").status_code == 200


def test_reset_enroll_calibrate_attempt_roundtrip():
    c = TestClient(app)
    rng = np.random.default_rng(7)
    assert c.post("/api/reset").status_code == 200
    for _ in range(8):
        r = c.post("/api/enroll", json=_rep(rng, 2.0, 0.09))
        assert r.status_code == 200 and r.json()["ok"]
    cal = c.post("/api/calibrate").json()
    assert cal["ok"] and cal["threshold"] > 0
    a = c.post("/api/attempt", json=_rep(rng, 2.0, 0.09)).json()
    assert a["verdict"] in ("ACCEPT", "REJECT")
    assert "distances" in a
    assert len(a["feature_values"]) == 9
    st = c.get("/api/state").json()
    assert st["enroll_reps"] == 8 and st["calibrated"] is True
