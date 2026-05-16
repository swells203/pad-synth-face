import numpy as np

from defid_demo.adapter import RepPayload
from defid_demo.service import DemoService


def _rep(rng, speed, dwell, jitter):
    """Synthesize a browser RepPayload for a motor profile."""
    pointer, ts = [], 1000.0
    x = y = 50.0
    for i in range(45):
        x += speed + rng.normal(0, jitter)
        y += speed * 0.4 + rng.normal(0, jitter)
        ts += 16.0 + rng.normal(0, 2.0)
        pointer.append({"x": x, "y": y, "ts": ts})
    keys, kt = [], 3000.0
    for i in range(8):
        code = f"K{i}"
        keys.append({"code": code, "phase": "down", "ts": kt})
        kt += dwell * 1000.0
        keys.append({"code": code, "phase": "up", "ts": kt})
        kt += 180.0 + rng.normal(0, 20.0)
    return RepPayload(pointer=pointer, keys=keys)


def test_full_flow_accepts_enrollee_rejects_impostor():
    svc = DemoService()
    rng = np.random.default_rng(7)

    # Enrollee profile.
    for _ in range(8):
        r = svc.enroll(_rep(rng, speed=2.0, dwell=0.09, jitter=0.4))
        assert r["ok"], r

    cal = svc.calibrate()
    assert cal["threshold"] > 0
    assert cal["kept"] >= 1

    # Genuine confirm (same profile) must ACCEPT.
    g = svc.attempt(_rep(rng, speed=2.0, dwell=0.09, jitter=0.4))
    assert g["verdict"] == "ACCEPT", g

    # The attempt result carries the live feature vector (spec §11).
    assert set(g["feature_values"]) == set(svc.state()["features"])
    assert all(isinstance(v, float) for v in g["feature_values"].values())

    # Three distinct impostor profiles must all REJECT.
    rejects = 0
    for sp, dw in [(6.0, 0.20), (0.6, 0.04), (4.0, 0.30)]:
        a = svc.attempt(_rep(rng, speed=sp, dwell=dw, jitter=1.5))
        if a["verdict"] == "REJECT":
            rejects += 1
    assert rejects == 3


def test_bad_rep_is_rejected_with_reason_and_not_enrolled():
    svc = DemoService()
    bad = RepPayload(pointer=[{"x": 1, "y": 1, "ts": 0.0}], keys=[])
    r = svc.enroll(bad)
    assert not r["ok"] and r["reason"]
    assert svc.state()["enroll_reps"] == 0


def test_calibrate_requires_minimum_reps():
    svc = DemoService()
    rng = np.random.default_rng(1)
    for _ in range(2):
        svc.enroll(_rep(rng, 2.0, 0.09, 0.4))
    out = svc.calibrate()
    assert out["ok"] is False and "more" in out["reason"]


def test_reset_clears_state():
    svc = DemoService()
    rng = np.random.default_rng(1)
    svc.enroll(_rep(rng, 2.0, 0.09, 0.4))
    svc.reset()
    assert svc.state()["enroll_reps"] == 0
