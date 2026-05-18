import pytest

from defid_demo.adapter import RepPayload, payload_to_session


def test_pointer_samples_become_seconds_and_xy():
    p = RepPayload(
        pointer=[
            {"x": 10.0, "y": 20.0, "ts": 1000.0},
            {"x": 12.0, "y": 25.0, "ts": 1016.0},
            {"x": 15.0, "y": 30.0, "ts": 1033.0},
        ],
        keys=[],
    )
    touch, key = payload_to_session(p)
    assert touch[0] == {"t": 0.0, "x": 10.0, "y": 20.0}
    assert touch[1]["t"] == pytest.approx(0.016)
    assert touch[2]["t"] == pytest.approx(0.033)
    assert key == []


def test_keys_pair_down_up_by_code_in_completion_order():
    p = RepPayload(
        pointer=[],
        keys=[
            {"code": "KeyA", "phase": "down", "ts": 2000.0},
            {"code": "KeyB", "phase": "down", "ts": 2050.0},
            {"code": "KeyA", "phase": "up", "ts": 2090.0},
            {"code": "KeyB", "phase": "up", "ts": 2160.0},
        ],
    )
    touch, key = payload_to_session(p)
    assert [(k["phase"], round(k["t"], 3)) for k in key] == [
        ("down", 0.0), ("up", 0.09), ("down", 0.05), ("up", 0.16),
    ]
    downs = [k["t"] for k in key if k["phase"] == "down"]
    ups = [k["t"] for k in key if k["phase"] == "up"]
    assert round(ups[0] - downs[0], 3) == 0.09
    assert round(ups[1] - downs[1], 3) == 0.11


def test_unmatched_key_events_are_dropped():
    p = RepPayload(
        pointer=[],
        keys=[
            {"code": "KeyA", "phase": "up", "ts": 10.0},
            {"code": "KeyB", "phase": "down", "ts": 20.0},
        ],
    )
    _, key = payload_to_session(p)
    assert key == []
