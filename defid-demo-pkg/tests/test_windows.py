import numpy as np

from defid.features import FEATURE_NAMES, extract_features
from defid.session import BehavioralSession
from defid_demo.windows import FEATURE_SUBSET, SUBSET_IDX, extract_windows


def _swipe(n):
    return [{"t": i * 0.02, "x": float(i) * 1.3, "y": float(i) * 0.7}
            for i in range(n)]


def _keys(n):
    out = []
    for i in range(n):
        out.append({"t": i * 0.25, "phase": "down"})
        out.append({"t": i * 0.25 + 0.07, "phase": "up"})
    return out


def test_subset_is_first_nine_touch_keystroke_features():
    assert SUBSET_IDX == list(range(9))
    assert FEATURE_SUBSET == FEATURE_NAMES[:9]
    assert "key_paste_ratio" not in FEATURE_SUBSET
    assert "tremor_std" not in FEATURE_SUBSET


def test_window_matches_sliced_extract_features():
    touch, key = _swipe(40), _keys(8)
    W = extract_windows(touch, key, k=5, overlap=0.5)
    assert W.shape == (5, 9)
    assert np.all(np.isfinite(W))

    step = int(len(touch) * (1 - 0.5) / (5 - 1 + 1e-9))
    win_len = len(touch) - step * (5 - 1)
    s = BehavioralSession(
        session_id="d", label="genuine", subject_id="d",
        touch=touch[0:win_len], key=key, motion=[],
        ontology_version="d", generator_version="d", seed=0,
    )
    expected = extract_features(s)[:9]
    assert np.allclose(W[0], expected)


def test_two_distinct_motor_profiles_separate_in_feature_space():
    rng = np.random.default_rng(0)

    def rep(speed, dwell):
        t = 0.0
        touch = []
        for i in range(40):
            t += 0.02
            touch.append({"t": t, "x": i * speed + rng.normal(0, 0.3),
                          "y": i * 0.5 + rng.normal(0, 0.3)})
        key = []
        tk = 0.0
        for _ in range(8):
            key.append({"t": tk, "phase": "down"})
            tk += dwell
            key.append({"t": tk, "phase": "up"})
            tk += 0.15
        return touch, key

    a = np.vstack([extract_windows(*rep(1.0, 0.08)) for _ in range(4)])
    b = np.vstack([extract_windows(*rep(3.0, 0.20)) for _ in range(4)])
    assert np.linalg.norm(a.mean(0) - b.mean(0)) > 1.0
