from pathlib import Path

import numpy as np

from defid.generator import GENERATOR_VERSION, generate_session

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_generate_is_deterministic():
    a = generate_session("genuine", "subj-1", seed=7, ontology_dir=ONT)
    b = generate_session("genuine", "subj-1", seed=7, ontology_dir=ONT)
    assert a.model_dump_json() == b.model_dump_json()


def test_three_labels_supported():
    for label in ("genuine", "imposter", "bot"):
        s = generate_session(label, "subj-1", seed=1, ontology_dir=ONT)
        assert s.label == label
        assert len(s.touch) > 0
        assert len(s.motion) > 0


def test_bot_has_lower_jitter_than_genuine():
    g = generate_session("genuine", "subj-1", seed=3, ontology_dir=ONT)
    b = generate_session("bot", "subj-1", seed=3, ontology_dir=ONT)

    def touch_speed_var(sess):
        xs = np.array([p["x"] for p in sess.touch])
        return float(np.var(np.diff(xs))) if len(xs) > 2 else 0.0

    assert touch_speed_var(b) < touch_speed_var(g)


def test_genuine_subject_profile_is_stable_across_seeds():
    s1 = generate_session("genuine", "subj-9", seed=1, ontology_dir=ONT)
    s2 = generate_session("genuine", "subj-9", seed=2, ontology_dir=ONT)

    def mean_speed(sess):
        pts = np.array([[p["t"], p["x"], p["y"]] for p in sess.touch])
        d = np.linalg.norm(np.diff(pts[:, 1:], axis=0), axis=1)
        dt = np.diff(pts[:, 0])
        return float(np.mean(d / np.maximum(dt, 1e-3)))

    m1, m2 = mean_speed(s1), mean_speed(s2)
    assert abs(m1 - m2) / max(m1, m2) < 0.5


def test_domain_b_shifts_distribution():
    a = generate_session("genuine", "subj-1", seed=5, ontology_dir=ONT, domain="a")
    b = generate_session("genuine", "subj-1", seed=5, ontology_dir=ONT, domain="b")
    sa = np.array([p["x"] for p in a.touch])
    sb = np.array([p["x"] for p in b.touch])
    assert not np.allclose(sa, sb)
