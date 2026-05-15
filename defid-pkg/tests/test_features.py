from pathlib import Path

import numpy as np

from defid.features import FEATURE_NAMES, extract_features
from defid.generator import generate_session

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_feature_vector_fixed_length_and_finite():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    v = extract_features(s)
    assert v.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(v).all()


def test_features_are_deterministic():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    assert np.array_equal(extract_features(s), extract_features(s))


def test_bot_separable_from_genuine_on_jitter_feature():
    g = generate_session("genuine", "subj-1", seed=2, ontology_dir=ONT)
    b = generate_session("bot", "subj-1", seed=2, ontology_dir=ONT)
    gi = FEATURE_NAMES.index("touch_speed_std")
    assert extract_features(b)[gi] < extract_features(g)[gi]


def test_genuine_vs_imposter_differ():
    g = generate_session("genuine", "subj-1", seed=2, ontology_dir=ONT)
    imp = generate_session("imposter", "subj-1", seed=2, ontology_dir=ONT)
    assert not np.allclose(extract_features(g), extract_features(imp))
