import numpy as np
import pytest

from defid.models import MahalanobisAuth
from defid_demo.demo_auth import DemoAuth


def test_is_a_mahalanobis_auth_subclass():
    assert issubclass(DemoAuth, MahalanobisAuth)


def test_constant_columns_are_dropped_and_recorded():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(30, 5))
    X[:, 2] = 7.0  # constant column
    a = DemoAuth(alpha=0.1).fit(X)
    assert a.kept_idx == [0, 1, 3, 4]
    assert a.dropped_names == []  # names set via fit_named; see below
    d = a.score(rng.normal(0, 1, size=(4, 5)))
    assert d.shape == (4,) and np.all(np.isfinite(d))


def test_fit_named_records_dropped_feature_names():
    X = np.ones((10, 3))
    X[:, 0] = np.arange(10)
    X[:, 1] = np.linspace(0, 1, 10)
    a = DemoAuth(alpha=0.1).fit_named(X, ["fa", "fb", "fc"])
    assert a.dropped_names == ["fc"]


def test_determinism_same_input_same_scores():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(40, 6))
    q = rng.normal(0, 1, size=(8, 6))
    s1 = DemoAuth(alpha=0.15).fit(X).score(q)
    s2 = DemoAuth(alpha=0.15).fit(X).score(q)
    assert np.array_equal(s1, s2)


def test_calibrate_then_classify_separates_genuine_from_outlier():
    rng = np.random.default_rng(3)
    enroll = rng.normal(0.0, 1.0, size=(40, 6))
    holdout = rng.normal(0.0, 1.0, size=(10, 6))
    a = DemoAuth(alpha=0.1).fit(enroll)
    a.calibrate(holdout)
    assert a.threshold is not None and a.threshold > 0

    genuine_attempt = rng.normal(0.0, 1.0, size=(5, 6))
    impostor_attempt = rng.normal(8.0, 1.0, size=(5, 6))
    g = a.classify(genuine_attempt)
    im = a.classify(impostor_attempt)
    assert g["verdict"] == "ACCEPT"
    assert im["verdict"] == "REJECT"
    assert im["frac_above"] >= 0.5
    assert len(g["distances"]) == 5


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        DemoAuth().score(np.zeros((2, 5)))


def test_all_constant_input_raises():
    with pytest.raises(ValueError, match="all feature columns are constant"):
        DemoAuth().fit(np.full((10, 4), 3.0))
