import numpy as np

from defid.models import MahalanobisAuth


def test_auth_scores_imposter_higher_than_genuine():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.0, 1.0, size=(80, 6))
    imposter = rng.normal(4.0, 1.0, size=(40, 6))

    m = MahalanobisAuth().fit(genuine)
    g_scores = m.score(genuine)
    i_scores = m.score(imposter)
    assert i_scores.mean() > g_scores.mean()


def test_auth_eer_low_on_separable_data():
    rng = np.random.default_rng(1)
    genuine = rng.normal(0.0, 1.0, size=(100, 6))
    imposter = rng.normal(5.0, 1.0, size=(100, 6))

    m = MahalanobisAuth().fit(genuine)
    eer = m.eer(genuine, imposter)
    assert eer < 0.1


def test_auth_is_deterministic():
    rng = np.random.default_rng(2)
    g = rng.normal(0.0, 1.0, size=(50, 6))
    s1 = MahalanobisAuth().fit(g).score(g)
    s2 = MahalanobisAuth().fit(g).score(g)
    assert np.array_equal(s1, s2)
