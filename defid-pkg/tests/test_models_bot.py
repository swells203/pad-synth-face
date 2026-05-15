import numpy as np

from defid.models import LogisticBotClassifier


def test_bot_classifier_separates_classes():
    rng = np.random.default_rng(0)
    human = rng.normal(0.0, 1.0, size=(100, 5))
    bot = rng.normal(3.0, 0.3, size=(100, 5))
    X = np.vstack([human, bot])
    y = np.array([0] * 100 + [1] * 100)

    clf = LogisticBotClassifier(seed=0).fit(X, y)
    preds = (clf.predict_proba(X) >= 0.5).astype(int)
    acc = (preds == y).mean()
    assert acc > 0.9


def test_bot_classifier_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    p1 = LogisticBotClassifier(seed=3).fit(X, y).predict_proba(X)
    p2 = LogisticBotClassifier(seed=3).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)
