"""Pure-numpy PoC models. No torch, deterministic, CPU-instant.

MahalanobisAuth: one-class continuous-auth scorer. Fit on the enrolled
user's feature windows; score = Mahalanobis distance to that profile.
Higher score = more imposter-like.
"""

from __future__ import annotations

import numpy as np

from pad_synth_core.eval.baseline import compute_eer


class MahalanobisAuth:
    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None

    def fit(self, genuine: np.ndarray) -> "MahalanobisAuth":
        x = np.asarray(genuine, dtype=np.float64)
        self._mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * self.reg
        self._inv_cov = np.linalg.pinv(cov)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._inv_cov is not None
        d = np.asarray(x, dtype=np.float64) - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self._inv_cov, d))

    def eer(self, genuine: np.ndarray, imposter: np.ndarray) -> float:
        gs = self.score(genuine)
        is_ = self.score(imposter)
        scores = np.concatenate([gs, is_]).tolist()
        labels = [0] * len(gs) + [1] * len(is_)  # 1 = imposter = positive
        return compute_eer(scores, labels)


class LogisticBotClassifier:
    """Deterministic logistic regression via fixed-iteration gradient
    descent on standardized features. Label 1 = bot."""

    def __init__(self, seed: int = 0, lr: float = 0.1, iters: int = 2000) -> None:
        self.seed = seed
        self.lr = lr
        self.iters = iters
        self._w: np.ndarray | None = None
        self._b = 0.0
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mu) / self._sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBotClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        Xs = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        self._w = rng.normal(0.0, 0.01, size=Xs.shape[1])
        self._b = 0.0
        n = Xs.shape[0]
        for _ in range(self.iters):
            z = Xs @ self._w + self._b
            p = 1.0 / (1.0 + np.exp(-z))
            gw = Xs.T @ (p - y) / n
            gb = float(np.mean(p - y))
            self._w -= self.lr * gw
            self._b -= self.lr * gb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._w is not None
        Xs = self._standardize(np.asarray(X, dtype=np.float64))
        return 1.0 / (1.0 + np.exp(-(Xs @ self._w + self._b)))
