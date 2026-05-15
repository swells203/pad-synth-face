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
