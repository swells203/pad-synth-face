"""DemoAuth: a MahalanobisAuth subclass adding constant-column drop,
covariance shrinkage, empirical threshold calibration, and per-attempt
verdict aggregation. defid.models is not modified."""

from __future__ import annotations

import numpy as np

from defid.models import MahalanobisAuth


class DemoAuth(MahalanobisAuth):
    def __init__(self, reg: float = 1e-3, alpha: float = 0.10) -> None:
        super().__init__(reg=reg)
        self.alpha = alpha
        self.kept_idx: list[int] = []
        self.dropped_names: list[str] = []
        self.threshold: float | None = None

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x[:, self.kept_idx]

    def fit(self, genuine: np.ndarray) -> "DemoAuth":
        x = np.asarray(genuine, dtype=np.float64)
        var = x.var(axis=0)
        self.kept_idx = [i for i in range(x.shape[1]) if var[i] > 1e-12]
        if not self.kept_idx:
            raise ValueError(
                "DemoAuth.fit: all feature columns are constant — "
                "no usable signal (need more varied enrollment reps)"
            )
        xk = x[:, self.kept_idx]
        self._mean = xk.mean(axis=0)
        cov = np.atleast_2d(np.cov(xk, rowvar=False))
        cov = (1.0 - self.alpha) * cov + self.alpha * np.diag(np.diag(cov))
        cov = cov + np.eye(cov.shape[0]) * self.reg
        self._inv_cov = np.linalg.pinv(cov)
        return self

    def fit_named(self, genuine: np.ndarray, names: list[str]) -> "DemoAuth":
        self.fit(genuine)
        kept = set(self.kept_idx)
        self.dropped_names = [n for i, n in enumerate(names) if i not in kept]
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._inv_cov is None or not self.kept_idx:
            raise RuntimeError("DemoAuth.score called before fit()")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] <= max(self.kept_idx):
            raise ValueError(
                f"DemoAuth.score: expected 2D input with > {max(self.kept_idx)} "
                f"columns, got shape {x.shape}"
            )
        d = self._project(x) - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self._inv_cov, d))

    def calibrate(self, holdout_genuine: np.ndarray) -> float:
        d = self.score(holdout_genuine)
        if d.size >= 4:
            self.threshold = float(d.mean() + 3.0 * d.std())
        else:
            self.threshold = float(d.max() * 1.10)
        return self.threshold

    def classify(self, attempt_windows: np.ndarray) -> dict:
        assert self.threshold is not None, "calibrate() first"
        d = self.score(attempt_windows)
        frac = float(np.mean(d > self.threshold))
        return {
            "verdict": "REJECT" if frac >= 0.5 else "ACCEPT",
            "frac_above": frac,
            "distances": [float(v) for v in d],
            "threshold": self.threshold,
        }
