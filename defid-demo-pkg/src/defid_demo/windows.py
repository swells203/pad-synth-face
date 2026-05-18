"""Windowed feature extraction. Calls the real defid.features.extract_features
and keeps the touch+keystroke subset (indices 0..8). The demo therefore runs
the identical validated extractor formulas."""

from __future__ import annotations

import numpy as np

from defid.features import FEATURE_NAMES, extract_features
from defid.session import BehavioralSession

SUBSET_IDX = list(range(9))
FEATURE_SUBSET = FEATURE_NAMES[:9]


def _session(touch: list[dict], key: list[dict]) -> BehavioralSession:
    return BehavioralSession(
        session_id="demo", label="genuine", subject_id="demo",
        touch=touch, key=key, motion=[],
        ontology_version="demo", generator_version="demo", seed=0,
    )


def extract_windows(
    touch: list[dict], key: list[dict], k: int = 5, overlap: float = 0.5
) -> np.ndarray:
    """k overlapping touch sub-windows; each row = subset features for that
    touch slice paired with the rep's full keystroke stream."""
    n = len(touch)
    if n < k:
        v = extract_features(_session(touch, key))[SUBSET_IDX]
        return np.tile(v, (k, 1))
    step = int(n * (1 - overlap) / (k - 1 + 1e-9))
    step = max(step, 1)
    win_len = n - step * (k - 1)
    rows = []
    for i in range(k):
        lo = i * step
        sub = touch[lo : lo + win_len]
        rows.append(extract_features(_session(sub, key))[SUBSET_IDX])
    return np.asarray(rows, dtype=np.float64)
