"""Deterministic procedural-bonafide fixture (extracted from conftest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def build_fixture_bonafide(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for identity in range(8):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir(exist_ok=True)
        for sample in range(2):
            base = rng.integers(50, 200, size=3)
            arr = np.tile(base, (64, 64, 1)).astype(np.uint8)
            noise = rng.integers(-20, 20, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root
