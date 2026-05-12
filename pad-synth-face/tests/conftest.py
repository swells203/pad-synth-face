"""Shared test fixtures: a tiny on-disk bonafide dataset for fast tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fixture_bonafide_dir(tmp_path: Path) -> Path:
    """Create 16 procedural 'face-like' RGB images on disk, identity 0..7 × 2 each."""
    root = tmp_path / "digiface_fixture"
    root.mkdir()
    rng = np.random.default_rng(0)
    for identity in range(8):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir()
        for sample in range(2):
            base = rng.integers(50, 200, size=3)
            arr = np.tile(base, (64, 64, 1)).astype(np.uint8)
            noise = rng.integers(-20, 20, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root
