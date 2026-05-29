from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pad_synth_core import IMAGE_SHAPE


@pytest.fixture
def fixture_pad_dataset_root(tmp_path: Path) -> Path:
    """Tiny PAD-shaped dataset for the eval smoke test."""
    root = tmp_path / "ds"
    (root / "face" / "bonafide").mkdir(parents=True)
    (root / "face" / "print").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(8):
        b = rng.integers(120, 200, size=IMAGE_SHAPE, dtype=np.uint8)
        a = rng.integers(20, 100, size=IMAGE_SHAPE, dtype=np.uint8)
        Image.fromarray(b).save(root / "face" / "bonafide" / f"{i}.jpg")
        Image.fromarray(a).save(root / "face" / "print" / f"{i}.jpg")
    return root
