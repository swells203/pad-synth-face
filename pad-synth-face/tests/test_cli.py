import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_core import IMAGE_SHAPE


def _build_pad_dataset(root: Path, seed: int, n: int = 6) -> Path:
    """Build a minimal PAD-shaped on-disk dataset for eval testing."""
    (root / "face" / "bonafide").mkdir(parents=True)
    (root / "face" / "print").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        b = rng.integers(100, 220, size=IMAGE_SHAPE, dtype=np.uint8)
        a = rng.integers(10, 90, size=IMAGE_SHAPE, dtype=np.uint8)
        Image.fromarray(b).save(root / "face" / "bonafide" / f"{i}.jpg")
        Image.fromarray(a).save(root / "face" / "print" / f"{i}.jpg")
    return root


def test_cli_eval_subcommand_runs_in_domain_only(tmp_path: Path):
    train_root = _build_pad_dataset(tmp_path / "train", seed=0, n=8)

    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "eval",
         "--train-root", str(train_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert "eer_in_domain" in output
    assert output["eer_cross_domain"] is None


def test_cli_eval_subcommand_runs_cross_domain(tmp_path: Path):
    train_root = _build_pad_dataset(tmp_path / "train", seed=0, n=8)
    eval_root = _build_pad_dataset(tmp_path / "eval", seed=99, n=6)

    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "eval",
         "--train-root", str(train_root),
         "--eval-root", str(eval_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["eer_in_domain"] is not None
    assert output["eer_cross_domain"] is not None
    assert 0.0 <= output["eer_cross_domain"] <= 1.0
