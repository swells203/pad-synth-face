import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def _seed_raw(root: Path, identities: int, samples: int, size: int) -> Path:
    for i in range(identities):
        d = root / f"{i:08d}"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(samples):
            arr = (np.random.default_rng(i * 100 + s).random((size, size, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{s:03d}.png")
    return root


def test_prepare_digiface_resizes_to_64_and_preserves_layout(tmp_path):
    src = _seed_raw(tmp_path / "raw", identities=3, samples=4, size=112)
    dst = tmp_path / "out_64"

    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_digiface_64.py"),
         "--src", str(src), "--dst", str(dst)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr

    ids = sorted(p.name for p in dst.iterdir() if p.is_dir())
    assert ids == ["00000000", "00000001", "00000002"]
    for i in ids:
        samples = sorted((dst / i).glob("*.png"))
        assert len(samples) == 4
        for s in samples:
            with Image.open(s) as im:
                assert im.size == (64, 64), f"{s} is {im.size}, expected (64, 64)"

    import json
    meta = json.loads((dst / "_meta.json").read_text())
    assert meta["target_size"] == 64
    assert meta["identities"] == 3
    assert meta["samples_total"] == 12


def test_prepare_digiface_is_idempotent(tmp_path):
    """Re-running on an existing dst dir should be a no-op (skip already-done)."""
    src = _seed_raw(tmp_path / "raw", identities=2, samples=2, size=112)
    dst = tmp_path / "out_64"

    r1 = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_digiface_64.py"),
         "--src", str(src), "--dst", str(dst)],
        capture_output=True, text=True, check=False,
    )
    assert r1.returncode == 0

    sample_path = dst / "00000000" / "000.png"
    original_mtime = sample_path.stat().st_mtime

    r2 = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_digiface_64.py"),
         "--src", str(src), "--dst", str(dst)],
        capture_output=True, text=True, check=False,
    )
    assert r2.returncode == 0
    assert sample_path.stat().st_mtime == original_mtime, "idempotent: should not re-write existing files"
