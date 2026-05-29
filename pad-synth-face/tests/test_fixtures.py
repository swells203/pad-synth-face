from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_core import IMAGE_SHAPE
from pad_synth_face._fixtures import (
    build_extended_fixture_bonafide,
    build_fixture_bonafide,
)


def test_extended_fixture_creates_16_identities(tmp_path: Path):
    root = build_extended_fixture_bonafide(tmp_path / "extended")
    identity_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(identity_dirs) == 16


def test_extended_fixture_has_four_samples_per_identity(tmp_path: Path):
    root = build_extended_fixture_bonafide(tmp_path / "extended")
    for identity_dir in root.iterdir():
        if identity_dir.is_dir():
            pngs = list(identity_dir.glob("*.png"))
            assert len(pngs) == 4


def test_extended_fixture_pixel_stats_differ_from_basic(tmp_path: Path):
    """Set A and Set B fixtures must have visibly different pixel distributions
    so the cross-domain eval has actual domain shift to measure."""
    basic_root = build_fixture_bonafide(tmp_path / "basic")
    ext_root = build_extended_fixture_bonafide(tmp_path / "extended")

    def _mean_color(root: Path) -> np.ndarray:
        all_pixels = []
        for png in sorted(root.rglob("*.png")):
            arr = np.array(Image.open(png).convert("RGB"))
            all_pixels.append(arr.reshape(-1, 3))
        stacked = np.concatenate(all_pixels, axis=0)
        return stacked.mean(axis=0)

    basic_mean = _mean_color(basic_root)
    ext_mean = _mean_color(ext_root)
    # Different distributions → mean RGB must differ by at least 10 units in
    # some channel (out of 0-255). This is a coarse but objective check.
    assert np.any(np.abs(basic_mean - ext_mean) > 10)


def test_extended_fixture_is_deterministic(tmp_path: Path):
    """Same call → byte-identical output."""
    import hashlib

    def _hash_tree(root: Path) -> str:
        h = hashlib.sha256()
        for png in sorted(root.rglob("*.png")):
            h.update(png.read_bytes())
        return h.hexdigest()

    a = build_extended_fixture_bonafide(tmp_path / "a")
    b = build_extended_fixture_bonafide(tmp_path / "b")
    assert _hash_tree(a) == _hash_tree(b)


def test_extended_fixture_images_are_64x64_rgb(tmp_path: Path):
    root = build_extended_fixture_bonafide(tmp_path / "extended")
    first = next(root.rglob("*.png"))
    arr = np.array(Image.open(first).convert("RGB"))
    assert arr.shape == IMAGE_SHAPE
    assert arr.dtype == np.uint8
