from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pad_synth_face.bonafide import DigiFaceLoader


def _seed_real_layout(root: Path, identities: int, samples: int, ext: str) -> Path:
    """Build a DigiFace-shaped fake dataset: <root>/<id>/<i>.{png|jpg}."""
    for i in range(identities):
        d = root / f"{i:08d}"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(samples):
            arr = (np.random.default_rng(i * 100 + s).random((16, 16, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{s:03d}.{ext}", format="PNG" if ext == "png" else "JPEG")
    return root


def test_loader_default_lists_all_identities(tmp_path):
    root = _seed_real_layout(tmp_path / "src", identities=5, samples=2, ext="png")
    loader = DigiFaceLoader(root)
    assert loader.list_identities() == [f"{i:08d}" for i in range(5)]


def test_restrict_to_filters_identities(tmp_path):
    root = _seed_real_layout(tmp_path / "src", identities=5, samples=2, ext="png")
    loader = DigiFaceLoader(root, restrict_to=["00000001", "00000003"])
    assert loader.list_identities() == ["00000001", "00000003"]


def test_restrict_to_intersects_with_on_disk(tmp_path):
    """Identities in restrict_to but not on disk are silently dropped."""
    root = _seed_real_layout(tmp_path / "src", identities=3, samples=1, ext="png")
    loader = DigiFaceLoader(root, restrict_to=["00000001", "99999999"])
    assert loader.list_identities() == ["00000001"]


def test_glob_picks_up_jpg_files(tmp_path):
    """Real DigiFace may ship .jpg; loader must find them."""
    root = _seed_real_layout(tmp_path / "src", identities=2, samples=3, ext="jpg")
    loader = DigiFaceLoader(root)
    samples = loader.samples_for_identity("00000000")
    assert len(samples) == 3
    assert all(s.path.suffix == ".jpg" for s in samples)


def test_glob_mixes_png_and_jpg(tmp_path):
    """A directory containing both extensions should yield both."""
    base = tmp_path / "src" / "00000000"
    base.mkdir(parents=True)
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(base / "a.png")
    Image.fromarray(arr).save(base / "b.jpg")
    loader = DigiFaceLoader(tmp_path / "src")
    samples = loader.samples_for_identity("00000000")
    assert len(samples) == 2
    assert {s.path.suffix for s in samples} == {".png", ".jpg"}


def test_no_restrict_argument_preserves_v1_default(tmp_path):
    """Backwards compat: positional-only root still works."""
    root = _seed_real_layout(tmp_path / "src", identities=3, samples=1, ext="png")
    loader = DigiFaceLoader(root)
    assert loader.list_identities() == ["00000000", "00000001", "00000002"]
