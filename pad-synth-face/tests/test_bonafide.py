from pathlib import Path

import numpy as np

from pad_synth_face.bonafide import DigiFaceLoader


def test_loader_lists_identities(fixture_bonafide_dir: Path):
    loader = DigiFaceLoader(fixture_bonafide_dir)
    ids = loader.list_identities()
    assert len(ids) == 8
    assert ids == sorted(ids)


def test_loader_loads_image_as_uint8_rgb(fixture_bonafide_dir: Path):
    loader = DigiFaceLoader(fixture_bonafide_dir)
    identity = loader.list_identities()[0]
    samples = loader.samples_for_identity(identity)
    assert len(samples) == 2
    arr = loader.load(samples[0])
    assert arr.dtype == np.uint8
    assert arr.shape == (64, 64, 3)


def test_identity_disjoint_split_is_deterministic(fixture_bonafide_dir: Path):
    loader = DigiFaceLoader(fixture_bonafide_dir)
    split_a = loader.identity_disjoint_split(seed=42, ratios=(0.5, 0.25, 0.25))
    split_b = loader.identity_disjoint_split(seed=42, ratios=(0.5, 0.25, 0.25))
    assert split_a == split_b
    all_ids = set(loader.list_identities())
    train, dev, test = split_a
    assert set(train) | set(dev) | set(test) == all_ids
    assert not (set(train) & set(dev))
    assert not (set(train) & set(test))
    assert not (set(dev) & set(test))


def test_identity_disjoint_split_all_to_test(fixture_bonafide_dir: Path):
    """When ratios are (0, 0, 1), every identity must land in the test split."""
    loader = DigiFaceLoader(fixture_bonafide_dir)
    train, dev, test = loader.identity_disjoint_split(
        seed=0, ratios=(0.0, 0.0, 1.0)
    )
    assert train == []
    assert dev == []
    assert set(test) == set(loader.list_identities())
    assert len(test) == 8
