from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.mask import MaskAttack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ontology():
    return load_ontology(REPO_ROOT / "ontology" / "face" / "mask.yaml")


def test_mask_attack_returns_same_shape_uint8():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = MaskAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.dtype == np.uint8
    assert out.shape == bonafide.shape


def test_mask_attack_is_deterministic():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    rng1 = sample_rng(123)
    p1 = attack.sample_params(rng1)
    out1 = attack.simulate(bonafide, p1, rng1)

    rng2 = sample_rng(123)
    p2 = attack.sample_params(rng2)
    out2 = attack.simulate(bonafide, p2, rng2)

    assert p1 == p2
    assert np.array_equal(out1, out2)
