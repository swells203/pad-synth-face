from pathlib import Path

import numpy as np

from pad_synth_core import IMAGE_SHAPE
from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.print import PrintAttack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ontology():
    return load_ontology(REPO_ROOT / "ontology" / "face" / "print.yaml")


def test_print_attack_returns_same_shape_uint8():
    bonafide = np.full(IMAGE_SHAPE, 128, dtype=np.uint8)
    attack = PrintAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.dtype == np.uint8
    assert out.shape == bonafide.shape


def test_print_attack_actually_modifies_the_image():
    bonafide = np.full(IMAGE_SHAPE, 128, dtype=np.uint8)
    attack = PrintAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    # The output should not be byte-identical to the input
    assert not np.array_equal(out, bonafide)


def test_print_attack_is_deterministic_under_same_seed():
    bonafide = np.full(IMAGE_SHAPE, 128, dtype=np.uint8)
    attack = PrintAttack(_ontology())

    rng1 = sample_rng(99)
    params1 = attack.sample_params(rng1)
    out1 = attack.simulate(bonafide, params1, rng1)

    rng2 = sample_rng(99)
    params2 = attack.sample_params(rng2)
    out2 = attack.simulate(bonafide, params2, rng2)

    assert params1 == params2
    assert np.array_equal(out1, out2)
