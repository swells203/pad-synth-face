from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.print import PrintAttack

REPO = Path(__file__).resolve().parents[2]


def _attack() -> PrintAttack:
    return PrintAttack(load_ontology(REPO / "ontology" / "face" / "print.yaml"))


def test_simulate_returns_correct_shape_and_dtype():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = _attack()
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.shape == bonafide.shape
    assert out.dtype == np.uint8


def test_simulate_uses_icc_profile_strength_param():
    """params dict must contain the new icc_profile_strength axis."""
    attack = _attack()
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    assert "icc_profile_strength" in params
    assert 0.5 <= params["icc_profile_strength"] <= 1.0


def test_simulate_low_dpi_has_more_dot_structure_than_high_dpi():
    """Two attacks differing only in print_dpi yield outputs with different
    high-frequency dot structure (low-DPI -> more coarse transitions)."""
    bonafide = np.full((64, 64, 3), 200, dtype=np.uint8)
    attack = _attack()
    # Identical other params; only print_dpi differs.
    base = {
        "paper_type": "matte",
        "tilt_degrees": 0.0,
        "holder_present": False,
        "cutout": "none",
        "icc_profile_strength": 0.75,
    }
    out_low = attack.simulate(bonafide, {**base, "print_dpi": 150}, sample_rng(7))
    out_high = attack.simulate(bonafide, {**base, "print_dpi": 1200}, sample_rng(7))

    def transitions(img: np.ndarray) -> int:
        g = (img[:, :, 1] > 127).astype(np.int32)
        return int(np.abs(np.diff(g, axis=1)).sum())

    assert transitions(out_low) < transitions(out_high)


def test_simulate_deterministic_under_same_seed():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = _attack()

    rng1 = sample_rng(99)
    p1 = attack.sample_params(rng1)
    o1 = attack.simulate(bonafide, p1, rng1)

    rng2 = sample_rng(99)
    p2 = attack.sample_params(rng2)
    o2 = attack.simulate(bonafide, p2, rng2)

    assert p1 == p2
    assert np.array_equal(o1, o2)


def test_simulate_actually_modifies_the_image():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = _attack()
    rng = sample_rng(2)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert not np.array_equal(out, bonafide)
