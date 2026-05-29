from pathlib import Path

import numpy as np

from pad_synth_core import IMAGE_SHAPE, IMAGE_SIZE
from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.mask import MaskAttack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ontology():
    return load_ontology(REPO_ROOT / "ontology" / "face" / "mask.yaml")


def test_mask_attack_returns_same_shape_uint8():
    bonafide = np.full(IMAGE_SHAPE, 128, dtype=np.uint8)
    attack = MaskAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.dtype == np.uint8
    assert out.shape == bonafide.shape


def test_mask_attack_is_deterministic():
    bonafide = np.full(IMAGE_SHAPE, 128, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    rng1 = sample_rng(123)
    p1 = attack.sample_params(rng1)
    out1 = attack.simulate(bonafide, p1, rng1)

    rng2 = sample_rng(123)
    p2 = attack.sample_params(rng2)
    out2 = attack.simulate(bonafide, p2, rng2)

    assert p1 == p2
    assert np.array_equal(out1, out2)


def test_mask_jitter_different_seeds_differ():
    """Load-bearing anti-watermark invariant: two rngs -> two outputs."""
    bonafide = np.full(IMAGE_SHAPE, 150, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    rng1 = sample_rng(1)
    out1 = attack.simulate(bonafide, attack.sample_params(rng1), rng1)
    rng2 = sample_rng(2)
    out2 = attack.simulate(bonafide, attack.sample_params(rng2), rng2)

    assert not np.array_equal(out1, out2)


def test_mask_output_is_not_quantised():
    """Anti-palette guard (the exact v2 halftone mistake): continuous output
    must have far more than the 16-colour palette that produced the watermark."""
    rng = sample_rng(0)
    bonafide = rng.integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
    attack = MaskAttack(_ontology())
    srng = sample_rng(5)
    out = attack.simulate(bonafide, attack.sample_params(srng), srng)
    n_colors = np.unique(out.reshape(-1, 3), axis=0).shape[0]
    # The v2 halftone watermark collapsed to ~16 colours; >1000 distinct
    # colours (of 4096 pixels) confirms continuous, non-quantised output.
    assert n_colors > 1000


def test_mask_materials_are_distinguishable():
    """The three mask_type bundles must produce measurably different images."""
    bonafide = np.full(IMAGE_SHAPE, 140, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    outs = {}
    for mat in ("paper", "silicone", "resin"):
        rng = sample_rng(42)
        params = {**attack.sample_params(rng), "mask_type": mat}
        outs[mat] = attack.simulate(bonafide, params, rng)

    assert not np.array_equal(outs["paper"], outs["silicone"])
    assert not np.array_equal(outs["silicone"], outs["resin"])
    means = {m: float(o.mean()) for m, o in outs.items()}
    # Materials differ by more than rounding noise.
    assert max(means.values()) - min(means.values()) > 1.0


def test_mask_preserves_shape_and_range_on_random_input():
    rng = sample_rng(9)
    bonafide = rng.integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
    attack = MaskAttack(_ontology())
    srng = sample_rng(3)
    out = attack.simulate(bonafide, attack.sample_params(srng), srng)
    assert out.shape == IMAGE_SHAPE
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_aperture_mismatch_darkens_eye_region():
    """Guard that the aperture stage fires (not silently a no-op)."""
    from pad_synth_face.attacks.mask import _aperture_mismatch

    img = np.ones(IMAGE_SHAPE, dtype=np.float32)
    rng = sample_rng(0)
    out = _aperture_mismatch(img, 0.0, rng)
    # Left-eye centre (~0.36*IMAGE_SIZE, 0.30*IMAGE_SIZE) must be darker than a corner.
    assert out[int(0.30 * IMAGE_SIZE), int(0.36 * IMAGE_SIZE)].mean() < out[2, 2].mean()
