import numpy as np

from pad_synth_face.attacks.print import _apply_halftone


def test_jitter_different_rng_states_produce_different_outputs():
    """The load-bearing invariant: two different rngs -> two different halftone
    outputs. Without this, the watermark survives the v2.1 work."""
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    out_a = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(1))
    out_b = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(2))
    assert not np.array_equal(out_a, out_b)


def test_jitter_same_rng_state_produces_identical_output():
    """Determinism: same rng seed -> byte-identical output (pipeline invariant)."""
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    out_a = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(7))
    out_b = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(7))
    assert np.array_equal(out_a, out_b)


def test_no_rng_preserves_deterministic_v2_path():
    """When rng=None, behavior is byte-identical to v2 (no jitter, deterministic
    screen). This keeps the existing v2 unit tests passing unchanged."""
    rgb = np.full((32, 32, 3), 0.4, dtype=np.float32)
    a = _apply_halftone(rgb, print_dpi=300)
    b = _apply_halftone(rgb, print_dpi=300)
    assert np.array_equal(a, b)


def test_jitter_preserves_shape_dtype_and_range():
    rgb = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    out = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(11))
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_jitter_consumes_rng_state_across_calls():
    """Two consecutive halftone calls using the SAME rng object yield different
    outputs (proves the rng is being advanced by ~16 draws per call)."""
    rgb = np.full((32, 32, 3), 0.5, dtype=np.float32)
    rng = np.random.default_rng(42)
    out1 = _apply_halftone(rgb, print_dpi=300, rng=rng)
    out2 = _apply_halftone(rgb, print_dpi=300, rng=rng)
    assert not np.array_equal(out1, out2)
