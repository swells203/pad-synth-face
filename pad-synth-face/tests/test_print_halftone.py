import numpy as np

from pad_synth_core import IMAGE_SHAPE
from pad_synth_face.attacks.print import (
    _apply_halftone,
    _inv_cmyk,
    _to_cmyk,
)


def test_to_cmyk_roundtrip_neutral_gray():
    """Pure gray RGB(128,128,128) -> CMYK and back yields (128,128,128)."""
    rgb = np.full((4, 4, 3), 128, dtype=np.float32) / 255.0
    cmyk = _to_cmyk(rgb)
    back = _inv_cmyk(cmyk)
    assert back.shape == rgb.shape
    np.testing.assert_allclose(back, rgb, atol=1e-3)


def test_to_cmyk_roundtrip_white_and_black():
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    np.testing.assert_allclose(_inv_cmyk(_to_cmyk(rgb)), rgb, atol=1e-3)
    rgb = np.ones((2, 2, 3), dtype=np.float32)
    np.testing.assert_allclose(_inv_cmyk(_to_cmyk(rgb)), rgb, atol=1e-3)


def test_halftone_changes_dot_count_with_dpi():
    """Lower DPI -> larger cells -> fewer transitions/dots in the screen."""
    rgb = np.full(IMAGE_SHAPE, 0.5, dtype=np.float32)
    low = _apply_halftone(rgb, print_dpi=150)
    high = _apply_halftone(rgb, print_dpi=1200)
    # Count horizontal sign-flip transitions in the green channel of each.
    def transitions(img):
        bin_ = (img[:, :, 1] > 0.5).astype(np.int32)
        return int(np.abs(np.diff(bin_, axis=1)).sum())
    n_low = transitions(low)
    n_high = transitions(high)
    assert n_low < n_high, f"expected low-DPI={n_low} < high-DPI={n_high}"


def test_halftone_deterministic():
    """Same input -> byte-identical output (no RNG in halftoning)."""
    rgb = np.full((32, 32, 3), 0.4, dtype=np.float32)
    a = _apply_halftone(rgb, print_dpi=300)
    b = _apply_halftone(rgb, print_dpi=300)
    assert np.array_equal(a, b)


def test_halftone_preserves_shape_and_dtype():
    rgb = np.random.default_rng(0).random(IMAGE_SHAPE).astype(np.float32)
    out = _apply_halftone(rgb, print_dpi=300)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
