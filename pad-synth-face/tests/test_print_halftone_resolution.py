import numpy as np
import pytest

from pad_synth_face.attacks.print import _apply_halftone


def test_64x64_preserves_shape_and_dtype():
    """Sanity: the deterministic-screen path runs at 64x64 (the existing
    halftone tests pin the exact byte-level output -- this just sanity-checks
    shape/dtype for the rewrite)."""
    for dpi in (150, 300, 600, 1200):
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        out = _apply_halftone(rgb, print_dpi=dpi)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.float32


def test_cell_px_scales_with_image_dim():
    """At 224x224 the deterministic halftone pattern must contain visibly
    larger cells than at 64x64 (same physical print at higher capture
    resolution). Measure dominant column-period via FFT."""
    rgb_64 = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rgb_224 = np.full((224, 224, 3), 0.5, dtype=np.float32)
    out_64 = _apply_halftone(rgb_64, print_dpi=150)
    out_224 = _apply_halftone(rgb_224, print_dpi=150)

    def _dominant_period(img: np.ndarray) -> float:
        row = img[img.shape[0] // 2, :, 0].astype(np.float64)
        spec = np.abs(np.fft.rfft(row - row.mean()))
        if spec.size <= 1:
            return float("inf")
        peak_k = int(np.argmax(spec[1:])) + 1
        return len(row) / peak_k  # period in pixels

    period_64 = _dominant_period(out_64)
    period_224 = _dominant_period(out_224)
    # Load-bearing invariant: the FRACTIONAL period (period / image_width)
    # is constant across resolutions — i.e. the dominant period scales
    # proportionally with image dim. Under the old absolute-pixel formula,
    # period_224 / period_64 would still be ~3.5 (because FFT bin spacing
    # is 1/N), but the *fractional* period would shrink at 224 because
    # cell_px stays at 8. So we pin the ratio to (224/64) = 3.5 -- which
    # only holds when cell_px itself scales with image dim.
    ratio = period_224 / period_64
    expected = 224.0 / 64.0  # 3.5
    assert abs(ratio - expected) / expected < 0.10, (
        f"expected period_224 / period_64 ≈ {expected:.2f} (cell_px scales "
        f"with image), got ratio={ratio:.3f} "
        f"(period_64={period_64:.2f}, period_224={period_224:.2f})"
    )


def test_apply_halftone_rejects_non_positive_dpi():
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        _apply_halftone(rgb, print_dpi=0)
    with pytest.raises(ValueError, match="positive"):
        _apply_halftone(rgb, print_dpi=-150)
