import numpy as np

from pad_synth_face.attacks.print import _ICC_PARAMS, _apply_icc


def test_icc_params_table_complete():
    """All three paper types present with the spec'd parameter tuples."""
    assert set(_ICC_PARAMS.keys()) == {"matte", "glossy", "photo"}
    # Tuple shape: (gamut_compression: float, (Δx, Δy): tuple, tone_gamma: float)
    for v in _ICC_PARAMS.values():
        assert len(v) == 3
        assert isinstance(v[0], float)
        assert isinstance(v[1], tuple) and len(v[1]) == 2
        assert isinstance(v[2], float)


def test_icc_strength_zero_is_near_identity():
    """At strength=0, gamut compression is off; output ~= input modulo
    white-point shift and tone gamma (which still apply at full strength)."""
    rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = _apply_icc(rgb, "glossy", strength=0.0)
    # Glossy has tiny white-point shift and gamma 0.95 -> output close to 0.5
    np.testing.assert_allclose(out, rgb, atol=0.1)


def test_icc_matte_warms_relative_to_glossy():
    """Matte's positive Δx (warm shift) should raise R relative to glossy."""
    rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
    matte = _apply_icc(rgb, "matte", strength=1.0)
    glossy = _apply_icc(rgb, "glossy", strength=1.0)
    assert matte[..., 0].mean() > glossy[..., 0].mean()


def test_icc_compresses_extremes():
    """Gamut compression pushes extremes toward 0.5 (more visible at strength=1)."""
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0] = 1.0  # pure red
    matte = _apply_icc(rgb, "matte", strength=1.0)
    assert matte[..., 0].mean() < 1.0  # was 1.0, now pulled down


def test_icc_output_clipped_to_unit():
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 16, 3)).astype(np.float32)
    for paper in ("matte", "glossy", "photo"):
        out = _apply_icc(rgb, paper, strength=1.0)
        assert out.shape == rgb.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
