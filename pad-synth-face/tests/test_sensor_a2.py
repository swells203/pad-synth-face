"""A2 capture-realism tests: extended preset fields + per-effect helpers."""

import numpy as np

from pad_synth_core.rng import sample_rng
from pad_synth_face.sensor import MOBILE_FRONT_2024, WEBCAM_1080P


def test_mobile_preset_has_a2_fields():
    p = MOBILE_FRONT_2024
    assert p.lens_k1_range == (-0.10, 0.10)
    assert p.motion_blur_px_range == (1, 7)
    assert p.jpeg_passes_range == (1, 3)


def test_webcam_preset_has_a2_fields():
    p = WEBCAM_1080P
    assert p.lens_k1_range == (-0.05, 0.05)
    assert p.motion_blur_px_range == (1, 4)
    assert p.jpeg_passes_range == (1, 2)


def test_preset_a2_ranges_are_valid_intervals():
    for p in (MOBILE_FRONT_2024, WEBCAM_1080P):
        assert p.lens_k1_range[0] <= 0.0 <= p.lens_k1_range[1]
        assert p.motion_blur_px_range[0] >= 1
        assert p.motion_blur_px_range[0] <= p.motion_blur_px_range[1]
        assert p.jpeg_passes_range[0] >= 1
        assert p.jpeg_passes_range[0] <= p.jpeg_passes_range[1]


def test_lens_distort_identity_when_k1_zero():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _lens_distort(img, k1=0.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # k1=0 with linear interpolation is exactly identity at integer sample grid
    assert np.array_equal(out, img)


def test_lens_distort_changes_image_when_k1_nonzero():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(1).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _lens_distort(img, k1=0.10)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Non-zero k1 must visibly change a non-degenerate image
    assert not np.array_equal(out, img)


def test_lens_distort_deterministic():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(2).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out1 = _lens_distort(img, k1=0.08)
    out2 = _lens_distort(img, k1=0.08)
    assert np.array_equal(out1, out2)


def test_lens_distort_barrel_and_pincushion_differ():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(3).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    barrel = _lens_distort(img, k1=-0.10)
    pincushion = _lens_distort(img, k1=0.10)
    assert not np.array_equal(barrel, pincushion)


def test_motion_blur_identity_when_length_one():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _motion_blur(img, length_px=1, angle_rad=0.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_motion_blur_smooths_when_length_large():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(1).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _motion_blur(img, length_px=7, angle_rad=0.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Linear smoothing of a high-frequency noise image must reduce variance
    assert out.var() < img.var() * 0.85


def test_motion_blur_direction_matters():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(2).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    horiz = _motion_blur(img, length_px=7, angle_rad=0.0)
    vert = _motion_blur(img, length_px=7, angle_rad=np.pi / 2.0)
    assert not np.array_equal(horiz, vert)


def test_motion_blur_deterministic():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(3).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out1 = _motion_blur(img, length_px=5, angle_rad=0.6)
    out2 = _motion_blur(img, length_px=5, angle_rad=0.6)
    assert np.array_equal(out1, out2)


def test_noise_scales_with_signal_level():
    """Shot noise must be larger on bright pixels than dark pixels."""
    from pad_synth_face.sensor import _noise

    dark = np.full((128, 128, 3), 10, dtype=np.uint8)
    bright = np.full((128, 128, 3), 200, dtype=np.uint8)
    rng_d = sample_rng(0)
    rng_b = sample_rng(0)  # identical rng so only the signal differs
    noisy_dark = _noise(dark, iso=800, rng=rng_d)
    noisy_bright = _noise(bright, iso=800, rng=rng_b)
    # bright signal -> larger shot sigma -> wider noise std
    assert noisy_bright.astype(np.float32).std() > noisy_dark.astype(np.float32).std()


def test_noise_scales_with_iso():
    """Doubling ISO must measurably increase noise on a fixed signal."""
    from pad_synth_face.sensor import _noise

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    low = _noise(img, iso=100, rng=sample_rng(0))
    high = _noise(img, iso=1600, rng=sample_rng(0))
    assert high.astype(np.float32).std() > low.astype(np.float32).std()


def test_noise_deterministic_given_rng():
    from pad_synth_face.sensor import _noise

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    out1 = _noise(img, iso=400, rng=sample_rng(5))
    out2 = _noise(img, iso=400, rng=sample_rng(5))
    assert np.array_equal(out1, out2)


def test_noise_jitters_with_rng():
    from pad_synth_face.sensor import _noise

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    out1 = _noise(img, iso=400, rng=sample_rng(5))
    out2 = _noise(img, iso=400, rng=sample_rng(6))
    assert not np.array_equal(out1, out2)


def test_jpeg_chain_single_pass_matches_jpeg_roundtrip():
    from pad_synth_face.sensor import _jpeg_chain, _jpeg_roundtrip

    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    chained = _jpeg_chain(img, qf_per_pass=[85])
    single = _jpeg_roundtrip(img, qf=85)
    assert np.array_equal(chained, single)


def test_jpeg_chain_multiple_passes_degrades_more_than_single():
    from pad_synth_face.sensor import _jpeg_chain

    img = (np.random.default_rng(1).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    one_pass = _jpeg_chain(img, qf_per_pass=[75])
    three_pass = _jpeg_chain(img, qf_per_pass=[75, 75, 75])
    # Each re-encode at the same QF accumulates loss; pixel-wise L2 grows.
    delta_one = np.abs(img.astype(np.int16) - one_pass.astype(np.int16)).mean()
    delta_three = np.abs(img.astype(np.int16) - three_pass.astype(np.int16)).mean()
    assert delta_three > delta_one


def test_jpeg_chain_deterministic():
    from pad_synth_face.sensor import _jpeg_chain

    img = (np.random.default_rng(2).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out1 = _jpeg_chain(img, qf_per_pass=[90, 80])
    out2 = _jpeg_chain(img, qf_per_pass=[90, 80])
    assert np.array_equal(out1, out2)


def test_jpeg_chain_preserves_shape_dtype():
    from pad_synth_face.sensor import _jpeg_chain

    img = (np.random.default_rng(3).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _jpeg_chain(img, qf_per_pass=[88, 82, 78])
    assert out.shape == img.shape
    assert out.dtype == np.uint8
