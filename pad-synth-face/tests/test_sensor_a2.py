"""A2 capture-realism tests: extended preset fields + per-effect helpers."""

import numpy as np

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
