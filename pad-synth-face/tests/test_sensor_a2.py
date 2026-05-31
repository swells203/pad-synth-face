"""A2 capture-realism tests: extended preset fields + per-effect helpers."""

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
