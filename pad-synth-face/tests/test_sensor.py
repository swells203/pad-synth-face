import numpy as np

from pad_synth_core.rng import sample_rng
from pad_synth_face.sensor import MOBILE_FRONT_2024, apply_sensor


def test_apply_sensor_preserves_shape():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    rng = sample_rng(0)
    out, params = apply_sensor(img, MOBILE_FRONT_2024, rng)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_apply_sensor_is_deterministic():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    out1, p1 = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(5))
    out2, p2 = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(5))
    assert np.array_equal(out1, out2)
    assert p1 == p2


def test_apply_sensor_adds_noise():
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    out, _ = apply_sensor(flat, MOBILE_FRONT_2024, sample_rng(11))
    assert out.std() > 1.0  # Noise must produce nonzero variance


def test_apply_sensor_vignettes_corners_darker_than_center():
    flat = np.full((128, 128, 3), 200, dtype=np.uint8)
    out, _ = apply_sensor(flat, MOBILE_FRONT_2024, sample_rng(0))
    corner = out[0:8, 0:8].mean()
    center = out[60:68, 60:68].mean()
    assert corner < center
