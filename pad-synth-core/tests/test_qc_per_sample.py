import numpy as np

from pad_synth_core.qc.per_sample import (
    QCResult,
    check_image_basic,
)


def test_qc_passes_on_normal_image():
    img = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert isinstance(res, QCResult)
    assert res.ok
    assert res.reason is None


def test_qc_fails_on_wrong_shape():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok
    assert "shape" in res.reason


def test_qc_fails_on_all_black():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok
    assert "histogram" in res.reason


def test_qc_fails_on_all_white():
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok


def test_qc_fails_on_nan_in_float_input():
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[0, 0, 0] = np.nan
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok
    assert "nan" in res.reason.lower()
