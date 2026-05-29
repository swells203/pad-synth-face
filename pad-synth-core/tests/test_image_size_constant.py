from pad_synth_core import IMAGE_SHAPE, IMAGE_SIZE


def test_image_size_is_224():
    assert IMAGE_SIZE == 224


def test_image_shape_matches_size():
    assert IMAGE_SHAPE == (IMAGE_SIZE, IMAGE_SIZE, 3)
