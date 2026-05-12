import numpy as np
import pytest

from pad_synth_core.rng import derive_sample_seed, sample_rng


def test_derive_sample_seed_is_deterministic():
    a = derive_sample_seed(42, "face", "print", 0)
    b = derive_sample_seed(42, "face", "print", 0)
    assert a == b


def test_derive_sample_seed_varies_with_inputs():
    s1 = derive_sample_seed(42, "face", "print", 0)
    s2 = derive_sample_seed(42, "face", "print", 1)
    s3 = derive_sample_seed(42, "face", "replay", 0)
    s4 = derive_sample_seed(43, "face", "print", 0)
    assert len({s1, s2, s3, s4}) == 4


def test_derive_sample_seed_fits_in_uint32():
    for i in range(100):
        s = derive_sample_seed(42, "face", "print", i)
        assert 0 <= s < 2**32


def test_sample_rng_is_seeded_numpy_generator():
    rng = sample_rng(123)
    val = rng.integers(0, 1000)
    rng2 = sample_rng(123)
    val2 = rng2.integers(0, 1000)
    assert val == val2


def test_sample_rng_rejects_unseeded_use():
    with pytest.raises(TypeError):
        sample_rng()  # type: ignore[call-arg]


def test_derive_sample_seed_no_delimiter_collision():
    # Without length-prefix encoding, these two inputs would produce the
    # same payload "0|face|pr|int|0" and collide. With length-prefix
    # encoding they must produce distinct seeds.
    a = derive_sample_seed(0, "face", "pr|int", 0)
    b = derive_sample_seed(0, "face|pr", "int", 0)
    assert a != b
