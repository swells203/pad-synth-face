import torch

from pad_synth_core.eval.models_zoo import (
    FACTORIES,
    make_resnet18,
    make_small_cnn,
    make_tiny_cnn,
)


def _param_count(m):
    return sum(p.numel() for p in m.parameters())


def test_factories_exposed():
    assert set(FACTORIES.keys()) == {"L1", "L2", "L3"}
    assert FACTORIES["L1"] is make_tiny_cnn
    assert FACTORIES["L2"] is make_small_cnn
    assert FACTORIES["L3"] is make_resnet18


def test_tiny_cnn_shape_and_size():
    m = make_tiny_cnn()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    assert _param_count(m) < 1_000  # the floor — truly tiny


def test_small_cnn_shape_and_size():
    m = make_small_cnn()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    # ~97k params per spec; allow some headroom.
    assert 50_000 < _param_count(m) < 200_000


def test_resnet18_shape_and_size():
    m = make_resnet18()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    # ~11M params for torchvision ResNet18 (head replaced).
    assert 10_000_000 < _param_count(m) < 12_000_000
