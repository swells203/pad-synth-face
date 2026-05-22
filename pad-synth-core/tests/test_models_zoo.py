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
    """L1 must match pad_synth_core.eval.baseline.TinyCNN exactly (channels 3->8->16)."""
    m = make_tiny_cnn()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    assert _param_count(m) < 2_000  # the floor — matches baseline.TinyCNN (~1.4k params)
    # Architecture lock: conv1 must be Conv2d(3,8,...) and conv2 Conv2d(8,16,...).
    convs = [m_ for m_ in m.modules() if isinstance(m_, torch.nn.Conv2d)]
    assert len(convs) == 2
    assert (convs[0].in_channels, convs[0].out_channels) == (3, 8)
    assert (convs[1].in_channels, convs[1].out_channels) == (8, 16)


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
