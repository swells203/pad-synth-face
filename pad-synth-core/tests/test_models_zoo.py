import torch

from pad_synth_core.eval.models_zoo import (
    FACTORIES,
    make_resnet18,
    make_resnet18_pretrained,
    make_small_cnn,
    make_tiny_cnn,
)


def _param_count(m):
    return sum(p.numel() for p in m.parameters())


def test_factories_exposed():
    assert set(FACTORIES.keys()) == {"L1", "L2", "L3", "L4"}
    assert FACTORIES["L1"] is make_tiny_cnn
    assert FACTORIES["L2"] is make_small_cnn
    assert FACTORIES["L3"] is make_resnet18
    assert FACTORIES["L4"] is make_resnet18_pretrained


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


def test_l4_pretrained_forward_returns_2logits():
    """L4 must accept the canonical 224x224 input and produce (B, 2) logits."""
    m = make_resnet18_pretrained()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_l4_shares_resnet18_param_count():
    """L4 has the same total parameter count as L3 (only the weight INIT differs)."""
    l3 = make_resnet18()
    l4 = make_resnet18_pretrained()
    assert _param_count(l3) == _param_count(l4)
    # Sanity: ~11M params (ResNet18 with 2-class head).
    assert 11_000_000 < _param_count(l4) < 12_000_000
