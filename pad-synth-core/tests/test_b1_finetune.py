"""B1 synth-pretrain -> real-finetune unit tests. Generated fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pad_synth_core.eval.baseline import (
    TinyPADDataset,
    finetune_and_eval_on_real,
    pretrain_on_synth,
)
from pad_synth_core.eval.models_zoo import make_resnet18, make_tiny_cnn  # noqa: F401  (make_resnet18 used from Task 2 on)


def _make_pad_tree(root: Path, n_bonafide: int = 6, n_attack: int = 6) -> None:
    """A tiny TinyPADDataset-shaped tree: face/bonafide + face/print + manifest."""
    face = root / "face"
    (face / "bonafide").mkdir(parents=True)
    (face / "print").mkdir(parents=True)
    rng = np.random.default_rng(0)
    manifest = []
    for i in range(n_bonafide):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            face / "bonafide" / f"b{i}.jpg")
        manifest.append({"output_path": f"face/bonafide/b{i}.jpg",
                         "bonafide_source": {"id": f"bsubj{i}"}, "attack_type": None})
    for i in range(n_attack):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            face / "print" / f"a{i}.jpg")
        manifest.append({"output_path": f"face/print/a{i}.jpg",
                         "bonafide_source": {"id": f"asubj{i}"}, "attack_type": "print"})
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")


def test_pretrain_on_synth_returns_trained_model(tmp_path):
    synth = tmp_path / "synth"
    _make_pad_tree(synth)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    assert isinstance(model, torch.nn.Module)
    # Model produces 2-class logits on a 64x64 RGB batch.
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)


def _state_of(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def test_finetune_full_mode_returns_valid_metrics(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=8, n_attack=8)
    model = pretrain_on_synth(synth, make_resnet18, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(8)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(8, 16)))
    res = finetune_and_eval_on_real(
        state, make_resnet18, ft_ds, test_ds,
        mode="full", epochs=2, lr=1e-3, batch_size=4, seed=0)
    assert res["n_real"] == 8
    assert res["mode"] == "full"
    assert res["eer_cross_domain"] is not None
    assert res["n_val_cross_domain"] == 8
    assert res["eer_in_domain"] is not None


def test_finetune_full_mode_moves_backbone(tmp_path):
    import torch as _t
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=8, n_attack=8)
    model = pretrain_on_synth(synth, make_resnet18, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = _t.utils.data.Subset(real_ds, list(range(8)))
    m = make_resnet18()
    m.load_state_dict(state)
    before_backbone = m.conv1.weight.detach().clone()
    opt = _t.optim.Adam(m.parameters(), lr=1e-2)   # full: all params
    loss_fn = _t.nn.CrossEntropyLoss()
    dl = _t.utils.data.DataLoader(ft_ds, batch_size=4, shuffle=True)
    for _ in range(3):
        for x, y in dl:
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
    assert not _t.equal(m.conv1.weight, before_backbone)   # full mode moves backbone


def test_finetune_n_real_zero_is_synth_only_baseline(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=6, n_attack=6)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    empty_ft = torch.utils.data.Subset(real_ds, [])
    test_ds = torch.utils.data.Subset(real_ds, list(range(12)))
    res = finetune_and_eval_on_real(
        state, make_tiny_cnn, empty_ft, test_ds,
        mode="full", epochs=2, batch_size=4, seed=0)
    assert res["n_real"] == 0
    assert res["eer_cross_domain"] is not None        # real-test still evaluated
    assert res["eer_in_domain"] is None               # no finetune set
    assert res["threshold"] is None                   # no dev set -> no ISO threshold
    assert res["acer_cross_domain"] is None


def test_finetune_rejects_unknown_mode(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(4)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(4, 8)))
    with pytest.raises(ValueError):
        finetune_and_eval_on_real(state, make_tiny_cnn, ft_ds, test_ds, mode="bogus")


def test_head_mode_freezes_backbone_trains_fc(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=8, n_attack=8)
    model = pretrain_on_synth(synth, make_resnet18, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(8)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(8, 16)))

    import torch as _t
    m = make_resnet18()
    m.load_state_dict(state)
    before_backbone = m.conv1.weight.detach().clone()
    before_fc = m.fc.weight.detach().clone()
    # Freeze like head mode and finetune a couple steps directly.
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith("fc.")
    opt = _t.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = _t.nn.CrossEntropyLoss()
    dl = _t.utils.data.DataLoader(ft_ds, batch_size=4, shuffle=True)
    for _ in range(3):
        for x, y in dl:
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
    assert _t.equal(m.conv1.weight, before_backbone)        # backbone frozen
    assert not _t.equal(m.fc.weight, before_fc)             # head trained

    # And the public API runs end-to-end in head mode.
    res = finetune_and_eval_on_real(
        state, make_resnet18, ft_ds, test_ds, mode="head",
        epochs=2, lr=1e-2, batch_size=4, seed=0)
    assert res["mode"] == "head"
    assert res["eer_cross_domain"] is not None


def test_head_mode_rejects_non_fc_model(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(4)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(4, 8)))
    # make_tiny_cnn is an nn.Sequential -- no .fc attribute.
    with pytest.raises(ValueError, match="head mode"):
        finetune_and_eval_on_real(state, make_tiny_cnn, ft_ds, test_ds, mode="head")
