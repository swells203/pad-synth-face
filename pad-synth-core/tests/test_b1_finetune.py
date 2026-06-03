"""B1 synth-pretrain -> real-finetune unit tests. Generated fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pad_synth_core.eval.baseline import TinyPADDataset, pretrain_on_synth
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
