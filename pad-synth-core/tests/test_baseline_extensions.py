from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pad_synth_core import IMAGE_SHAPE
from pad_synth_core.eval.baseline import train_and_cross_domain_eval
from pad_synth_core.eval.models_zoo import make_resnet18_pretrained, make_small_cnn


def _build_tiny_dataset(root: Path, n_bonafide: int = 4, n_attack: int = 4) -> Path:
    base = root / "face"
    for label_dir, n in (("bonafide", n_bonafide), ("print", n_attack)):
        d = base / label_dir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (np.random.default_rng(i).random(IMAGE_SHAPE) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{i:04d}.jpg")
    return root


def test_default_signature_is_backwards_compatible(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(train_root=root, epochs=1, seed=0)
    assert set(out.keys()) >= {
        "eer_in_domain", "val_accuracy_in_domain", "n_train",
        "n_val_in_domain", "eer_cross_domain",
    }
    assert out["eer_cross_domain"] is None


def test_model_factory_injection(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, model_factory=make_small_cnn,
    )
    assert isinstance(out["eer_in_domain"], float)


def test_device_cpu_explicit(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, device="cpu",
    )
    assert isinstance(out["eer_in_domain"], float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA available")
def test_device_cuda(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, device="cuda",
    )
    assert isinstance(out["eer_in_domain"], float)


@pytest.fixture(scope="module")
def _pretrained_available():
    """Skip pretrained-smoke if the weights download fails (e.g. no network
    in CI). Warms the cache once per session."""
    try:
        make_resnet18_pretrained()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pretrained weights unavailable: {e}")


def test_pretrained_resnet18_factory_runs_through_train_and_eval(
    _pretrained_available, tmp_path,
):
    """One-cell smoke: pretrained ResNet18 trained on a tiny fixture dataset,
    1 epoch on CPU, returns finite EER. Locks the path that the Spark sweep
    will exercise at scale."""
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, device="cpu",
        model_factory=make_resnet18_pretrained,
    )
    assert isinstance(out["eer_in_domain"], float)
    assert 0.0 <= out["eer_in_domain"] <= 1.0
