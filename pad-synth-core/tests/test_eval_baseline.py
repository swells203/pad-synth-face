import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from pad_synth_core.eval.baseline import compute_eer, train_and_eval_tiny_cnn


def test_compute_eer_perfect_separation():
    scores = [0.9, 0.95, 0.99, 0.1, 0.05, 0.01]
    labels = [1, 1, 1, 0, 0, 0]
    eer = compute_eer(scores, labels)
    assert eer < 0.01


def test_compute_eer_random_around_half():
    rng = torch.Generator().manual_seed(0)
    scores = torch.rand(200, generator=rng).tolist()
    labels = ([0] * 100) + ([1] * 100)
    eer = compute_eer(scores, labels)
    assert 0.30 < eer < 0.70


def test_train_eval_smoke(fixture_pad_dataset_root: Path):
    result = train_and_eval_tiny_cnn(
        dataset_root=fixture_pad_dataset_root,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert "eer" in result
    assert "val_accuracy" in result
    assert 0.0 <= result["eer"] <= 1.0
