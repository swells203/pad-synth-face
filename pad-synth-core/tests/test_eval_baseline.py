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


def test_train_and_cross_domain_eval_in_domain_mode(fixture_pad_dataset_root):
    """When eval_root is None, behavior matches train_and_eval_tiny_cnn."""
    from pad_synth_core.eval.baseline import train_and_cross_domain_eval

    result = train_and_cross_domain_eval(
        train_root=fixture_pad_dataset_root,
        eval_root=None,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert "eer_in_domain" in result
    assert "val_accuracy_in_domain" in result
    assert "n_train" in result
    assert "n_val_in_domain" in result
    # Cross-domain fields should be present but None.
    assert result["eer_cross_domain"] is None
    assert result["n_val_cross_domain"] is None


def test_train_and_cross_domain_eval_with_separate_eval_root(
    fixture_pad_dataset_root, tmp_path
):
    """When eval_root is provided, the result includes cross-domain numbers."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from pad_synth_core.eval.baseline import train_and_cross_domain_eval

    # Build a second tiny PAD-shaped dataset as the cross-domain eval set.
    eval_root = tmp_path / "ds_b"
    (eval_root / "face" / "bonafide").mkdir(parents=True)
    (eval_root / "face" / "print").mkdir(parents=True)
    rng = np.random.default_rng(99)
    for i in range(6):
        b = rng.integers(100, 220, size=(64, 64, 3), dtype=np.uint8)
        a = rng.integers(10, 90, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(b).save(eval_root / "face" / "bonafide" / f"{i}.jpg")
        Image.fromarray(a).save(eval_root / "face" / "print" / f"{i}.jpg")

    result = train_and_cross_domain_eval(
        train_root=fixture_pad_dataset_root,
        eval_root=eval_root,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert result["eer_cross_domain"] is not None
    assert 0.0 <= result["eer_cross_domain"] <= 1.0
    assert result["n_val_cross_domain"] == 12  # 6 bonafide + 6 print
    assert isinstance(result["val_accuracy_cross_domain"], float)


def test_train_and_eval_tiny_cnn_still_works_after_refactor(
    fixture_pad_dataset_root,
):
    """The original entry point must remain functional and return the
    documented field names ('eer', 'val_accuracy', etc.)."""
    from pad_synth_core.eval.baseline import train_and_eval_tiny_cnn

    result = train_and_eval_tiny_cnn(
        dataset_root=fixture_pad_dataset_root,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert "eer" in result
    assert "val_accuracy" in result
    assert "n_train" in result
    assert "n_val" in result
