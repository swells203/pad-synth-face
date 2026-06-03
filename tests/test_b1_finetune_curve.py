"""B1 curve runner tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "b1_finetune_curve",
    Path(__file__).resolve().parents[1] / "scripts" / "b1_finetune_curve.py",
)
b1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(b1)


def _make_pad_tree(root: Path, n_bonafide: int, n_attack: int) -> None:
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


def test_run_curve_writes_per_n_json_and_summary(tmp_path):
    from pad_synth_core.eval.models_zoo import make_tiny_cnn
    synth, real, out = tmp_path / "synth", tmp_path / "real", tmp_path / "out"
    _make_pad_tree(synth, 8, 8)
    _make_pad_tree(real, 12, 12)
    summary = b1.run_curve(
        synth_root=synth, real_root=real, n_list=[0, 4],
        output_dir=out, model_factory=make_tiny_cnn, mode="full",
        test_fraction=0.4, pretrain_epochs=1, finetune_epochs=1,
        finetune_lr=1e-3, batch_size=4, seed=0, device=None)
    assert (out / "runs" / "N0_seed0.json").exists()
    assert (out / "runs" / "N4_seed0.json").exists()
    r0 = json.loads((out / "runs" / "N0_seed0.json").read_text())
    assert "eer_cross_domain" in r0 and r0["n_real"] == 0
    assert (out / "curve_summary.json").exists()
    assert any(row["n_real"] == 4 and not row["skipped"] for row in summary["rows"])


def test_run_curve_skips_not_caps_oversized_n(tmp_path):
    from pad_synth_core.eval.models_zoo import make_tiny_cnn
    synth, real, out = tmp_path / "synth", tmp_path / "real", tmp_path / "out"
    _make_pad_tree(synth, 8, 8)
    _make_pad_tree(real, 10, 10)
    summary = b1.run_curve(
        synth_root=synth, real_root=real, n_list=[0, 100000],
        output_dir=out, model_factory=make_tiny_cnn, mode="full",
        test_fraction=0.4, pretrain_epochs=1, finetune_epochs=1,
        finetune_lr=1e-3, batch_size=4, seed=0, device=None)
    big = [row for row in summary["rows"] if row["n_real"] == 100000][0]
    assert big["skipped"] is True
    assert not (out / "runs" / "N100000_seed0.json").exists()


def test_split_real_guard_rejects_single_class_test(tmp_path):
    # A real set with ONLY bonafide -> any test split is single-class.
    real = tmp_path / "real"
    _make_pad_tree(real, n_bonafide=12, n_attack=0)
    with pytest.raises(ValueError, match="single class|both"):
        b1.split_real(real, test_fraction=0.4, seed=0)


def test_main_returns_zero(tmp_path):
    synth, real, out = tmp_path / "synth", tmp_path / "real", tmp_path / "out"
    _make_pad_tree(synth, 8, 8)
    _make_pad_tree(real, 12, 12)
    rc = b1.main([
        "--synth-root", str(synth), "--real-root", str(real),
        "--n-list", "0,4", "--model", "L1", "--test-fraction", "0.4",
        "--pretrain-epochs", "1", "--finetune-epochs", "1",
        "--output-dir", str(out), "--seed", "0"])
    assert rc == 0
