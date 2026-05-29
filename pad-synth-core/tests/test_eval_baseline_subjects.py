import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pad_synth_core import IMAGE_SHAPE
from pad_synth_core.eval.baseline import (
    TinyPADDataset,
    subject_disjoint_split,
    train_and_cross_domain_eval,
)


def _img(path: Path, seed: int) -> None:
    arr = (np.random.default_rng(seed).random(IMAGE_SHAPE) * 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _record(rel: str, label: str, attack_type: str | None, subject: str) -> dict:
    # Minimal SampleRecord-shaped dict (only the fields TinyPADDataset reads).
    return {
        "sample_id": Path(rel).stem,
        "modality": "face",
        "label": label,
        "attack_type": attack_type,
        "bonafide_source": {"dataset": "FX", "id": subject, "license": "x"},
        "pipeline_version": "x",
        "core_version": "x",
        "ontology_version": "x",
        "seed": 0,
        "output_path": rel,
        "output_sha256": "x",
    }


def test_dataset_populates_subjects_and_attack_types_from_manifest(tmp_path):
    root = tmp_path / "ds"
    _img(root / "face" / "bonafide" / "b0.jpg", 0)
    _img(root / "face" / "bonafide" / "b1.jpg", 1)
    _img(root / "face" / "print"    / "p0.jpg", 2)
    _img(root / "face" / "replay"   / "r0.jpg", 3)
    manifest = [
        _record("face/bonafide/b0.jpg", "bonafide", None,     "subject_A"),
        _record("face/bonafide/b1.jpg", "bonafide", None,     "subject_B"),
        _record("face/print/p0.jpg",    "attack",   "print",  "subject_A"),
        _record("face/replay/r0.jpg",   "attack",   "replay", "subject_B"),
    ]
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest) + "\n")

    ds = TinyPADDataset(root)
    assert len(ds) == 4
    # Parallel attribute lists, indexed identically to ds.items.
    paths = [str(p) for p, _ in ds.items]
    by_path = dict(zip(paths, zip(ds.subjects, ds.attack_types, strict=True), strict=True))
    assert by_path[str(root / "face" / "bonafide" / "b0.jpg")] == ("subject_A", None)
    assert by_path[str(root / "face" / "bonafide" / "b1.jpg")] == ("subject_B", None)
    assert by_path[str(root / "face" / "print"    / "p0.jpg")] == ("subject_A", "print")
    assert by_path[str(root / "face" / "replay"   / "r0.jpg")] == ("subject_B", "replay")


def test_dataset_without_manifest_has_all_none(tmp_path):
    root = tmp_path / "ds"
    _img(root / "face" / "bonafide" / "b0.jpg", 0)
    _img(root / "face" / "print" / "p0.jpg", 1)
    ds = TinyPADDataset(root)  # no manifest
    assert len(ds) == 2
    assert ds.subjects == [None, None]
    assert ds.attack_types == [None, None]


def _build_ds(tmp_path, n_subjects=6, samples_per=3):
    root = tmp_path / "ds"
    rng = np.random.default_rng(0)
    recs = []
    for s in range(n_subjects):
        for k in range(samples_per):
            rel = f"face/bonafide/s{s:02d}_{k}.jpg"
            _img(root / rel, rng.integers(0, 1 << 31))
            recs.append(_record(rel, "bonafide", None, f"subject_{s:02d}"))
        rel = f"face/print/s{s:02d}_a.jpg"
        _img(root / rel, rng.integers(0, 1 << 31))
        recs.append(_record(rel, "attack", "print", f"subject_{s:02d}"))
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return TinyPADDataset(root)


def test_subject_disjoint_split_has_no_identity_leak(tmp_path):
    ds = _build_ds(tmp_path)
    train, val = subject_disjoint_split(ds, val_fraction=0.25, seed=0)
    train_subjects = {ds.subjects[i] for i in train.indices}
    val_subjects   = {ds.subjects[i] for i in val.indices}
    assert train_subjects.isdisjoint(val_subjects)
    assert len(train) + len(val) == len(ds)
    assert len(val) >= 1 and len(train) >= 1


def test_subject_disjoint_split_is_deterministic(tmp_path):
    ds = _build_ds(tmp_path)
    a_tr, a_vl = subject_disjoint_split(ds, val_fraction=0.25, seed=42)
    b_tr, b_vl = subject_disjoint_split(ds, val_fraction=0.25, seed=42)
    assert a_tr.indices == b_tr.indices
    assert a_vl.indices == b_vl.indices


def test_subject_disjoint_split_falls_back_to_random_without_manifest(tmp_path):
    root = tmp_path / "ds_no_manifest"
    for i in range(8):
        _img(root / f"face/bonafide/b{i}.jpg", i)
        _img(root / f"face/print/p{i}.jpg", i + 100)
    ds = TinyPADDataset(root)  # subjects all None
    train, val = subject_disjoint_split(ds, val_fraction=0.25, seed=0)
    assert len(train) + len(val) == len(ds)
    assert len(val) >= 1 and len(train) >= 1
    # Both are torch.utils.data.Subset (random_split also returns Subsets in modern torch).
    assert isinstance(train, torch.utils.data.Subset)
    assert isinstance(val, torch.utils.data.Subset)


def test_train_and_eval_returns_iso_metrics(tmp_path):
    # Train set with a manifest -> subject-disjoint split engages.
    train_ds = _build_ds(tmp_path / "train")
    train_root = Path(train_ds.items[0][0]).parents[2]  # face/<x>/<file>.jpg -> root
    # Cross-domain eval set with its own manifest.
    eval_ds = _build_ds(tmp_path / "eval")
    eval_root = Path(eval_ds.items[0][0]).parents[2]

    out = train_and_cross_domain_eval(
        train_root=train_root, eval_root=eval_root,
        epochs=1, batch_size=4, seed=0, device="cpu",
        target_apcer=0.05,
    )
    # New additive keys.
    for k in ("threshold", "target_apcer",
              "apcer_cross_domain", "bpcer_cross_domain", "acer_cross_domain",
              "apcer_per_pai_cross_domain"):
        assert k in out, f"missing new key {k!r}"
    assert out["target_apcer"] == 0.05
    assert 0.0 <= out["apcer_cross_domain"] <= 1.0
    assert 0.0 <= out["bpcer_cross_domain"] <= 1.0
    assert 0.0 <= out["acer_cross_domain"] <= 1.0
    assert isinstance(out["apcer_per_pai_cross_domain"], dict)
    assert out["threshold"] is not None and 0.0 <= out["threshold"] <= 2.0
    # Old keys still present and finite.
    assert 0.0 <= out["eer_in_domain"] <= 1.0
    assert 0.0 <= out["eer_cross_domain"] <= 1.0


def test_iso_metrics_are_none_when_dev_has_no_pai_metadata(tmp_path):
    """Manifest-less train_root -> dev_atypes all None -> ISO metrics return
    None (not a misleading sentinel). Threshold-free EER stays meaningful."""
    # Manifest-less synthetic train set.
    train_root = tmp_path / "train_no_manifest"
    for i in range(8):
        _img(train_root / f"face/bonafide/b{i}.jpg", i)
        _img(train_root / f"face/print/p{i}.jpg", i + 100)
    # Eval set WITH manifest (so per-PAI APCER would otherwise be computable).
    eval_ds = _build_ds(tmp_path / "eval")
    eval_root = Path(eval_ds.items[0][0]).parents[2]

    out = train_and_cross_domain_eval(
        train_root=train_root, eval_root=eval_root,
        epochs=1, batch_size=4, seed=0, device="cpu",
    )
    # ISO metrics not computable without dev PAI metadata.
    assert out["threshold"] is None
    assert out["apcer_cross_domain"] is None
    assert out["bpcer_cross_domain"] is None
    assert out["acer_cross_domain"] is None
    assert out["apcer_per_pai_cross_domain"] is None
    # But threshold-free EER still works.
    assert 0.0 <= out["eer_in_domain"] <= 1.0
    assert 0.0 <= out["eer_cross_domain"] <= 1.0
