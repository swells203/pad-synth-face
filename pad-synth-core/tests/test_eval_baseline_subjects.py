import json
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_core.eval.baseline import TinyPADDataset


def _img(path: Path, seed: int) -> None:
    arr = (np.random.default_rng(seed).random((64, 64, 3)) * 255).astype("uint8")
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
    by_path = dict(zip(paths, zip(ds.subjects, ds.attack_types)))
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


import torch

from pad_synth_core.eval.baseline import subject_disjoint_split


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
