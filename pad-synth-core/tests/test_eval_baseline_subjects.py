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
