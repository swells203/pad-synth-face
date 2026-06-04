import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_core import IMAGE_SHAPE
from pad_synth_face._fixtures import build_fixture_real_attack
from pad_synth_face.real_attack import ingest_real_attack


def _ingest(tmp_path: Path):
    src = build_fixture_real_attack(tmp_path / "src")
    out = tmp_path / "out"
    summary = ingest_real_attack(
        src=src, out=out,
        dataset_name="FIXTURE-RA", license="test-only",
        source_url="https://example.org/fixture",
    )
    return out, summary


def test_canonical_layout_and_counts(tmp_path):
    out, summary = _ingest(tmp_path)
    bona = sorted((out / "face" / "bonafide").glob("*.jpg"))
    pr = sorted((out / "face" / "print").glob("*.jpg"))
    rp = sorted((out / "face" / "replay").glob("*.jpg"))
    assert len(bona) == 6 and len(pr) == 6 and len(rp) == 6
    assert summary["counts"] == {"bonafide": 6, "print": 6, "replay": 6}
    assert sorted(summary["attack_types"]) == ["print", "replay"]
    arr = np.array(Image.open(bona[0]).convert("RGB"))
    assert arr.shape == IMAGE_SHAPE


def test_manifest_labels_and_attack_type(tmp_path):
    out, _ = _ingest(tmp_path)
    recs = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    by_label = {}
    for r in recs:
        by_label.setdefault(r["label"], []).append(r)
    assert len(by_label["bonafide"]) == 6
    assert len(by_label["attack"]) == 12
    assert all(r["attack_type"] is None for r in by_label["bonafide"])
    assert {r["attack_type"] for r in by_label["attack"]} == {"print", "replay"}
    assert all(r["bonafide_source"]["dataset"] == "FIXTURE-RA" for r in recs)
    assert all(r["bonafide_source"]["license"] == "test-only" for r in recs)


def test_provenance_event_written(tmp_path):
    out, _ = _ingest(tmp_path)
    prov = [json.loads(line) for line in (out / "provenance.jsonl").read_text().splitlines()]
    ra = [e for e in prov if e["type"] == "real_attack_dataset_ingested"]
    assert len(ra) == 1
    assert ra[0]["name"] == "FIXTURE-RA"
    assert ra[0]["license"] == "test-only"
    assert sorted(ra[0]["attack_types"]) == ["print", "replay"]


def test_idempotent_and_deterministic(tmp_path):
    src = build_fixture_real_attack(tmp_path / "src")
    out = tmp_path / "out"
    common = dict(src=src, out=out, dataset_name="FIXTURE-RA",
                  license="test-only", source_url="https://example.org/fixture")
    s1 = ingest_real_attack(**common)
    def digest():
        h = hashlib.sha256()
        for p in sorted((out / "face").rglob("*.jpg")):
            h.update(p.read_bytes())
        return h.hexdigest()
    d1 = digest()
    s2 = ingest_real_attack(**common)  # re-run
    assert s2["counts"] == {"bonafide": 0, "print": 0, "replay": 0}
    assert digest() == d1
    assert s1["counts"] == {"bonafide": 6, "print": 6, "replay": 6}
    prov = [json.loads(line) for line in (out / "provenance.jsonl").read_text().splitlines()]
    ra = [e for e in prov if e["type"] == "real_attack_dataset_ingested"]
    assert len(ra) == 1  # second (no-op) run recorded no new event


def test_qc_skips_degenerate_images(tmp_path):
    # A source with one valid (noisy) and one degenerate (flat) bonafide image.
    src = tmp_path / "src"
    (src / "bonafide").mkdir(parents=True)
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    Image.fromarray(noisy).save(src / "bonafide" / "good.png")
    flat = np.full((96, 96, 3), 128, dtype=np.uint8)  # std 0 -> fails QC
    Image.fromarray(flat).save(src / "bonafide" / "bad.png")

    out = tmp_path / "out"
    summary = ingest_real_attack(
        src=src, out=out, dataset_name="QC", license="x",
        source_url="https://example.org/qc",
    )
    assert summary["counts"]["bonafide"] == 1
    assert summary["qc_skipped"] == 1
    assert len(sorted((out / "face" / "bonafide").glob("*.jpg"))) == 1


def test_max_per_class_caps_output(tmp_path):
    src = build_fixture_real_attack(tmp_path / "src")
    out = tmp_path / "out"
    summary = ingest_real_attack(
        src=src, out=out, dataset_name="CAP", license="x",
        source_url="https://example.org/cap", max_per_class=2,
    )
    assert summary["counts"] == {"bonafide": 2, "print": 2, "replay": 2}


def _build_subject_src(root: Path) -> Path:
    """A real-attack src whose images live under <class>/<subject>/<name>."""
    rng = np.random.default_rng(0)
    def _img(p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(p)
    _img(root / "bonafide" / "subjA" / "a0.png")
    _img(root / "bonafide" / "subjB" / "b0.png")
    _img(root / "attack" / "print" / "subjA" / "p0.png")
    _img(root / "attack" / "replay" / "subjB" / "r0.png")
    return root


def _subject_of(staging: Path):
    def fn(fp: Path) -> str:
        parts = fp.relative_to(staging).parts
        return parts[1] if parts[0] == "bonafide" else parts[2]
    return fn


def test_subject_id_fn_sets_person_id(tmp_path):
    src = _build_subject_src(tmp_path / "src")
    out = tmp_path / "out"
    ingest_real_attack(
        src=src, out=out, dataset_name="SUBJ", license="x",
        source_url="https://example.org/subj", subject_id_fn=_subject_of(src),
    )
    recs = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    ids = {r["bonafide_source"]["id"] for r in recs}
    # ids are the SUBJECT names, not file paths
    assert ids == {"subjA", "subjB"}


def test_subject_id_fn_none_preserves_relpath(tmp_path):
    src = _build_subject_src(tmp_path / "src")
    out = tmp_path / "out"
    ingest_real_attack(
        src=src, out=out, dataset_name="SUBJ", license="x",
        source_url="https://example.org/subj",  # no subject_id_fn
    )
    recs = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    ids = {r["bonafide_source"]["id"] for r in recs}
    # back-compat: ids are source-relative file paths
    assert any(i.endswith("a0.png") and "subjA" in i for i in ids)
    assert all("/" in i for i in ids)
