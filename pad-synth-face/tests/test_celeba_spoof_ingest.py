"""CelebA-Spoof staging tests. Generated fixtures mimic the on-disk format;
no real or licensed faces enter the repo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_face.celeba_spoof import (
    SPOOF_TYPE_INDEX,
    SPOOF_TYPE_TO_CLASS,
    stage_celeba_spoof,
)


def _lbl(code: int) -> list[int]:
    v = [0] * 43
    v[SPOOF_TYPE_INDEX] = code
    return v


def _build_celeba_fixture(root: Path) -> Path:
    """Mimic CelebA-Spoof: Data/train/<subj>/{live,spoof}/<name> + label JSON."""
    rng = np.random.default_rng(0)
    labels: dict[str, list[int]] = {}

    def _img(relpath: str, code: int):
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)).save(p)
        labels[relpath] = _lbl(code)

    _img("Data/train/subjA/live/0.jpg", 0)     # bonafide
    _img("Data/train/subjA/spoof/1.jpg", 1)    # print (Photo)
    _img("Data/train/subjB/spoof/2.jpg", 7)    # replay (PC)
    _img("Data/train/subjC/spoof/3.jpg", 4)    # mask (Face Mask)
    _img("Data/train/subjD/spoof/4.jpg", 5)    # Upper-Body Mask -> EXCLUDED
    _img("Data/train/subjD/spoof/5.jpg", 10)   # mask (3D Mask)

    meta = root / "metas" / "intra_test"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "train_label.json").write_text(json.dumps(labels))
    return root


def test_mapping_constant_excludes_partial_masks():
    assert SPOOF_TYPE_TO_CLASS[0] == "bonafide"
    assert SPOOF_TYPE_TO_CLASS[1] == "print" and SPOOF_TYPE_TO_CLASS[3] == "print"
    assert SPOOF_TYPE_TO_CLASS[7] == "replay" and SPOOF_TYPE_TO_CLASS[9] == "replay"
    assert SPOOF_TYPE_TO_CLASS[4] == "mask" and SPOOF_TYPE_TO_CLASS[10] == "mask"
    assert 5 not in SPOOF_TYPE_TO_CLASS and 6 not in SPOOF_TYPE_TO_CLASS


def test_stage_builds_class_symlink_tree(tmp_path):
    src = _build_celeba_fixture(tmp_path / "celeba")
    staging = tmp_path / "staging"
    counts = stage_celeba_spoof(src, staging, splits=("train",))
    # classes
    assert (staging / "bonafide" / "subjA" / "live_0.jpg").is_symlink()
    assert (staging / "attack" / "print" / "subjA" / "spoof_1.jpg").is_symlink()
    assert (staging / "attack" / "replay" / "subjB" / "spoof_2.jpg").is_symlink()
    assert (staging / "attack" / "mask" / "subjC" / "spoof_3.jpg").is_symlink()
    assert (staging / "attack" / "mask" / "subjD" / "spoof_5.jpg").is_symlink()
    # code 5 (partial) excluded
    assert not (staging / "attack" / "mask" / "subjD" / "spoof_4.jpg").exists()
    assert counts["bonafide"] == 1 and counts["print"] == 1
    assert counts["replay"] == 1 and counts["mask"] == 2
    assert counts["skipped"] == 1  # the code-5 image


def test_stage_max_subjects_caps(tmp_path):
    src = _build_celeba_fixture(tmp_path / "celeba")
    staging = tmp_path / "staging"
    counts = stage_celeba_spoof(src, staging, splits=("train",), max_subjects=1)
    # exactly one subject staged (which one is seed-shuffle dependent, not lexical)
    assert counts["n_subjects"] == 1
    staged = [p.name for p in (staging / "bonafide").iterdir()] if (staging / "bonafide").exists() else []
    staged += [p.name for cls in (staging / "attack").iterdir() for p in cls.iterdir()] if (staging / "attack").exists() else []
    assert len(set(staged)) == 1  # all staged images belong to a single subject


def test_stage_max_subjects_is_representative_not_lexical(tmp_path):
    """Regression: capping must sample subjects representatively. CelebA-Spoof
    subject IDs correlate with attack type, so lexical sorted()[:N] yields a
    skewed mix. The seeded shuffle must pull BOTH attack families when capping."""
    src = tmp_path / "celeba"
    rng = np.random.default_rng(0)
    labels = {}
    def _img(subj, kind, code, i):
        rel = f"Data/train/{subj}/{kind}/{i}.jpg"
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
        labels[rel] = _lbl(code)
    # subjects 00..09 are PRINT-only (code 1); 10..19 are REPLAY-only (code 7).
    for s in range(10):
        _img(f"{s:02d}", "live", 0, 0); _img(f"{s:02d}", "spoof", 1, 1)        # print
    for s in range(10, 20):
        _img(f"{s:02d}", "live", 0, 0); _img(f"{s:02d}", "spoof", 7, 1)        # replay
    (src / "metas" / "intra_test").mkdir(parents=True, exist_ok=True)
    (src / "metas" / "intra_test" / "train_label.json").write_text(json.dumps(labels))
    # Cap to 10 subjects: lexical-first would be 00..09 = all print, 0 replay.
    counts = stage_celeba_spoof(src, tmp_path / "stg", splits=("train",),
                                max_subjects=10, seed=0)
    assert counts["n_subjects"] == 10
    assert counts["print"] > 0 and counts["replay"] > 0  # representative, not all-print


def test_stage_reads_txt_label_fallback(tmp_path):
    # No JSON label file -> the .txt fallback (relpath <43 space-separated ints>).
    src = tmp_path / "celeba"
    rng = np.random.default_rng(2)
    def _img(relpath):
        p = src / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)).save(p)
    _img("Data/train/subjX/live/0.jpg")
    _img("Data/train/subjX/spoof/1.jpg")
    meta = src / "metas" / "intra_test"
    meta.mkdir(parents=True, exist_ok=True)
    def _line(relpath, code):
        labels = [0] * 43
        labels[SPOOF_TYPE_INDEX] = code
        return relpath + " " + " ".join(str(x) for x in labels)
    (meta / "train_label.txt").write_text(
        _line("Data/train/subjX/live/0.jpg", 0) + "\n"
        + _line("Data/train/subjX/spoof/1.jpg", 1) + "\n")
    counts = stage_celeba_spoof(src, tmp_path / "staging", splits=("train",))
    assert counts["bonafide"] == 1 and counts["print"] == 1


def test_stage_is_idempotent_on_rerun(tmp_path):
    src = _build_celeba_fixture(tmp_path / "celeba")
    staging = tmp_path / "staging"
    c1 = stage_celeba_spoof(src, staging, splits=("train",))
    c2 = stage_celeba_spoof(src, staging, splits=("train",))  # must not raise
    assert c1 == c2
    assert (staging / "bonafide" / "subjA" / "live_0.jpg").is_symlink()


def test_end_to_end_stage_then_ingest_person_ids(tmp_path):
    """stage -> ingest_real_attack(subject_id_fn) -> canonical dataset whose
    manifest carries person ids (not file paths)."""
    from pad_synth_face.real_attack import ingest_real_attack

    src = _build_celeba_fixture(tmp_path / "celeba")
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    stage_celeba_spoof(src, staging, splits=("train",))

    def subject_id_fn(fp: Path) -> str:
        parts = fp.relative_to(staging).parts
        return parts[1] if parts[0] == "bonafide" else parts[2]

    summary = ingest_real_attack(
        src=staging, out=out, dataset_name="CelebA-Spoof",
        license="CelebA-Spoof non-commercial research",
        source_url="https://github.com/ZhangYuanhan-AI/CelebA-Spoof",
        subject_id_fn=subject_id_fn,
    )
    assert summary["counts"]["bonafide"] == 1
    assert summary["counts"]["mask"] == 2
    recs = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    ids = {r["bonafide_source"]["id"] for r in recs}
    # person ids, NOT file paths
    assert ids == {"subjA", "subjB", "subjC", "subjD"}
    assert all("/" not in i for i in ids)


def test_fixture_b1_chain_runs(tmp_path):
    """stage -> ingest -> B1 run_curve on a CelebA-shaped fixture (plumbing)."""
    import importlib.util
    from pad_synth_face.real_attack import ingest_real_attack
    from pad_synth_core.eval.models_zoo import make_tiny_cnn

    # Bigger fixture so the split has both classes on each side.
    src = tmp_path / "celeba"
    rng = np.random.default_rng(1)
    labels = {}
    def _img(relpath, code):
        p = src / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)).save(p)
        labels[relpath] = _lbl(code)
    for s in range(8):
        _img(f"Data/train/subj{s}/live/0.jpg", 0)
        _img(f"Data/train/subj{s}/spoof/1.jpg", 1 if s % 2 else 7)
    (src / "metas" / "intra_test").mkdir(parents=True, exist_ok=True)
    (src / "metas" / "intra_test" / "train_label.json").write_text(json.dumps(labels))

    staging, out = tmp_path / "staging", tmp_path / "out"
    stage_celeba_spoof(src, staging, splits=("train",))

    def subject_id_fn(fp: Path) -> str:
        parts = fp.relative_to(staging).parts
        return parts[1] if parts[0] == "bonafide" else parts[2]
    ingest_real_attack(
        src=staging, out=out, dataset_name="CelebA-Spoof", license="nc",
        source_url="u", subject_id_fn=subject_id_fn)

    # Load the B1 runner and run a tiny curve: use the ingested celeba fixture as
    # both synth (pretrain) and real (finetune/test) root -- a pure plumbing
    # check that stage->ingest->B1 connects; EER values are meaningless.
    spec = importlib.util.spec_from_file_location(
        "b1", Path(__file__).resolve().parents[2] / "scripts" / "b1_finetune_curve.py")
    b1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b1)
    summary = b1.run_curve(
        synth_root=out, real_root=out, n_list=[0, 2],
        output_dir=tmp_path / "b1out", model_factory=make_tiny_cnn, mode="full",
        test_fraction=0.4, pretrain_epochs=1, finetune_epochs=1,
        finetune_lr=1e-3, batch_size=4, seed=0, device=None)
    assert any(r["n_real"] == 2 and not r["skipped"] for r in summary["rows"])
