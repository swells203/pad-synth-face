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
    assert (staging / "bonafide" / "subjA" / "0.jpg").is_symlink()
    assert (staging / "attack" / "print" / "subjA" / "1.jpg").is_symlink()
    assert (staging / "attack" / "replay" / "subjB" / "2.jpg").is_symlink()
    assert (staging / "attack" / "mask" / "subjC" / "3.jpg").is_symlink()
    assert (staging / "attack" / "mask" / "subjD" / "5.jpg").is_symlink()
    # code 5 (partial) excluded
    assert not (staging / "attack" / "mask" / "subjD" / "4.jpg").exists()
    assert counts["bonafide"] == 1 and counts["print"] == 1
    assert counts["replay"] == 1 and counts["mask"] == 2
    assert counts["skipped"] == 1  # the code-5 image


def test_stage_max_subjects_caps(tmp_path):
    src = _build_celeba_fixture(tmp_path / "celeba")
    staging = tmp_path / "staging"
    counts = stage_celeba_spoof(src, staging, splits=("train",), max_subjects=1)
    # only subjA (sorted-first) is staged
    assert counts["n_subjects"] == 1
    assert (staging / "bonafide" / "subjA").is_dir()
    assert not (staging / "attack" / "replay" / "subjB").exists()
