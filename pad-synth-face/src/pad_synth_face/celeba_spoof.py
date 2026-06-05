"""Stage a CelebA-Spoof dataset into a real-attack <bonafide|attack/<type>> tree.

CelebA-Spoof (https://github.com/ZhangYuanhan-AI/CelebA-Spoof) stores images
under Data/{train,test}/<subject>/{live,spoof}/<name> with a per-image 43-int
annotation in metas/intra_test/{train,test}_label.json. The spoof-type code is
at index 40. This maps codes to our {bonafide, print, replay, mask} classes
(partial masks 5/6 excluded) and SYMLINKS matching images into a staging tree
that ingest_real_attack consumes. No copying -- the dataset is tens of GB.

Format assumptions (path layout + SPOOF_TYPE_INDEX) are named constants; confirm
them against the real label file on first download (see docs/celeba-spoof-b1.md).
Research-only data; never committed (datasets/ is gitignored).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

# Index of the spoof-type code in CelebA-Spoof's 43-int per-image annotation.
SPOOF_TYPE_INDEX = 40

# CelebA-Spoof spoof-type taxonomy -> our classes. 5 (Upper-Body Mask) and
# 6 (Region Mask) are intentionally absent -> skipped (the synthetic pipeline
# models no partial masks, so including them would bias the transfer measure).
SPOOF_TYPE_TO_CLASS = {
    0: "bonafide",
    1: "print", 2: "print", 3: "print",          # Photo, Poster, A4
    7: "replay", 8: "replay", 9: "replay",        # PC, Pad, Phone
    4: "mask", 10: "mask",                         # Face Mask, 3D Mask
}


def _subject_of(image_relpath: str) -> str:
    """Subject = the path segment after Data/<split>/.  e.g.
    'Data/train/2880/live/x.png' -> '2880'."""
    parts = Path(image_relpath).parts
    if len(parts) < 3:
        raise ValueError(f"unexpected CelebA-Spoof image path shape: {image_relpath!r} "
                         "(expected Data/<split>/<subject>/...)")
    return parts[2]


def _read_labels(src: Path, splits: tuple[str, ...]) -> dict[str, int]:
    """Map image relpath -> spoof-type code, across the requested split label
    files. Handles JSON ({relpath: [ints]}) and whitespace txt (relpath int...)."""
    codes: dict[str, int] = {}
    for sp in splits:
        lf = src / "metas" / "intra_test" / f"{sp}_label.json"
        if lf.exists():
            data = json.loads(lf.read_text())
            if not isinstance(data, dict):
                raise ValueError(f"{lf}: expected a JSON object {{relpath: [labels]}}, got {type(data).__name__}")
            for relpath, labels in data.items():
                if len(labels) <= SPOOF_TYPE_INDEX:
                    raise ValueError(f"{lf}: annotation for {relpath!r} has {len(labels)} entries, need > {SPOOF_TYPE_INDEX}")
                codes[relpath] = int(labels[SPOOF_TYPE_INDEX])
            continue
        txt = src / "metas" / "intra_test" / f"{sp}_label.txt"
        if txt.exists():
            for line in txt.read_text().splitlines():
                toks = line.split()
                if not toks:
                    continue
                codes[toks[0]] = int(toks[1 + SPOOF_TYPE_INDEX])
    return codes


def stage_celeba_spoof(
    src: Path,
    staging: Path,
    max_subjects: int | None = None,
    splits: tuple[str, ...] = ("train", "test"),
    seed: int = 0,
) -> dict[str, Any]:
    """Symlink CelebA-Spoof images into <staging>/bonafide/<subj>/ and
    <staging>/attack/<type>/<subj>/, mapping spoof codes to our classes.

    When max_subjects caps the subset, subjects are selected by a SEEDED
    SHUFFLE, not lexically: CelebA-Spoof subject IDs correlate with
    attack-collection batches, so sorted()[:N] yields a badly skewed
    attack-type mix (e.g. no replay / no codes 5,6). The shuffle makes the
    capped subset representative; full-dataset (max_subjects=None) is unaffected.
    """
    src, staging = Path(src), Path(staging)
    codes = _read_labels(src, splits)

    subjects = sorted({_subject_of(p) for p in codes})
    if max_subjects is not None:
        random.Random(seed).shuffle(subjects)
        subjects = sorted(subjects[:max_subjects])
    keep = set(subjects)

    counts = {"bonafide": 0, "print": 0, "replay": 0, "mask": 0,
              "skipped": 0, "n_subjects": len(keep)}
    for relpath, code in sorted(codes.items()):
        subj = _subject_of(relpath)
        if subj not in keep:
            continue
        cls = SPOOF_TYPE_TO_CLASS.get(code)
        if cls is None:
            counts["skipped"] += 1
            continue
        # Unique per source image: join the path parts after Data/<split>/<subj>/
        # so two images that share a basename (e.g. across nested folders) cannot
        # collide in the staging tree.
        name = "_".join(Path(relpath).parts[3:])
        if cls == "bonafide":
            dst = staging / "bonafide" / subj / name
        else:
            dst = staging / "attack" / cls / subj / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath((src / relpath).resolve(), dst.parent.resolve())
        if not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(target)
        counts[cls] += 1
    return counts
