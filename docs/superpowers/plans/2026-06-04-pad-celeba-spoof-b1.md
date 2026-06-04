# CelebA-Spoof ingest + B1 run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a downloaded CelebA-Spoof tree into the canonical `_real_attack` eval set with person-disjoint subject ids, and wire a B1 run pinned to the 0.40 synth→real baseline.

**Architecture:** A staging adapter maps CelebA-Spoof spoof-type codes → `{bonafide, print, replay, mask}` and symlinks images into a `<staging>/bonafide/<subj>/…` + `<staging>/attack/<type>/<subj>/…` tree; the existing `ingest_real_attack` (extended with one optional `subject_id_fn`) finishes the canonical 224 dataset with person-id manifests; the existing B1 runner runs unchanged.

**Tech Stack:** Python 3.11, Pillow, NumPy, pytest. Reuses `pad_synth_face.real_attack.ingest_real_attack`, `scripts/b1_finetune_curve.py`, the `_real_attack` layout. No new deps.

**Spec:** `docs/superpowers/specs/2026-06-04-pad-celeba-spoof-b1-design.md`

**Branch:** `feat/pad-celeba-spoof-b1` (already created from main; spec committed as `2daafc2`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `pad-synth-face/src/pad_synth_face/real_attack.py` | Ingest; gains optional `subject_id_fn` | **Modify** (1 param + 1 line at the id site) |
| `pad-synth-face/tests/test_real_attack_ingest.py` | Existing ingest tests; add `subject_id_fn` cases | **Modify** (append 2 tests) |
| `pad-synth-face/src/pad_synth_face/celeba_spoof.py` | Staging: spoof-code → class, symlink tree, preserve subject | **Create** |
| `pad-synth-face/tests/test_celeba_spoof_ingest.py` | Staging + end-to-end tests | **Create** |
| `scripts/prepare_celeba_spoof.py` | CLI shim: stage → ingest (with subject_id_fn) | **Create** |
| `docs/celeba-spoof-b1.md` | Runbook: download → prepare → B1 run | **Create** |

No change to the B1 runner, model zoo, or `train_and_cross_domain_eval`.

---

## Task 1: Extend `ingest_real_attack` with optional `subject_id_fn`

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/real_attack.py` (signature + line 101)
- Test: `pad-synth-face/tests/test_real_attack_ingest.py` (append)

**Context:** `ingest_real_attack` currently sets each manifest row's `bonafide_source.id` to the source file's path relative to `src` (real_attack.py:101). Add an optional `subject_id_fn: Callable[[Path], str] | None = None`; when given, the id becomes `subject_id_fn(fp)`. Default `None` preserves today's behaviour for every existing caller. This lets the CelebA-Spoof shim inject real person ids so B1's `subject_disjoint_split` is genuinely person-disjoint.

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-face/tests/test_real_attack_ingest.py`:

```python
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
        parts = fp.resolve().relative_to(staging.resolve()).parts
        # bonafide/<subj>/<name>  OR  attack/<type>/<subj>/<name>
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
```

Add `from pathlib import Path` is already imported; `np`, `Image`, `json`, `ingest_real_attack` already imported at the top of the file.

- [ ] **Step 2: Run, verify failure**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-face/tests/test_real_attack_ingest.py -v -k "subject_id_fn"
```

Expected: FAIL — `TypeError: ingest_real_attack() got an unexpected keyword argument 'subject_id_fn'`.

- [ ] **Step 3: Add the param + Callable import**

In `pad-synth-face/src/pad_synth_face/real_attack.py`, ensure `Callable` is imported (the file imports `from typing import Any` — change to `from typing import Any, Callable`). Add the param to the signature (after `max_per_class`):

```python
    max_per_class: int | None = None,
    subject_id_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
```

Replace the `BonafideSource(... id=...)` line (currently `id=str(fp.relative_to(src))`):

```python
                    bonafide_source=BonafideSource(
                        dataset=dataset_name,
                        id=(subject_id_fn(fp) if subject_id_fn is not None
                            else str(fp.relative_to(src))),
                        license=license,
                    ),
```

- [ ] **Step 4: Run the full real_attack test file, verify pass**

```bash
.venv/bin/pytest pad-synth-face/tests/test_real_attack_ingest.py -v
```

Expected: all pass — the 2 new `subject_id_fn` tests AND the 6 pre-existing tests (back-compat intact).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/real_attack.py pad-synth-face/tests/test_real_attack_ingest.py
git commit -m "feat(pad-real): ingest_real_attack optional subject_id_fn (person-disjoint ids)

When provided, the manifest's bonafide_source.id is the dataset's real subject
id instead of the source file path -- so subject_disjoint_split becomes
genuinely person-disjoint. Default None preserves existing behaviour."
```

---

## Task 2: `stage_celeba_spoof` — spoof-code mapping + symlink staging

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/celeba_spoof.py`
- Test: `pad-synth-face/tests/test_celeba_spoof_ingest.py`

**Context:** CelebA-Spoof stores images under `Data/{train,test}/<subject>/{live,spoof}/<name>` and a per-image 43-int annotation in `metas/intra_test/{train,test}_label.json` (JSON dict `{image_relpath: [43 ints]}`); the spoof-type code sits at index 40 (`SPOOF_TYPE_INDEX`). This staging step reads those codes, maps them to `{bonafide, print, replay, mask}` (codes 5/6 excluded), and **symlinks** matching images into a `<staging>/bonafide/<subj>/…` + `<staging>/attack/<type>/<subj>/…` tree. Symlinks avoid copying tens of GB. The index/path layout are named constants — confirm on first real download (spec §5).

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_celeba_spoof_ingest.py`:

```python
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
```

- [ ] **Step 2: Run, verify failure**

```bash
.venv/bin/pytest pad-synth-face/tests/test_celeba_spoof_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pad_synth_face.celeba_spoof'`.

- [ ] **Step 3: Implement `celeba_spoof.py`**

Create `pad-synth-face/src/pad_synth_face/celeba_spoof.py`:

```python
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
    return parts[2]


def _read_labels(src: Path, splits: tuple[str, ...]) -> dict[str, int]:
    """Map image relpath -> spoof-type code, across the requested split label
    files. Handles JSON ({relpath: [ints]}) and whitespace txt (relpath int...)."""
    codes: dict[str, int] = {}
    for sp in splits:
        lf = src / "metas" / "intra_test" / f"{sp}_label.json"
        if lf.exists():
            for relpath, labels in json.loads(lf.read_text()).items():
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
) -> dict[str, Any]:
    """Symlink CelebA-Spoof images into <staging>/bonafide/<subj>/ and
    <staging>/attack/<type>/<subj>/, mapping spoof codes to our classes."""
    src, staging = Path(src), Path(staging)
    codes = _read_labels(src, splits)

    subjects = sorted({_subject_of(p) for p in codes})
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
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
        name = Path(relpath).name
        if cls == "bonafide":
            dst = staging / "bonafide" / subj / name
        else:
            dst = staging / "attack" / cls / subj / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to((src / relpath).resolve())
        counts[cls] += 1
    return counts
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest pad-synth-face/tests/test_celeba_spoof_ingest.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/celeba_spoof.py pad-synth-face/tests/test_celeba_spoof_ingest.py
git commit -m "feat(pad-celeba): stage_celeba_spoof -- spoof-code mapping + symlink tree

Maps CelebA-Spoof codes to bonafide/print/replay/mask (partial masks 5/6
excluded), symlinks into a <bonafide|attack/<type>>/<subject>/ staging tree,
preserving subject for person-disjoint splits. Format behind named constants."
```

---

## Task 3: `prepare_celeba_spoof.py` shim + end-to-end staging→ingest test

**Files:**
- Create: `scripts/prepare_celeba_spoof.py`
- Test: `pad-synth-face/tests/test_celeba_spoof_ingest.py` (append end-to-end test)

**Context:** The CLI ties staging to `ingest_real_attack` with a subject-aware `subject_id_fn` so the canonical dataset carries person ids. The shim builds the `subject_id_fn` closure over the staging root (mirrors Task 1's `_subject_of`: `bonafide/<subj>/…` → `<subj>`, `attack/<type>/<subj>/…` → `<subj>`).

- [ ] **Step 1: Write the failing end-to-end test**

Append to `pad-synth-face/tests/test_celeba_spoof_ingest.py`:

```python
def test_end_to_end_stage_then_ingest_person_ids(tmp_path):
    """stage -> ingest_real_attack(subject_id_fn) -> canonical dataset whose
    manifest carries person ids (not file paths)."""
    from pad_synth_face.real_attack import ingest_real_attack

    src = _build_celeba_fixture(tmp_path / "celeba")
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    stage_celeba_spoof(src, staging, splits=("train",))

    def subject_id_fn(fp: Path) -> str:
        parts = fp.resolve().relative_to(staging.resolve()).parts
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
```

- [ ] **Step 2: Run, verify it fails for the right reason**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-face/tests/test_celeba_spoof_ingest.py -v -k end_to_end
```

Expected: PASS already if Tasks 1+2 are done (this test uses only their code). If it FAILS, the failure pinpoints a real Task 1/2 integration bug — fix before continuing. (It is included here because it is the first test that exercises stage + the new `subject_id_fn` together.)

- [ ] **Step 3: Create the CLI shim**

Create `scripts/prepare_celeba_spoof.py`:

```python
#!/usr/bin/env python3
"""CLI: stage a CelebA-Spoof tree -> canonical real-attack dataset with
person-disjoint subject ids. See docs/celeba-spoof-b1.md.

CelebA-Spoof is non-commercial research-only -- this answers the B1 buy
decision, it does not produce a shippable model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.celeba_spoof import stage_celeba_spoof  # noqa: E402
from pad_synth_face.real_attack import ingest_real_attack  # noqa: E402

_DEFAULT_LICENSE = (
    "CelebA-Spoof: non-commercial research and educational use only "
    "(see github.com/ZhangYuanhan-AI/CelebA-Spoof)"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="CelebA-Spoof root")
    ap.add_argument("--out", type=Path,
                    default=REPO / "datasets/_real_attack/celeba_spoof")
    ap.add_argument("--staging", type=Path,
                    default=REPO / "datasets/_real_attack/_staging_celeba")
    ap.add_argument("--license", default=_DEFAULT_LICENSE)
    ap.add_argument("--source-url",
                    default="https://github.com/ZhangYuanhan-AI/CelebA-Spoof")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--max-per-class", type=int, default=None)
    args = ap.parse_args()

    staging = args.staging
    stage_summary = stage_celeba_spoof(args.src, staging, max_subjects=args.max_subjects)

    def subject_id_fn(fp: Path) -> str:
        parts = fp.resolve().relative_to(staging.resolve()).parts
        return parts[1] if parts[0] == "bonafide" else parts[2]

    ingest_summary = ingest_real_attack(
        src=staging, out=args.out, dataset_name="CelebA-Spoof",
        license=args.license, source_url=args.source_url,
        max_per_class=args.max_per_class, subject_id_fn=subject_id_fn,
    )
    json.dump({"stage": stage_summary, "ingest": ingest_summary},
              sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the end-to-end test + parse-check the shim**

```bash
.venv/bin/pytest pad-synth-face/tests/test_celeba_spoof_ingest.py -v
.venv/bin/python -c "import ast; ast.parse(open('scripts/prepare_celeba_spoof.py').read()); print('shim parses OK')"
```

Expected: all CelebA-Spoof tests PASS; `shim parses OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_celeba_spoof.py pad-synth-face/tests/test_celeba_spoof_ingest.py
git commit -m "feat(pad-celeba): prepare_celeba_spoof CLI (stage -> ingest w/ person ids)

Thin shim: stage_celeba_spoof then ingest_real_attack with a staging-aware
subject_id_fn -> canonical celeba_spoof dataset with person-disjoint subject
ids. End-to-end test covers stage+ingest producing person-id manifests."
```

---

## Task 4: Runbook `docs/celeba-spoof-b1.md`

**Files:**
- Create: `docs/celeba-spoof-b1.md`

**Context:** Operator flow: download (researcher agreement) → confirm format → prepare → B1 run pinned to the 0.40 baseline. Includes the format-verification first step (spec §5).

- [ ] **Step 1: Write the runbook**

Create `docs/celeba-spoof-b1.md`:

````markdown
# CelebA-Spoof → B1 finetune curve

Answers the pivotal question from the 2026-06-03 reality check (synth→real EER
≈ 0.40 on n=55): **does real finetune data rescue it?** CelebA-Spoof (625k
images, 10,177 subjects) is the free, image-based real set big enough to run B1
meaningfully. Spec: `docs/superpowers/specs/2026-06-04-pad-celeba-spoof-b1-design.md`.

**Licence:** CelebA-Spoof is **non-commercial research only** (with a
"derived data" clause). This answers the buy decision; a model trained on it
**cannot ship** (see memory `pad-commercial-licensing`). Real images are never
committed (`datasets/` is gitignored).

## 1. Obtain

Accept the researcher release agreement and download from
https://github.com/ZhangYuanhan-AI/CelebA-Spoof (Google Drive links). Unzip to a
local path, e.g. `~/data/CelebA_Spoof`.

## 2. Confirm the format (one-time)

The adapter assumes images under `Data/{train,test}/<subject>/{live,spoof}/…`
and a 43-int annotation in `metas/intra_test/{train,test}_label.json` with the
spoof-type code at index 40 (`SPOOF_TYPE_INDEX` in `pad_synth_face/celeba_spoof.py`).
Inspect one label entry and confirm; if the index or layout differs, update the
named constants (one line each). Quick check:

```bash
python3 -c "import json; d=json.load(open('~/data/CelebA_Spoof/metas/intra_test/train_label.json'.replace('~','$HOME'))); k=next(iter(d)); print(k, len(d[k]), 'code@40=', d[k][40])"
```

## 3. Stage + ingest

```bash
.venv/bin/python scripts/prepare_celeba_spoof.py \
  --src ~/data/CelebA_Spoof \
  --out datasets/_real_attack/celeba_spoof \
  --max-subjects 1500          # plenty for N=0..1000 + a disjoint test
```

Writes `datasets/_real_attack/celeba_spoof/` (canonical 224, manifest with
person ids, provenance recording the licence). `--max-subjects` keeps the
symlink/ingest light; raise it for the full run.

## 4. B1 run (pinned to the 0.40 baseline)

Generate the synth pretrain set if needed, then run B1 on the Spark with the
**same synth pretrain + L4** as the reality check so N=0 reproduces ≈0.40:

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d3 \
  --real-root  datasets/_real_attack/celeba_spoof \
  --n-list 0,50,200,1000 --finetune-mode full --model L4 \
  --test-fraction 0.3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_b1_celeba \
  --device cuda
```

## 5. Read the result

- **N=0** should land ≈ 0.40 (matches the reality check; sanity check).
- **EER drops clearly as N grows** → real finetune data rescues the model →
  the commercial-data purchase is justified (buy a *commercially-licensed*
  equivalent to actually ship).
- **Flat near 0.40** → real data alone doesn't fix it; the gap is deeper
  (architecture / domain-generalisation) → don't buy yet.

Append the curve table to
`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.
````

- [ ] **Step 2: Commit**

```bash
git add docs/celeba-spoof-b1.md
git commit -m "docs(pad-celeba): runbook obtain->confirm-format->prepare->B1 run"
```

---

## Task 5: Fixture end-to-end B1 validation + full suite + finish

**Files:**
- None modified unless a failure surfaces.

**Context:** Prove the full chain (stage → ingest → B1 `run_curve`) runs on a generated CelebA-Spoof-shaped fixture, with a genuinely person-disjoint split. This is the inert-scaffolding mechanical proof (no real CelebA-Spoof needed), mirroring the other harnesses' dry-runs.

- [ ] **Step 1: Add a fixture B1 chain test**

Append to `pad-synth-face/tests/test_celeba_spoof_ingest.py`:

```python
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
        parts = fp.resolve().relative_to(staging.resolve()).parts
        return parts[1] if parts[0] == "bonafide" else parts[2]
    ingest_real_attack(
        src=staging, out=out, dataset_name="CelebA-Spoof", license="nc",
        source_url="u", subject_id_fn=subject_id_fn)

    # Load the B1 runner and run a tiny curve: synth=real-shaped fixture as a
    # stand-in pretrain set (plumbing only), real=the ingested celeba fixture.
    spec = importlib.util.spec_from_file_location(
        "b1", Path(__file__).resolve().parents[2] / "scripts" / "b1_finetune_curve.py")
    b1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b1)
    summary = b1.run_curve(
        synth_root=out, real_root=out, n_list=[0, 2],
        output_dir=tmp_path / "b1out", model_factory=make_tiny_cnn, mode="full",
        test_fraction=0.4, pretrain_epochs=1, finetune_epochs=1,
        finetune_lr=1e-3, batch_size=4, seed=0, device=None)
    assert any(r["n_real"] == 2 and not r["skipped"] for r in summary["rows"])
```

(This uses the ingested celeba fixture as both synth and real root — a pure
plumbing check that stage→ingest→B1 connects; EER values are meaningless.)

- [ ] **Step 2: Run the new test + the full repo suite**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-face/tests/test_celeba_spoof_ingest.py -v
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -q
```

Expected: all CelebA-Spoof tests pass; full suite green (1 CUDA skip is fine). In particular the pre-existing `test_real_attack_ingest.py` tests still pass (Task 1 back-compat).

- [ ] **Step 3: Confirm inert without real data**

```bash
ls datasets/_real_attack/celeba_spoof 2>/dev/null && echo "UNEXPECTED: data present" || echo "OK: no CelebA-Spoof staged (pure scaffolding)"
git status --short
```

Expected: no `celeba_spoof` dir; clean working tree (all committed).

- [ ] **Step 4: Commit + review history**

```bash
git add pad-synth-face/tests/test_celeba_spoof_ingest.py
git commit -m "test(pad-celeba): fixture stage->ingest->B1 chain (mechanical plumbing proof)"
git log --oneline feat/pad-celeba-spoof-b1 ^main
```

Expected: ~6 commits (spec + Tasks 1–5). Then hand off to `superpowers:finishing-a-development-branch` to merge to local `main`.

---

## Final Verification

From `/Users/stuartwells/test`:

```bash
.venv/bin/pytest pad-synth-face/tests/test_celeba_spoof_ingest.py pad-synth-face/tests/test_real_attack_ingest.py -v
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -q
```

Expected: CelebA-Spoof + real_attack tests pass; full suite green. `stage_celeba_spoof` + the `subject_id_fn` extension + the shim exist; the fixture chain runs person-disjoint. Inert/ready: `docs/celeba-spoof-b1.md` is the turnkey path once CelebA-Spoof is downloaded, with N=0 reproducing ≈0.40 for direct comparison.
