# PAD Eval-Metrics Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ISO 30107-3 metrics (APCER/BPCER/ACER + per-PAI APCER) at a dev-fixed threshold (target APCER = 5%) and a subject-disjoint in-domain split — additive, so existing EER reporting and sweeps stay comparable.

**Architecture:** A new pure-functions module `eval/metrics.py` (compute_eer moved here; new `apcer_bpcer_acer`, `threshold_at_apcer`). `TinyPADDataset` learns to read the run's `manifest.jsonl` and expose `self.subjects` / `self.attack_types` parallel to `self.items`; a `subject_disjoint_split` helper groups by identity (random fallback when no manifest). `train_and_cross_domain_eval` swaps `random_split` for that helper, fixes the threshold on dev, applies it to cross-domain, returns additive metric keys. `scripts/spark_sweep.py` appends new CSV columns / JSON fields. No external dependencies.

**Tech Stack:** Python 3.12+, NumPy, PyTorch, Pydantic (existing), pytest. Reuses `pad_synth_core.manifest.SampleRecord`.

**Spec:** [`../specs/2026-05-28-pad-eval-metrics-upgrade-design.md`](../specs/2026-05-28-pad-eval-metrics-upgrade-design.md)

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `pad-synth-core/src/pad_synth_core/eval/metrics.py` | pure ISO metrics + EER | Create |
| `pad-synth-core/src/pad_synth_core/eval/baseline.py` | dataset manifest plumbing; subject-disjoint split; integrate metrics; re-export `compute_eer` | Modify |
| `scripts/spark_sweep.py` | additive CSV/JSON columns | Modify |
| `pad-synth-core/tests/test_eval_metrics.py` | hand-computed metric cases | Create |
| `pad-synth-core/tests/test_eval_baseline_subjects.py` | manifest plumbing + subject-disjoint split | Create |
| `pad-synth-core/tests/test_eval_baseline.py` | (already exists) extend with the new integration assertions | Modify |

---

## Task 1: `metrics.py` module (pure functions + tests)

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/eval/metrics.py`
- Create (test): `pad-synth-core/tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-core/tests/test_eval_metrics.py`:

```python
import math

from pad_synth_core.eval.metrics import (
    apcer_bpcer_acer,
    compute_eer,
    threshold_at_apcer,
)


def test_compute_eer_matches_known_case():
    # Bonafide low scores, attacks high scores: EER ~ 0 at threshold 0.5.
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert compute_eer(scores, labels) == 0.0


def test_apcer_bpcer_acer_hand_computed():
    # 4 bonafide (label 0), 6 attacks: 3 of type 'print', 3 of type 'replay'.
    # At threshold 0.5: bona scores {0.2,0.3,0.6,0.7} -> 2 above (BPCER 2/4=0.5);
    # print scores {0.4,0.4,0.9} -> 2 below (APCER_print 2/3); replay {0.6,0.7,0.8}
    # -> 0 below (APCER_replay 0/3). APCER_max = 2/3, ACER = (2/3 + 1/2) / 2.
    scores       = [0.2, 0.3, 0.6, 0.7, 0.4, 0.4, 0.9, 0.6, 0.7, 0.8]
    labels       = [0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ]
    attack_types = [None,None,None,None,"print","print","print","replay","replay","replay"]
    per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(scores, labels, attack_types, 0.5)
    assert math.isclose(per_pai["print"], 2/3)
    assert math.isclose(per_pai["replay"], 0.0)
    assert math.isclose(apcer_max, 2/3)
    assert math.isclose(bpcer, 0.5)
    assert math.isclose(acer, (2/3 + 0.5) / 2.0)


def test_apcer_ignores_bonafide_rows_and_handles_missing_types():
    # All-bonafide eval: APCER must be 0, ACER = BPCER / 2.
    per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(
        scores=[0.1, 0.9], labels=[0, 0], attack_types=[None, None], threshold=0.5,
    )
    assert per_pai == {}
    assert apcer_max == 0.0
    assert bpcer == 0.5
    assert acer == 0.25


def test_threshold_at_apcer_respects_budget_and_prefers_higher_threshold():
    # Two PAI species; choose threshold so overall APCER <= 0.5, with the
    # higher-threshold tie-break (lower BPCER under the same budget).
    scores       = [0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.7]
    labels       = [0,   0,   0,   0,   1,   1,   1  ]
    attack_types = [None,None,None,None,"print","print","replay"]
    thr, achieved = threshold_at_apcer(scores, labels, attack_types, target_apcer=0.5)
    # Verify the constraint actually holds at the returned threshold.
    per_pai, apcer_max, _, _ = apcer_bpcer_acer(scores, labels, attack_types, thr)
    assert apcer_max <= 0.5 + 1e-9
    assert achieved == apcer_max
    # And no strictly higher candidate threshold (from the score set) would also satisfy it.
    for t in sorted(set(scores)):
        if t > thr:
            _, ap, _, _ = apcer_bpcer_acer(scores, labels, attack_types, t)
            assert ap > 0.5 + 1e-9, f"higher thr {t} also satisfied -- tie-break broken"


def test_threshold_at_apcer_trivially_low_with_attacks_only_above_budget():
    # All attacks have very high scores -> APCER stays 0 across most thresholds;
    # function must still return a finite threshold and achieved <= target.
    scores = [0.1, 0.2, 0.95, 0.96, 0.97]
    labels = [0,   0,   1,    1,    1   ]
    types  = [None,None,"print","print","print"]
    thr, achieved = threshold_at_apcer(scores, labels, types, target_apcer=0.05)
    assert achieved <= 0.05 + 1e-9
    assert math.isfinite(thr)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: pad_synth_core.eval.metrics`.

- [ ] **Step 3: Create the metrics module**

Create `pad-synth-core/src/pad_synth_core/eval/metrics.py`:

```python
"""ISO 30107-3 PAD metrics + EER.

All functions are pure: `scores` is a list/array of attack-class probabilities
(P(attack)); `labels` is 0 (bona fide) or 1 (attack); `attack_types` is the
per-sample PAI species string for attack rows and None for bona fide rows.
The decision rule is `score >= threshold => classified attack`.

APCER per PAI species s = fraction of attacks of type s with score < threshold
(i.e. missed). APCER (overall) = max over PAI species (ISO worst-case).
BPCER = fraction of bona fide with score >= threshold. ACER = (APCER + BPCER)/2.

`threshold_at_apcer` scans candidate thresholds (the unique sample scores plus
sentinels just below the min and just above the max) and returns the highest
threshold whose overall APCER stays at or below `target_apcer`. APCER is
monotonically non-decreasing in the threshold, so 'highest threshold under
the budget' coincides with 'lowest BPCER under the budget' -- the best
operating point that respects the budget.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_eer(scores: list[float], labels: list[int]) -> float:
    """Threshold-free Equal Error Rate. Numerically identical to the prior
    implementation in `baseline.py` (kept here as the canonical home)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    thresholds = np.unique(s)
    best = 1.0
    eer = 0.5
    for t in thresholds:
        pred = (s >= t).astype(np.int64)
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        fpr = fp / n_neg
        fnr = fn / n_pos
        diff = abs(fpr - fnr)
        if diff < best:
            best = diff
            eer = (fpr + fnr) / 2.0
    return float(eer)


def apcer_bpcer_acer(
    scores: Iterable[float],
    labels: Iterable[int],
    attack_types: Iterable[str | None],
    threshold: float,
) -> tuple[dict[str, float], float, float, float]:
    """Return (apcer_per_pai, apcer_max, bpcer, acer) at the given threshold.

    Bona fide rows (label 0) are ignored for APCER. Attack rows (label 1) with
    attack_type=None are silently skipped (defensive -- caller should always
    set attack_type on attack rows).
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.int64)
    types = list(attack_types)

    # Per-PAI APCER.
    pai_species = sorted({t for t, lab in zip(types, y) if lab == 1 and t is not None})
    apcer_per_pai: dict[str, float] = {}
    for pai in pai_species:
        mask = np.array([lab == 1 and t == pai for t, lab in zip(types, y)])
        n = int(mask.sum())
        if n == 0:
            continue
        missed = int((s[mask] < threshold).sum())
        apcer_per_pai[pai] = missed / n
    apcer_max = max(apcer_per_pai.values()) if apcer_per_pai else 0.0

    # BPCER.
    bona_mask = (y == 0)
    n_bona = int(bona_mask.sum())
    bpcer = float((s[bona_mask] >= threshold).sum()) / n_bona if n_bona else 0.0

    acer = (apcer_max + bpcer) / 2.0
    return apcer_per_pai, float(apcer_max), float(bpcer), float(acer)


def threshold_at_apcer(
    scores: Iterable[float],
    labels: Iterable[int],
    attack_types: Iterable[str | None],
    target_apcer: float = 0.05,
) -> tuple[float, float]:
    """Return (threshold, achieved_apcer) -- the highest threshold whose overall
    APCER does not exceed `target_apcer`. APCER is monotone non-decreasing in
    threshold, so this is also the threshold minimising BPCER under the budget.
    """
    s_arr = np.asarray(list(scores), dtype=np.float64)
    if s_arr.size == 0:
        return 0.0, 0.0
    # Candidate thresholds: every unique score plus sentinels just below min
    # and just above max so we can fully traverse the operating range.
    cands = sorted(set(s_arr.tolist()))
    cands = [float(s_arr.min()) - 1.0] + cands + [float(s_arr.max()) + 1.0]
    types = list(attack_types)
    labels_list = list(labels)
    best_thr = cands[0]
    best_apcer = 0.0
    for t in cands:
        _, apcer_max, _, _ = apcer_bpcer_acer(s_arr.tolist(), labels_list, types, t)
        if apcer_max <= target_apcer and t >= best_thr:
            best_thr = float(t)
            best_apcer = float(apcer_max)
    return best_thr, best_apcer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_metrics.py -v`
Expected: PASS (all five tests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/metrics.py pad-synth-core/tests/test_eval_metrics.py
git commit -m "feat(pad-core/eval): ISO 30107-3 metrics module (APCER/BPCER/ACER + threshold_at_apcer)"
```

---

## Task 2: Re-export `compute_eer` from `baseline.py` (back-compat)

`compute_eer` lives in `metrics.py` now; `baseline.py` should re-export it so the existing import path `from pad_synth_core.eval.baseline import compute_eer` still works.

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py`

- [ ] **Step 1: Replace the in-file `compute_eer` with a re-export**

In `pad-synth-core/src/pad_synth_core/eval/baseline.py`, **delete the entire `compute_eer` function** (the one currently around lines 62–82 — `def compute_eer(scores, labels): ...`) and replace it with this single import line near the other imports at the top of the file:

```python
from pad_synth_core.eval.metrics import compute_eer  # re-exported for backward compatibility
```

Internal callers in `baseline.py` (`_eval_loader`) keep working unchanged — they import-by-name within the module.

- [ ] **Step 2: Verify existing eval tests still pass**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline.py pad-synth-core/tests/test_baseline_extensions.py -v`
Expected: all pass (the import path still resolves; the function is byte-equivalent).

- [ ] **Step 3: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py
git commit -m "refactor(pad-core/eval): re-export compute_eer from metrics (kept import path)"
```

---

## Task 3: Manifest plumbing in `TinyPADDataset`

Surface `self.subjects` and `self.attack_types` parallel to `self.items`, populated from `<root>/manifest.jsonl` when present. The `__getitem__` interface (returning `(tensor, label)`) is unchanged.

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py`
- Create (test): `pad-synth-core/tests/test_eval_baseline_subjects.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-core/tests/test_eval_baseline_subjects.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline_subjects.py -v`
Expected: FAIL — `AttributeError: 'TinyPADDataset' object has no attribute 'subjects'`.

- [ ] **Step 3: Extend `TinyPADDataset`**

In `pad-synth-core/src/pad_synth_core/eval/baseline.py`, add `import json` near the other imports if not already present, then replace the entire `TinyPADDataset.__init__` body with the version below (keeping `__len__` and `__getitem__` as they are today):

```python
    def __init__(self, root: Path) -> None:
        self.items: list[tuple[Path, int]] = []
        self.subjects: list[str | None] = []
        self.attack_types: list[str | None] = []
        face_root = Path(root) / "face"

        # Manifest provides per-sample subject + attack_type when present;
        # absent or unparseable -> graceful (subjects/attack_types stay None,
        # callers fall back to random splits).
        by_output_path: dict[str, tuple[str | None, str | None]] = {}
        manifest_path = Path(root) / "manifest.jsonl"
        if manifest_path.exists():
            for line in manifest_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    subj = (rec.get("bonafide_source") or {}).get("id")
                    by_output_path[rec["output_path"]] = (subj, rec.get("attack_type"))
                except (json.JSONDecodeError, KeyError):
                    continue

        def _add(path: Path, label: int) -> None:
            self.items.append((path, label))
            rel = str(path.relative_to(Path(root)))
            subj, atype = by_output_path.get(rel, (None, None))
            self.subjects.append(subj)
            self.attack_types.append(atype)

        # Bonafide samples live under face/bonafide/.
        for p in sorted((face_root / "bonafide").glob("*.jpg")):
            _add(p, 0)
        # All other face/<x>/ subdirectories are attack types (print, replay, ...).
        for subdir in sorted(p for p in face_root.iterdir() if p.is_dir()):
            if subdir.name == "bonafide":
                continue
            for p in sorted(subdir.glob("*.jpg")):
                _add(p, 1)
```

- [ ] **Step 4: Run tests + the existing eval suite**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline_subjects.py pad-synth-core/tests/test_eval_baseline.py pad-synth-core/tests/test_baseline_extensions.py -v`
Expected: all pass — the new tests + the existing eval tests (which don't write manifests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_eval_baseline_subjects.py
git commit -m "feat(pad-core/eval): TinyPADDataset surfaces subjects + attack_types from manifest"
```

---

## Task 4: `subject_disjoint_split` helper

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py`
- Modify (test): `pad-synth-core/tests/test_eval_baseline_subjects.py`

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-core/tests/test_eval_baseline_subjects.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline_subjects.py -v`
Expected: FAIL — `ImportError: cannot import name 'subject_disjoint_split'`.

- [ ] **Step 3: Add `subject_disjoint_split` to `baseline.py`**

In `pad-synth-core/src/pad_synth_core/eval/baseline.py`, add this function just after the `TinyPADDataset` class:

```python
def subject_disjoint_split(
    dataset: "TinyPADDataset",
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split a TinyPADDataset into (train, val) with disjoint subjects.

    Groups samples by `dataset.subjects` and assigns whole identities to the
    val side until the running val count reaches roughly `val_fraction` of
    the dataset. Samples with subject=None go to train (no leakage risk).
    Falls back to torch's random_split when every subject is None
    (preserves current behaviour for manifest-less datasets).
    """
    n = len(dataset)
    n_val_target = max(1, int(round(n * val_fraction)))
    n_train_target = max(1, n - n_val_target)

    subjects = getattr(dataset, "subjects", [None] * n)
    if not subjects or all(s is None for s in subjects):
        return torch.utils.data.random_split(
            dataset, [n_train_target, n_val_target],
            generator=torch.Generator().manual_seed(seed),
        )

    by_subj: dict[str, list[int]] = {}
    no_subj: list[int] = []
    for i, s in enumerate(subjects):
        if s is None:
            no_subj.append(i)
        else:
            by_subj.setdefault(s, []).append(i)

    rng = np.random.default_rng(seed)
    order = list(by_subj.keys())
    rng.shuffle(order)

    val_idx: list[int] = []
    val_subjects: set[str] = set()
    for s in order:
        if len(val_idx) >= n_val_target:
            break
        val_idx.extend(by_subj[s])
        val_subjects.add(s)

    train_idx = sorted(
        no_subj + [i for s in order if s not in val_subjects for i in by_subj[s]]
    )
    val_idx = sorted(val_idx)
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline_subjects.py -v`
Expected: PASS (all five tests in this file).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_eval_baseline_subjects.py
git commit -m "feat(pad-core/eval): subject_disjoint_split helper with random fallback"
```

---

## Task 5: Integrate ISO metrics + subject-disjoint split into `train_and_cross_domain_eval`

Fix the dev threshold at `target_apcer`, apply it to the cross-domain set, return additive keys. Keep all existing keys numerically unchanged where possible (cross-domain EER is invariant to threshold choice; in-domain numbers can shift slightly when a manifest is present because the split is now subject-disjoint, but this is the intentional, trust-improving change).

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py`
- Modify (test): `pad-synth-core/tests/test_eval_baseline_subjects.py`

- [ ] **Step 1: Write the failing integration test**

Append to `pad-synth-core/tests/test_eval_baseline_subjects.py`:

```python
from pad_synth_core.eval.baseline import train_and_cross_domain_eval


def test_train_and_eval_returns_iso_metrics(tmp_path):
    train_root = _build_ds(tmp_path / "train")._dataset_root() if False else None  # placeholder
    # Build a real on-disk training set with a manifest (so subject-disjoint kicks in)
    train_ds = _build_ds(tmp_path / "train")
    train_root = Path(train_ds.items[0][0]).parents[2]  # face/<x>/<file>.jpg -> root

    # Cross-domain eval set: print + replay attacks with their own manifest.
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
    # Old keys still present and finite.
    assert 0.0 <= out["eer_in_domain"] <= 1.0
    assert 0.0 <= out["eer_cross_domain"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "pad-synth-core/tests/test_eval_baseline_subjects.py::test_train_and_eval_returns_iso_metrics" -v`
Expected: FAIL — `TypeError: train_and_cross_domain_eval() got an unexpected keyword argument 'target_apcer'`.

- [ ] **Step 3: Refactor `train_and_cross_domain_eval`**

In `pad-synth-core/src/pad_synth_core/eval/baseline.py`:

(a) Add the new imports near the top (with the other `from pad_synth_core.eval.metrics import` line):

```python
from pad_synth_core.eval.metrics import apcer_bpcer_acer, threshold_at_apcer
```

(b) Add this helper just above `train_and_cross_domain_eval`:

```python
def _score_dataset(model, dataset, batch_size, device):
    """Run inference and return (scores, labels, attack_types) aligned 1:1
    with the dataset (or Subset) order, with no shuffling."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1].cpu().tolist()
            scores.extend(probs)
            labels.extend(y.tolist())
    if isinstance(dataset, torch.utils.data.Subset):
        attack_types = [dataset.dataset.attack_types[i] for i in dataset.indices]
    else:
        attack_types = list(dataset.attack_types)
    return scores, labels, attack_types
```

(c) Replace the entire body of `train_and_cross_domain_eval` with the version below. The signature gains one optional keyword (`target_apcer: float = 0.05`); every existing keyword is preserved.

```python
def train_and_cross_domain_eval(
    train_root: Path,
    eval_root: Path | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    target_apcer: float = 0.05,
) -> dict[str, Any]:
    """Train on train_root; eval in-domain (held-out 25 percent split, now
    subject-disjoint when a manifest is present) and optionally cross-domain
    (full eval_root if provided). Adds ISO 30107-3 metrics at a dev-fixed
    threshold (target APCER = `target_apcer`) on top of the existing EER
    reporting -- all new keys are additive."""
    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")

    train_ds_full = TinyPADDataset(train_root)
    train_ds, val_ds = subject_disjoint_split(train_ds_full, val_fraction=0.25, seed=seed)
    n_train, n_val = len(train_ds), len(val_ds)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = (model_factory or TinyCNN)().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    # In-domain scoring (dev split).
    dev_scores, dev_labels, dev_atypes = _score_dataset(model, val_ds, batch_size, dev)
    in_eer = compute_eer(dev_scores, dev_labels)
    in_acc = (
        sum(int((s >= 0.5) == y) for s, y in zip(dev_scores, dev_labels)) / max(len(dev_scores), 1)
    )
    threshold, _ = threshold_at_apcer(dev_scores, dev_labels, dev_atypes, target_apcer)

    cross_eer: float | None = None
    cross_acc: float | None = None
    n_val_cross: int | None = None
    apcer_per_pai: dict[str, float] | None = None
    apcer_max: float | None = None
    bpcer: float | None = None
    acer: float | None = None
    if eval_root is not None:
        cross_ds = TinyPADDataset(eval_root)
        cross_scores, cross_labels, cross_atypes = _score_dataset(model, cross_ds, batch_size, dev)
        cross_eer = compute_eer(cross_scores, cross_labels)
        cross_acc = (
            sum(int((s >= 0.5) == y) for s, y in zip(cross_scores, cross_labels))
            / max(len(cross_scores), 1)
        )
        n_val_cross = len(cross_ds)
        apcer_per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(
            cross_scores, cross_labels, cross_atypes, threshold,
        )

    return {
        # Existing keys -- preserved.
        "eer_in_domain": in_eer,
        "val_accuracy_in_domain": in_acc,
        "n_train": n_train,
        "n_val_in_domain": n_val,
        "eer_cross_domain": cross_eer,
        "val_accuracy_cross_domain": cross_acc,
        "n_val_cross_domain": n_val_cross,
        # New additive keys.
        "threshold": float(threshold),
        "target_apcer": float(target_apcer),
        "apcer_cross_domain": apcer_max,
        "bpcer_cross_domain": bpcer,
        "acer_cross_domain": acer,
        "apcer_per_pai_cross_domain": apcer_per_pai,
    }
```

(d) The previous `_eval_loader` function is no longer called anywhere in the file (the metrics path replaces it). Delete it.

- [ ] **Step 4: Run the new test + the existing baseline tests**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline_subjects.py pad-synth-core/tests/test_eval_baseline.py pad-synth-core/tests/test_baseline_extensions.py -v`
Expected: all pass. The existing tests use `set(out.keys()) >= {...}` so additive keys are fine; the test datasets they build have no manifest, so `subject_disjoint_split` falls back to `random_split` (today's behaviour) and existing in-domain numbers are unchanged.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_eval_baseline_subjects.py
git commit -m "feat(pad-core/eval): integrate ISO metrics + subject-disjoint split (additive return keys)"
```

---

## Task 6: Surface new metrics in `scripts/spark_sweep.py`

**Files:**
- Modify: `scripts/spark_sweep.py`

- [ ] **Step 1: Add the new CSV header columns**

In `scripts/spark_sweep.py`, find the CSV-header write (the line that calls `w.writerow(["capacity", "data_level", ...])` near the `summary.csv` opening). Extend the header list to:

```python
        w.writerow([
            "capacity", "data_level", "seed",
            "eer_in_domain", "eer_cross_domain", "train_seconds",
            "acer_cross_domain", "apcer_cross_domain", "bpcer_cross_domain", "threshold",
        ])
```

- [ ] **Step 2: Write the new fields per cell (CSV row + JSON record)**

In the same file, find the per-cell write where `rec = {...}` is built and `summary.csv` is appended. Extend the per-cell `rec` dict with the new keys from `train_and_cross_domain_eval`'s return:

```python
            "threshold": float(out["threshold"]),
            "target_apcer": float(out["target_apcer"]),
            "apcer_cross_domain": (
                float(out["apcer_cross_domain"])
                if out["apcer_cross_domain"] is not None else None
            ),
            "bpcer_cross_domain": (
                float(out["bpcer_cross_domain"])
                if out["bpcer_cross_domain"] is not None else None
            ),
            "acer_cross_domain": (
                float(out["acer_cross_domain"])
                if out["acer_cross_domain"] is not None else None
            ),
            "apcer_per_pai_cross_domain": out["apcer_per_pai_cross_domain"],
```

Update the corresponding CSV-append `csv.writer(fh).writerow([...])` to write the new columns in the same order as the header:

```python
            csv.writer(fh).writerow([
                L, D, seed, rec["eer_in_domain"], rec["eer_cross_domain"],
                f"{elapsed:.2f}",
                rec["acer_cross_domain"], rec["apcer_cross_domain"],
                rec["bpcer_cross_domain"], rec["threshold"],
            ])
```

(`apcer_per_pai_cross_domain` is a dict and goes in the per-cell JSON only, not the flat CSV.)

- [ ] **Step 3: Smoke-test the sweep against the fixture pipeline**

Run: `.venv/bin/python -m pytest tests/test_spark_sweep.py -v`
Expected: PASS. (If that test does an end-to-end sweep, it now produces the new columns; the existing assertions about cell counts / file paths are unaffected.)

- [ ] **Step 4: Commit**

```bash
git add scripts/spark_sweep.py
git commit -m "feat(pad-core/sweep): emit ISO metrics columns (acer/apcer/bpcer/threshold) + per-PAI in JSON"
```

---

## Task 7: Full-suite + lint checkpoint

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: green (baseline before this branch was 194 passed / 1 skipped; this adds the new tests in Tasks 1, 3, 4, 5). If any pre-existing test fails, do not "fix" it by relaxing assertions — investigate, since the changes here were designed to be additive.

- [ ] **Step 2: Lint the new/modified files**

Run:
```bash
uvx ruff check --select E,F,B,UP --line-length 100 --ignore E501 \
  pad-synth-core/src/pad_synth_core/eval/metrics.py \
  pad-synth-core/src/pad_synth_core/eval/baseline.py \
  pad-synth-core/tests/test_eval_metrics.py \
  pad-synth-core/tests/test_eval_baseline_subjects.py \
  scripts/spark_sweep.py
```
Expected: `All checks passed!`. (Do not run with the `I`/isort rule from the repo root — `uvx ruff` misclassifies the `src`-layout packages as third-party and will spuriously rewrite import blocks across the whole codebase. House style is "blank line between third-party and first-party imports"; match it by hand.)

- [ ] **Step 3: Commit (only if lint fixes were needed)**

```bash
git add -A
git commit -m "style(pad-eval-metrics): ruff fixes"
```

---

## Self-review notes

- **Spec coverage:** §4 metric defs → Task 1 (with hand-computed cases); §5 manifest plumbing + subject-disjoint split → Tasks 3 + 4; §6 dev-fixed threshold applied to test → Task 5; §7 additive sweep output → Task 6; §8 testing → covered across Tasks 1, 3, 4, 5 + the suite checkpoint in Task 7.
- **Backward compatibility:** every existing key in `train_and_cross_domain_eval`'s return is preserved unchanged; the new keyword `target_apcer` is optional with a default; `compute_eer` is re-exported so the old import path still works; `subject_disjoint_split` falls back to `random_split` when no manifest is present, which is the entire universe of the existing eval tests' fixtures.
- **No data/sweep code beyond the eval and the sweep writer is touched** — the spec promised no eval-API breakage and no protocol layer (leave-one-out) in this cycle.
- **Tie-break in `threshold_at_apcer`** (highest threshold under the APCER budget = lowest BPCER under the same budget) is enforced by the `t >= best_thr` step in the candidate scan and verified by the "no higher candidate also satisfies" assertion in `test_threshold_at_apcer_respects_budget_and_prefers_higher_threshold`.
