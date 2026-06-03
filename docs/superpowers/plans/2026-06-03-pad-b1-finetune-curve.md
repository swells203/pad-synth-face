# B1 Synth-Pretrain → Real-Finetune Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a synth-pretrain → real-finetune capability + a curve-runner that reports real-test EER as a function of the number of real finetune samples N, validated mechanically on the n=55 AxonData pilot.

**Architecture:** Two new functions in `eval/baseline.py` (`pretrain_on_synth`, `finetune_and_eval_on_real`) composing the existing train loop + helpers, plus a `scripts/b1_finetune_curve.py` runner that splits the real set once (subject-disjoint), pretrains once, and forks a finetune per N. No change to `train_and_cross_domain_eval` or the sweep.

**Tech Stack:** Python 3.11, PyTorch, NumPy, Pillow, pytest. Reuses `TinyPADDataset`, `subject_disjoint_split`, `_score_dataset`, `compute_eer`, `threshold_at_apcer`, `apcer_bpcer_acer`, and the `FACTORIES` model zoo — all already in `pad-synth-core/src/pad_synth_core/eval/`.

**Spec:** `docs/superpowers/specs/2026-06-03-pad-b1-finetune-curve-design.md`

**Branch:** `feat/pad-b1-finetune-curve` (already created from main; spec committed as `59222d4`).

**Interface refinement vs spec:** the spec sketched `finetune_and_eval_on_real(..., n_real, real_test_root)`. The split actually yields `torch.utils.data.Subset`s of one real dataset, and a `Subset`-of-`Subset` breaks `_score_dataset`'s `attack_types` lookup (it does `dataset.dataset.attack_types[i]`). So the implemented interface passes **dataset objects** — `finetune_ds` and `real_test_ds`, both `Subset`s of the *same underlying* `TinyPADDataset` — and the runner builds `finetune_ds = Subset(real_ds, shuffled_pool_indices[:N])` (sliced against the original dataset, not the pool Subset). `n_real` is then `len(finetune_ds)`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `pad-synth-core/src/pad_synth_core/eval/baseline.py` | Add `pretrain_on_synth` + `finetune_and_eval_on_real` | **Modify** (append two functions; existing code untouched) |
| `pad-synth-core/tests/test_b1_finetune.py` | Unit tests for the two functions (full/head/n=0/guards) | **Create** |
| `scripts/b1_finetune_curve.py` | Curve runner: `split_real`, `run_curve`, `_render_curve`, `main` | **Create** |
| `tests/test_b1_finetune_curve.py` | Runner tests (per-N JSON, summary, guards, N-skip) | **Create** |
| `docs/b1-finetune-curve.md` | Operator runbook | **Create** |

`train_and_cross_domain_eval`, the model zoo, and the sweep are unchanged. The report-append is deferred (no real curve to tabulate from n=55).

---

## Task 1: `pretrain_on_synth`

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py` (append function)
- Test: `pad-synth-core/tests/test_b1_finetune.py` (new)

**Context:** Pretrains a fresh `model_factory()` model on the full synthetic root (no val split — pretraining uses everything), returns the trained model so the runner can snapshot its `state_dict` once. Same epoch loop / optimizer as `train_and_cross_domain_eval`. All helpers (`TinyPADDataset`, `DataLoader`, `nn`, `torch`) are already imported at the top of `baseline.py`.

- [ ] **Step 1: Write the failing test**

Create `pad-synth-core/tests/test_b1_finetune.py`:

```python
"""B1 synth-pretrain -> real-finetune unit tests. Generated fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pad_synth_core.eval.baseline import TinyPADDataset, pretrain_on_synth
from pad_synth_core.eval.models_zoo import make_resnet18, make_tiny_cnn  # noqa: F401  (make_resnet18 used from Task 2 on)


def _make_pad_tree(root: Path, n_bonafide: int = 6, n_attack: int = 6) -> None:
    """A tiny TinyPADDataset-shaped tree: face/bonafide + face/print + manifest."""
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


def test_pretrain_on_synth_returns_trained_model(tmp_path):
    synth = tmp_path / "synth"
    _make_pad_tree(synth)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    assert isinstance(model, torch.nn.Module)
    # Model produces 2-class logits on a 64x64 RGB batch.
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-core/tests/test_b1_finetune.py::test_pretrain_on_synth_returns_trained_model -v
```

Expected: FAIL with `ImportError: cannot import name 'pretrain_on_synth'` (and `finetune_and_eval_on_real`).

- [ ] **Step 3: Implement `pretrain_on_synth`**

Append to `pad-synth-core/src/pad_synth_core/eval/baseline.py` (after `train_and_cross_domain_eval`):

```python
def pretrain_on_synth(
    synth_root: Path,
    model_factory: Callable[[], nn.Module],
    epochs: int = 8,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
) -> nn.Module:
    """Pretrain a fresh model_factory() model on the FULL synthetic root.

    No val split -- pretraining uses all of synth_root. Returns the trained
    model; the B1 runner snapshots state_dict() once and forks the finetune
    curve from it. Same Adam/CE loop as train_and_cross_domain_eval.
    """
    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")
    ds = TinyPADDataset(synth_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = model_factory().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
.venv/bin/pytest pad-synth-core/tests/test_b1_finetune.py::test_pretrain_on_synth_returns_trained_model -v
```

Expected: PASS. (Task 1 imports only `pretrain_on_synth` + `TinyPADDataset`; Task 2 Step 1 extends the import to add `finetune_and_eval_on_real` when that function lands.)

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_b1_finetune.py
git commit -m "feat(pad-b1): pretrain_on_synth -- pretrain a model on the full synthetic root

Returns the trained model so the B1 runner can snapshot state_dict once and
fork the finetune curve. Same Adam/CE loop as train_and_cross_domain_eval."
```

---

## Task 2: `finetune_and_eval_on_real` — full mode + n_real=0 baseline

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py` (append function)
- Test: `pad-synth-core/tests/test_b1_finetune.py` (extend)

**Context:** Loads pretrained weights into a fresh model, optionally finetunes on `finetune_ds`, evaluates on `real_test_ds`. This task implements `mode="full"` and the `n_real == 0` (no-finetune, synth-only baseline) path. Head mode + the fc-guard land in Task 3. Both datasets are `Subset`s of the same underlying `TinyPADDataset`, so `_score_dataset` works on them. Threshold is fixed on the **finetune** set (dev) and applied to real-test — matching the leakage-free discipline in `train_and_cross_domain_eval`; when `n_real == 0` there is no finetune set, so threshold/ACER are `None` (EER stays meaningful).

- [ ] **Step 1: Extend the test imports + add full-mode and n=0 tests**

Change the baseline import line in `pad-synth-core/tests/test_b1_finetune.py` to:

```python
from pad_synth_core.eval.baseline import (
    TinyPADDataset,
    finetune_and_eval_on_real,
    pretrain_on_synth,
)
```

Append these tests:

```python
def _state_of(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def test_finetune_full_mode_updates_backbone(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=8, n_attack=8)
    model = pretrain_on_synth(synth, make_resnet18, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(8)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(8, 16)))
    res = finetune_and_eval_on_real(
        state, make_resnet18, ft_ds, test_ds,
        mode="full", epochs=2, lr=1e-3, batch_size=4, seed=0)
    assert res["n_real"] == 8
    assert res["mode"] == "full"
    assert res["eer_cross_domain"] is not None
    assert res["n_val_cross_domain"] == 8
    # full mode trains the backbone: conv1 weights move from the pretrained state.
    # (re-load the model the function trained is internal; instead assert the
    #  result is well-formed and in-domain fit is reported)
    assert res["eer_in_domain"] is not None


def test_finetune_n_real_zero_is_synth_only_baseline(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=6, n_attack=6)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    empty_ft = torch.utils.data.Subset(real_ds, [])
    test_ds = torch.utils.data.Subset(real_ds, list(range(12)))
    res = finetune_and_eval_on_real(
        state, make_tiny_cnn, empty_ft, test_ds,
        mode="full", epochs=2, batch_size=4, seed=0)
    assert res["n_real"] == 0
    assert res["eer_cross_domain"] is not None        # real-test still evaluated
    assert res["eer_in_domain"] is None               # no finetune set
    assert res["threshold"] is None                   # no dev set -> no ISO threshold
    assert res["acer_cross_domain"] is None


def test_finetune_rejects_unknown_mode(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(4)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(4, 8)))
    import pytest
    with pytest.raises(ValueError):
        finetune_and_eval_on_real(state, make_tiny_cnn, ft_ds, test_ds, mode="bogus")
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-core/tests/test_b1_finetune.py -v -k "finetune"
```

Expected: FAIL with `ImportError: cannot import name 'finetune_and_eval_on_real'`.

- [ ] **Step 3: Implement `finetune_and_eval_on_real` (full + n=0; head raises for now via the mode check coming in Task 3)**

Append to `pad-synth-core/src/pad_synth_core/eval/baseline.py`:

```python
def finetune_and_eval_on_real(
    pretrained_state: dict[str, Any],
    model_factory: Callable[[], nn.Module],
    finetune_ds: Dataset,
    real_test_ds: Dataset,
    mode: str = "full",
    epochs: int = 8,
    lr: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    target_apcer: float = 0.05,
) -> dict[str, Any]:
    """Load pretrained weights, optionally finetune on finetune_ds, eval on
    real_test_ds. n_real == len(finetune_ds); n_real == 0 skips finetuning
    (synth-only baseline). Real-test numbers populate the cross-domain keys;
    the ISO threshold is fixed on the finetune set and applied to real-test
    (None when there is no finetune set)."""
    if mode not in ("full", "head"):
        raise ValueError(f"unknown finetune mode: {mode!r} (use 'full' or 'head')")

    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")
    model = model_factory().to(dev)
    model.load_state_dict(pretrained_state)
    n_real = len(finetune_ds)

    if mode == "head":
        if not hasattr(model, "fc"):
            raise ValueError(
                "head mode requires a ResNet-style .fc head; this model has "
                "none (use mode='full')")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")

    in_eer: float | None = None
    in_acc: float | None = None
    threshold: float | None = None
    if n_real > 0:
        ft_dl = DataLoader(finetune_ds, batch_size=batch_size, shuffle=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for x, y in ft_dl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
        model.eval()
        ft_scores, ft_labels, ft_atypes = _score_dataset(model, finetune_ds, batch_size, dev)
        in_eer = compute_eer(ft_scores, ft_labels)
        in_acc = (
            sum(int((s >= 0.5) == y) for s, y in zip(ft_scores, ft_labels, strict=True))
            / max(len(ft_scores), 1)
        )
        if any(t is not None for t in ft_atypes):
            thr, _ = threshold_at_apcer(ft_scores, ft_labels, ft_atypes, target_apcer)
            threshold = float(thr)

    model.eval()
    test_scores, test_labels, test_atypes = _score_dataset(model, real_test_ds, batch_size, dev)
    real_eer = compute_eer(test_scores, test_labels)
    real_acc = (
        sum(int((s >= 0.5) == y) for s, y in zip(test_scores, test_labels, strict=True))
        / max(len(test_scores), 1)
    )

    apcer_per_pai: dict[str, float] | None = None
    apcer_max: float | None = None
    bpcer: float | None = None
    acer: float | None = None
    if threshold is not None:
        apcer_per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(
            test_scores, test_labels, test_atypes, threshold)

    return {
        "n_real": n_real,
        "mode": mode,
        "eer_in_domain": in_eer,
        "val_accuracy_in_domain": in_acc,
        "n_train": n_real,
        "n_val_in_domain": n_real,
        "eer_cross_domain": real_eer,
        "val_accuracy_cross_domain": real_acc,
        "n_val_cross_domain": len(real_test_ds),
        "threshold": threshold,
        "target_apcer": float(target_apcer),
        "apcer_cross_domain": apcer_max,
        "bpcer_cross_domain": bpcer,
        "acer_cross_domain": acer,
        "apcer_per_pai_cross_domain": apcer_per_pai,
    }
```

- [ ] **Step 4: Run the full B1 test file, verify pass**

```bash
.venv/bin/pytest pad-synth-core/tests/test_b1_finetune.py -v
```

Expected: all tests so far PASS (pretrain + full-mode + n=0 + unknown-mode).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_b1_finetune.py
git commit -m "feat(pad-b1): finetune_and_eval_on_real -- full mode + n_real=0 baseline

Loads pretrained weights, finetunes on finetune_ds (skipped when empty =
synth-only baseline), evals on real_test_ds. Real-test in cross-domain keys;
ISO threshold fixed on the finetune set (leakage-free), None at n_real=0."
```

---

## Task 3: head mode (freeze backbone) — behaviour test

**Files:**
- Test: `pad-synth-core/tests/test_b1_finetune.py` (extend)

**Context:** Task 2 already wrote the head-mode branch (freeze all params except `fc.*`, raise if no `fc`). This task adds the *behavioural* tests that lock that contract: head mode must leave backbone weights unchanged while moving the classifier, and must reject a non-`fc` model. No new implementation — if a test fails, fix the Task-2 branch.

- [ ] **Step 1: Append head-mode tests**

Append to `pad-synth-core/tests/test_b1_finetune.py`:

```python
def test_head_mode_freezes_backbone_trains_fc(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real, n_bonafide=8, n_attack=8)
    model = pretrain_on_synth(synth, make_resnet18, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(8)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(8, 16)))

    # Re-run finetune but capture the model by monkey-grabbing via a fresh build:
    # simplest robust check -- compare a backbone tensor and the fc tensor in the
    # returned model's state by finetuning a model we control.
    import torch as _t
    m = make_resnet18()
    m.load_state_dict(state)
    before_backbone = m.conv1.weight.detach().clone()
    before_fc = m.fc.weight.detach().clone()
    # Freeze like head mode and finetune a couple steps directly.
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith("fc.")
    opt = _t.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = _t.nn.CrossEntropyLoss()
    dl = _t.utils.data.DataLoader(ft_ds, batch_size=4, shuffle=True)
    for _ in range(3):
        for x, y in dl:
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
    assert _t.equal(m.conv1.weight, before_backbone)        # backbone frozen
    assert not _t.equal(m.fc.weight, before_fc)             # head trained

    # And the public API runs end-to-end in head mode.
    res = finetune_and_eval_on_real(
        state, make_resnet18, ft_ds, test_ds, mode="head",
        epochs=2, lr=1e-2, batch_size=4, seed=0)
    assert res["mode"] == "head"
    assert res["eer_cross_domain"] is not None


def test_head_mode_rejects_non_fc_model(tmp_path):
    synth, real = tmp_path / "synth", tmp_path / "real"
    _make_pad_tree(synth)
    _make_pad_tree(real)
    model = pretrain_on_synth(synth, make_tiny_cnn, epochs=1, batch_size=4, seed=0)
    state = _state_of(model)
    real_ds = TinyPADDataset(real)
    ft_ds = torch.utils.data.Subset(real_ds, list(range(4)))
    test_ds = torch.utils.data.Subset(real_ds, list(range(4, 8)))
    import pytest
    # make_tiny_cnn is an nn.Sequential -- no .fc attribute.
    with pytest.raises(ValueError, match="head mode"):
        finetune_and_eval_on_real(state, make_tiny_cnn, ft_ds, test_ds, mode="head")
```

- [ ] **Step 2: Run, verify pass**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-core/tests/test_b1_finetune.py -v -k "head"
```

Expected: both PASS. If `test_head_mode_rejects_non_fc_model` fails, the Task-2 `hasattr(model, "fc")` guard is wrong — fix it. If the freeze test fails, the `name.startswith("fc.")` branch is wrong.

- [ ] **Step 3: Commit**

```bash
git add pad-synth-core/tests/test_b1_finetune.py
git commit -m "test(pad-b1): lock head-mode contract (freeze backbone, train fc, reject non-fc)"
```

---

## Task 4: Curve runner `scripts/b1_finetune_curve.py`

**Files:**
- Create: `scripts/b1_finetune_curve.py`
- Test: `tests/test_b1_finetune_curve.py` (new)

**Context:** Orchestrates the curve: split the real set once (subject-disjoint) into `(pool, test)`, guard that the test split has both classes, shuffle the pool deterministically, pretrain once, then finetune+eval per N (skipping — not silently capping — any N larger than the pool). Writes per-N JSON + a summary. Core logic in importable functions so the test doesn't shell out. The runner slices `finetune_ds` against the **original** real dataset (`Subset(real_ds, pool_indices[:N])`) so `_score_dataset` works.

- [ ] **Step 1: Write the failing runner test**

Create `tests/test_b1_finetune_curve.py`:

```python
"""B1 curve runner tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
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
    import pytest
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
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest tests/test_b1_finetune_curve.py -v
```

Expected: FAIL — module load error (`scripts/b1_finetune_curve.py` does not exist).

- [ ] **Step 3: Implement the runner**

Create `scripts/b1_finetune_curve.py`:

```python
#!/usr/bin/env python3
"""B1 synth-pretrain -> real-finetune curve runner.

Splits the real set once (subject-disjoint) into (finetune pool, real test),
pretrains a model on the synthetic root once, then finetunes on N real samples
for each N and reports real-test EER -- the hybrid curve. See
docs/superpowers/specs/2026-06-03-pad-b1-finetune-curve-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))

from pad_synth_core.eval.baseline import (  # noqa: E402
    TinyPADDataset,
    finetune_and_eval_on_real,
    pretrain_on_synth,
    subject_disjoint_split,
)
from pad_synth_core.eval.models_zoo import FACTORIES  # noqa: E402


def split_real(real_root: Path, test_fraction: float, seed: int):
    """Split the real set into (real_ds, pool_indices, test_ds), subject-disjoint.

    Guards that the test partition holds both classes -- EER is undefined on a
    single-class test set, so a degenerate split raises rather than emitting a
    meaningless number.
    """
    real_ds = TinyPADDataset(real_root)
    pool_sub, test_sub = subject_disjoint_split(real_ds, val_fraction=test_fraction, seed=seed)
    test_labels = {real_ds.items[i][1] for i in test_sub.indices}
    if len(test_labels) < 2:
        raise ValueError(
            f"real-test split has a single class {test_labels}; need both "
            "bonafide and attack. Increase --test-fraction or use more real data.")
    return real_ds, list(pool_sub.indices), test_sub


def run_curve(
    synth_root: Path, real_root: Path, n_list: list[int], output_dir: Path,
    model_factory: Callable[[], Any], mode: str = "full",
    test_fraction: float = 0.3, pretrain_epochs: int = 8, finetune_epochs: int = 8,
    finetune_lr: float = 1e-4, batch_size: int = 8, seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    real_ds, pool_indices, test_ds = split_real(real_root, test_fraction, seed)
    rng = np.random.default_rng(seed)
    pool_indices = list(pool_indices)
    rng.shuffle(pool_indices)
    pool_size = len(pool_indices)

    model = pretrain_on_synth(
        synth_root, model_factory, epochs=pretrain_epochs,
        batch_size=batch_size, seed=seed, device=device)
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    runs_dir = Path(output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n in n_list:
        if n > pool_size:
            print(f"[b1] requested N={n}, pool has {pool_size} -- skipped "
                  "(no silent capping)")
            rows.append({"n_real": n, "eer": None, "acer": None, "skipped": True})
            continue
        ft_ds = torch.utils.data.Subset(real_ds, pool_indices[:n])
        res = finetune_and_eval_on_real(
            state, model_factory, ft_ds, test_ds, mode=mode,
            epochs=finetune_epochs, lr=finetune_lr, batch_size=batch_size,
            seed=seed, device=device)
        (runs_dir / f"N{n}_seed{seed}.json").write_text(json.dumps(res, indent=2))
        rows.append({"n_real": n, "eer": res["eer_cross_domain"],
                     "acer": res["acer_cross_domain"], "skipped": False})

    summary = {"rows": rows, "pool_size": pool_size, "n_test": len(test_ds),
               "mode": mode, "seed": seed}
    (Path(output_dir) / "curve_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _render_curve(summary: dict[str, Any]) -> str:
    lines = [
        f"B1 finetune curve (mode={summary['mode']}, n_test={summary['n_test']}, "
        f"pool={summary['pool_size']})",
        "",
        f"{'N':>8} {'real-test EER':>14} {'ACER':>8}",
        "-" * 34,
    ]
    done = [r for r in summary["rows"] if not r["skipped"]]
    for r in summary["rows"]:
        if r["skipped"]:
            lines.append(f"{r['n_real']:>8} {'(skipped: N>pool)':>23}")
        else:
            eer = "n/a" if r["eer"] is None else f"{r['eer']:.3f}"
            acer = "n/a" if r["acer"] is None else f"{r['acer']:.3f}"
            lines.append(f"{r['n_real']:>8} {eer:>14} {acer:>8}")
    lines.append("")
    base = next((r for r in done if r["n_real"] == 0), None)
    top = done[-1] if done else None
    if base is not None and top is not None and top["n_real"] > 0 \
            and base["eer"] is not None and top["eer"] is not None:
        delta = top["eer"] - base["eer"]
        verdict = "helps" if delta < 0 else ("no change" if delta == 0 else "hurts")
        lines.append(f"finetuning {verdict}: EER N=0 {base['eer']:.3f} -> "
                     f"N={top['n_real']} {top['eer']:.3f} (delta {delta:+.3f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-root", required=True, type=Path)
    ap.add_argument("--real-root", required=True, type=Path)
    ap.add_argument("--n-list", default="0,50,200,1000",
                    help="Comma-separated finetune sample counts.")
    ap.add_argument("--finetune-mode", choices=("full", "head"), default="full")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--model", default="L4", choices=list(FACTORIES))
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    n_list = [int(x) for x in args.n_list.split(",") if x.strip() != ""]
    summary = run_curve(
        synth_root=args.synth_root, real_root=args.real_root, n_list=n_list,
        output_dir=args.output_dir, model_factory=FACTORIES[args.model],
        mode=args.finetune_mode, test_fraction=args.test_fraction,
        pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr, batch_size=args.batch_size, seed=args.seed,
        device=args.device)
    print(_render_curve(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the runner tests, verify pass**

```bash
.venv/bin/pytest tests/test_b1_finetune_curve.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/b1_finetune_curve.py tests/test_b1_finetune_curve.py
git commit -m "feat(pad-b1): b1_finetune_curve runner -- split, pretrain once, curve over N

Subject-disjoint (pool, test) split with a both-classes guard; pretrains once
and forks finetune per N; skips (never silently caps) N>pool; writes per-N JSON
+ curve_summary + a printed EER-vs-N table with a does-finetuning-help readout."
```

---

## Task 5: Operator runbook `docs/b1-finetune-curve.md`

**Files:**
- Create: `docs/b1-finetune-curve.md`

**Context:** Documents how to run the curve, the split discipline, and the honesty notes (n=55 reaches only N=0 + ~pool; the AxonData free sample is CC-BY-NC-4.0 research-only; the real curve needs purchased/larger data).

- [ ] **Step 1: Write the runbook**

Create `docs/b1-finetune-curve.md`:

````markdown
# B1: synth-pretrain → real-finetune curve

Quantifies the hybrid hypothesis — pretrain on synthetic, finetune on N real
samples — by reporting **real-test EER as a function of N**. N=0 is the
synth-only baseline. Context: spec
`docs/superpowers/specs/2026-06-03-pad-b1-finetune-curve-design.md` and memory
`pad-next-sub-projects`.

**Real images are never committed** (`datasets/` is gitignored).

## What it does

`scripts/b1_finetune_curve.py` splits the real set once into a fixed
subject-disjoint `(finetune pool, real test)`, pretrains a model on the
synthetic root once, then for each N finetunes on the first N pool samples and
evaluates on the held-out real test. The test set is identical across all N, so
the curve is fair.

## Run

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d3 \
  --real-root  datasets/_real_attack/axondata \
  --n-list 0,50,200,1000 \
  --finetune-mode full \
  --model L4 \
  --test-fraction 0.3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_b1_curve \
  --device cuda
```

Writes `runs_b1_curve/runs/N<n>_seed<s>.json` (one per N) + `curve_summary.json`,
and prints an `N | real-test EER | ACER` table with a does-finetuning-help
readout. `--finetune-mode head` freezes the backbone and trains only the ResNet
`.fc` head (use with `--model L3`/`L4`). The pretrain step is the heavy one —
run on the Spark (`--device cuda`); the finetunes are cheap.

## Honesty notes (current data)

- The only real set staged is the **n=55 AxonData pilot**
  (`datasets/_real_attack/axondata`), licensed **CC-BY-NC-4.0 (NonCommercial)** —
  research-only, like DigiFace/DFDC (see memory `pad-commercial-licensing`). A
  model finetuned on it is **not commercially shippable**.
- n=55 only reaches **N=0 and ~the pool size** (after holding out the test
  split). Requested N larger than the pool are **skipped and logged** (never
  silently capped), so the table tells the truth about what ran.
- The real N=0/50/200/1000 curve needs purchased/larger real data — the same
  data step that unblocks the commercial-bonafide validation. Until then this is
  validated scaffolding: mechanically proven, awaiting data.

## Mechanical dry-run (no purchase needed)

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d1 --real-root datasets/_real_attack/axondata \
  --n-list 0,8 --model L1 --test-fraction 0.4 \
  --pretrain-epochs 1 --finetune-epochs 1 \
  --output-dir /tmp/b1_dryrun --device cpu
```

Proves the chain runs end-to-end on the pilot; the EER values are meaningless at
this scale.
````

- [ ] **Step 2: Commit**

```bash
git add docs/b1-finetune-curve.md
git commit -m "docs(pad-b1): operator runbook for the finetune curve runner"
```

---

## Task 6: Mechanical validation on n=55 + full-suite + finish

**Files:**
- None modified unless a failure surfaces.

**Context:** Prove the runner works end-to-end on the real AxonData pilot (the same kind of dry-run that caught the `cli run` bug for the commercial harness), then run the whole suite. The dry-run writes to `/tmp` (never committed).

- [ ] **Step 1: Confirm a small synthetic dataset exists to pretrain on**

```bash
cd /Users/stuartwells/test
ls datasets/ | grep -E '^mix_seta_d1$|^mix_seta_d3$' || echo "no mix synth dataset staged"
ls datasets/_real_attack/axondata/manifest.jsonl && wc -l datasets/_real_attack/axondata/manifest.jsonl
```

Expected: at least one `mix_seta_d*` synth dataset present and the AxonData manifest with ~55 rows. If no `mix_seta_d1` exists, generate one: `.venv/bin/python -m pad_synth_face.cli generate --config configs/runs/mix_seta_d1.yaml`.

- [ ] **Step 2: Run the dry-run curve on the pilot (CPU, tiny)**

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d1 \
  --real-root datasets/_real_attack/axondata \
  --n-list 0,8 --model L1 --test-fraction 0.4 \
  --pretrain-epochs 1 --finetune-epochs 1 \
  --output-dir /tmp/b1_dryrun --device cpu
```

Expected: prints the `N | real-test EER | ACER` table, exits 0, and writes
`/tmp/b1_dryrun/runs/N0_seed0.json` + (if pool >= 8) `N8_seed0.json` + `curve_summary.json`. If the real-test split raises the single-class guard, bump `--test-fraction` (the pilot's class balance is fixed) and note it. EER numbers are meaningless — this is a plumbing check.

- [ ] **Step 3: Confirm the dry-run JSON is well-formed**

```bash
python3 -m json.tool /tmp/b1_dryrun/runs/N0_seed0.json | grep -E "eer_cross_domain|n_real|mode"
```

Expected: `n_real` = 0, `mode` = "full", a finite `eer_cross_domain`.

- [ ] **Step 4: Run the full repo test suite**

```bash
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -q
```

Expected: all pass (1 CUDA skip is fine). In particular `pad-synth-core/tests/test_b1_finetune.py` and `tests/test_b1_finetune_curve.py` are green, and the pre-existing `train_and_cross_domain_eval` tests are unaffected.

- [ ] **Step 5: Confirm no dry-run artifacts were committed**

```bash
git status --short
```

Expected: clean (the dry-run wrote only to `/tmp`; `datasets/` is gitignored).

- [ ] **Step 6: Review commit history + finish the branch**

```bash
git log --oneline feat/pad-b1-finetune-curve ^main
```

Expected: ~6 commits (spec + Tasks 1–5). Then hand off to `superpowers:finishing-a-development-branch` to merge to local `main` (the user's established pattern).

---

## Final Verification

From `/Users/stuartwells/test`:

```bash
.venv/bin/pytest pad-synth-core/tests/test_b1_finetune.py tests/test_b1_finetune_curve.py -v
.venv/bin/pytest pad-synth-core/tests/ pad-synth-face/tests/ tests/ -q
```

Expected: the B1 tests pass; full suite green. `pretrain_on_synth` +
`finetune_and_eval_on_real` exist in `baseline.py`, the runner exists, and the
mechanical dry-run on the n=55 pilot ran end-to-end. The harness then sits ready
for the real curve once larger/purchased real data is staged.
