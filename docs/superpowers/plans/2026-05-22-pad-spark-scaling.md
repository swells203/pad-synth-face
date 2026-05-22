# PAD Spark Scaling Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 3 × 3 × 3 (model capacity × dataset size × seed) factorial PAD experiment on the DGX Spark to disambiguate the Phase 1.5 open question (capacity- / data- / physics-limited), producing a committed decision-matrix-update report. No trained detector is kept.

**Architecture:** Six new generation configs (Sets A/B at three scales) drive the existing tested `pad-synth-face generate` CLI on the laptop; outputs are `rsync`'d to the Spark. A new `models_zoo.py` exposes three model factories (TinyCNN, SmallCNN ~97k params, ResNet18 ~11M params). The existing `train_and_cross_domain_eval` is extended with backwards-compatible `device` and `model_factory` parameters. A new `scripts/spark_sweep.py` runs the 27-cell grid on the GB10 GPU; results are `rsync`'d back and synthesized into a committed report.

**Tech Stack:** Python 3.12 (Spark), 3.13 (laptop), PyTorch nightly with CUDA 12.8 (Blackwell sm_121 support), torchvision, numpy, Pillow, pytest with `--import-mode=importlib`. uv on both hosts. No new packages — work lives in existing `pad-synth-core`, plus new top-level scripts and configs.

---

## Reference: facts the engineer needs

**Hosts.** Laptop: `~/test/` (this repo), Python via `.venv/bin/python`. Spark: `swells@spark-50d2.local`, NVIDIA GB10 (compute capability 12.1, Blackwell sm_121), CUDA 13.0, Python 3.12.3 system, `git` installed, **no uv, no torch** pre-installed. `~/ml/{checkpoints,datasets,logs,projects}` directories already exist.

**Existing fixtures (re-used unchanged at any scale).**
- `datasets/_fixtures/digiface/` — 8 identity dirs (Set A bonafide source).
- `datasets/_fixtures/extended_fixture/` — 16 identity dirs (Set B bonafide source).
- The generator multiplies via `bonafide.samples_per_bonafide` × N_identities × 2 (one bonafide-share + one attack-share, with `print` and `replay` weighted 1:1).

**Existing baseline math (verified).**
- Phase 1 (`phase1_smoke.yaml`): seed 20260511, 8 IDs × `samples_per_bonafide=6` × 2 = 96 samples, in-domain EER 0.29.
- Phase 1.5 (`phase15_setb.yaml`): seed 20260512, 16 IDs × `samples_per_bonafide=4` × 2 = 128 samples, cross-domain EER 0.36 (seed 0) / 0.39 multi-seed mean.

**Existing API to extend.** `pad-synth-core/src/pad_synth_core/eval/baseline.py::train_and_cross_domain_eval(train_root, eval_root=None, epochs=8, batch_size=8, seed=0) -> dict`. Returns keys `eer_in_domain`, `val_accuracy_in_domain`, `n_train`, `n_val_in_domain`, `eer_cross_domain`, `val_accuracy_cross_domain`, `n_val_cross_domain`.

**Spec reference.** `docs/superpowers/specs/2026-05-22-pad-spark-scaling-design.md`. §3 has the grid; §5 has the smoke gate; §7 the artifact schemas.

## File Structure

**Create**

- `configs/runs/spark_seta_d1.yaml`, `spark_seta_d2.yaml`, `spark_seta_d3.yaml` — Set A configs.
- `configs/runs/spark_setb_d1.yaml`, `spark_setb_d2.yaml`, `spark_setb_d3.yaml` — Set B configs.
- `pad-synth-core/src/pad_synth_core/eval/models_zoo.py` — `make_tiny_cnn`, `make_small_cnn`, `make_resnet18`, `FACTORIES` dict.
- `pad-synth-core/tests/test_models_zoo.py` — shape and parameter-count assertions.
- `pad-synth-face/tests/test_spark_configs.py` — load + validate the 6 new configs.
- `pad-synth-core/tests/test_baseline_extensions.py` — backwards-compatibility + device + model_factory tests.
- `scripts/spark_sweep.py` — orchestrator; runs 27 cells; writes JSON/CSV/MD.
- `tests/test_spark_sweep.py` — tiny end-to-end integration test (CPU, 1 cell, 1 epoch).
- `scripts/setup_spark.sh` — one-shot Spark bootstrap (uv install, venv, torch nightly, requirements freeze).
- `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` — populated in Task 10.
- `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/` — directory for rsync'd raw JSON + CSV.

**Modify**

- `pad-synth-core/src/pad_synth_core/eval/baseline.py` — extend `train_and_cross_domain_eval` signature with `device`/`model_factory` keyword args (backwards-compatible defaults).
- `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` — append a one-line update linking the new report.

---

## Task 1: Six sweep configs

**Files:**
- Create: `configs/runs/spark_seta_d1.yaml`, `spark_seta_d2.yaml`, `spark_seta_d3.yaml`
- Create: `configs/runs/spark_setb_d1.yaml`, `spark_setb_d2.yaml`, `spark_setb_d3.yaml`
- Create: `pad-synth-face/tests/test_spark_configs.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_spark_configs.py`:
```python
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "configs" / "runs"

EXPECTED = {
    # filename: (seed, sensor, fixture, samples_per_bonafide)
    "spark_seta_d1.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 6),
    "spark_seta_d2.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 32),
    "spark_seta_d3.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 256),
    "spark_setb_d1.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 4),
    "spark_setb_d2.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 32),
    "spark_setb_d3.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 256),
}


def test_sweep_configs_present_and_well_formed():
    for fname, (seed, sensor, fixture, spb) in EXPECTED.items():
        cfg = yaml.safe_load((CFG_DIR / fname).read_text())
        assert cfg["run"]["seed"] == seed, fname
        assert cfg["run"]["deterministic"] is True, fname
        assert cfg["run"]["output"] == f"./datasets/{Path(fname).stem}", fname
        assert cfg["modality"] == "face", fname
        assert cfg["sensor_preset"] == sensor, fname
        assert cfg["bonafide"]["root"] == fixture, fname
        assert cfg["bonafide"]["samples_per_bonafide"] == spb, fname
        assert set(cfg["attacks"].keys()) == {"print", "replay"}, fname
        assert cfg["attacks"]["print"]["weight"] == 1.0, fname
        assert cfg["attacks"]["replay"]["weight"] == 1.0, fname
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/stuartwells/test && .venv/bin/python -m pytest pad-synth-face/tests/test_spark_configs.py -q 2>&1 | tail -5`
Expected: 1 failed (`FileNotFoundError` on first config file).

- [ ] **Step 3: Create all six configs**

`configs/runs/spark_seta_d1.yaml`:
```yaml
run:
  name: spark_seta_d1
  output: ./datasets/spark_seta_d1
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_fixtures/digiface
  samples_per_bonafide: 6
  splits:
    train: 0.0
    dev: 0.0
    test: 1.0

attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml

sensor_preset: mobile-front-2024
```

`configs/runs/spark_seta_d2.yaml`: identical to `spark_seta_d1.yaml` except `name: spark_seta_d2`, `output: ./datasets/spark_seta_d2`, `samples_per_bonafide: 32`.

`configs/runs/spark_seta_d3.yaml`: identical to `spark_seta_d1.yaml` except `name: spark_seta_d3`, `output: ./datasets/spark_seta_d3`, `samples_per_bonafide: 256`.

`configs/runs/spark_setb_d1.yaml`:
```yaml
run:
  name: spark_setb_d1
  output: ./datasets/spark_setb_d1
  seed: 20260523
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_fixtures/extended_fixture
  samples_per_bonafide: 4
  splits:
    train: 0.0
    dev: 0.0
    test: 1.0

attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml

sensor_preset: webcam-1080p
```

`configs/runs/spark_setb_d2.yaml`: identical to `spark_setb_d1.yaml` except `name: spark_setb_d2`, `output: ./datasets/spark_setb_d2`, `samples_per_bonafide: 32`.

`configs/runs/spark_setb_d3.yaml`: identical to `spark_setb_d1.yaml` except `name: spark_setb_d3`, `output: ./datasets/spark_setb_d3`, `samples_per_bonafide: 256`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_spark_configs.py -q 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: prior count + 1.

- [ ] **Step 6: Commit**

```bash
git add configs/runs/spark_seta_d1.yaml configs/runs/spark_seta_d2.yaml configs/runs/spark_seta_d3.yaml configs/runs/spark_setb_d1.yaml configs/runs/spark_setb_d2.yaml configs/runs/spark_setb_d3.yaml pad-synth-face/tests/test_spark_configs.py
git commit -m "feat(pad-spark): six sweep configs (Set A/B at three scales)"
```

---

## Task 2: Generate the six datasets locally

**Files:** none created or modified (output goes to `datasets/`, which is gitignored).

- [ ] **Step 1: Run all six generations**

```bash
cd /Users/stuartwells/test
for f in spark_seta_d1 spark_seta_d2 spark_seta_d3 spark_setb_d1 spark_setb_d2 spark_setb_d3; do
  .venv/bin/python -m pad_synth_face.cli generate --config configs/runs/${f}.yaml | tail -3
done
```

Expected: each prints a JSON summary with `"generated"` equal to the count below and `"failed": 0`.

- [ ] **Step 2: Verify counts match the scaling math**

```bash
for d in datasets/spark_set{a,b}_d{1,2,3}; do
  n=$(wc -l < "$d/manifest.jsonl")
  bona=$(grep -c '"label":"bonafide"' "$d/manifest.jsonl")
  attack=$(grep -c '"label":"attack"' "$d/manifest.jsonl")
  printf "%-22s total=%5d  bonafide=%5d  attack=%5d\n" "$d" "$n" "$bona" "$attack"
done
```

Expected (exact):
```
datasets/spark_seta_d1   total=   96  bonafide=   48  attack=   48
datasets/spark_seta_d2   total=  512  bonafide=  256  attack=  256
datasets/spark_seta_d3   total= 4096  bonafide= 2048  attack= 2048
datasets/spark_setb_d1   total=  128  bonafide=   64  attack=   64
datasets/spark_setb_d2   total= 1024  bonafide=  512  attack=  512
datasets/spark_setb_d3   total= 8192  bonafide= 4096  attack= 4096
```

If any row's totals don't match: stop and report BLOCKED — the generator's scaling isn't behaving as the spec assumes.

- [ ] **Step 3: No commit**

`datasets/` is gitignored. Nothing to commit. The configs from Task 1 are the regenerable specification; the dataset bytes are deterministic outputs of those configs.

---

## Task 3: `models_zoo.py` — three model factories

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/eval/models_zoo.py`
- Create: `pad-synth-core/tests/test_models_zoo.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-core/tests/test_models_zoo.py`:
```python
import torch

from pad_synth_core.eval.models_zoo import (
    FACTORIES,
    make_resnet18,
    make_small_cnn,
    make_tiny_cnn,
)


def _param_count(m):
    return sum(p.numel() for p in m.parameters())


def test_factories_exposed():
    assert set(FACTORIES.keys()) == {"L1", "L2", "L3"}
    assert FACTORIES["L1"] is make_tiny_cnn
    assert FACTORIES["L2"] is make_small_cnn
    assert FACTORIES["L3"] is make_resnet18


def test_tiny_cnn_shape_and_size():
    m = make_tiny_cnn()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    assert _param_count(m) < 1_000  # the floor — truly tiny


def test_small_cnn_shape_and_size():
    m = make_small_cnn()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    # ~97k params per spec; allow some headroom.
    assert 50_000 < _param_count(m) < 200_000


def test_resnet18_shape_and_size():
    m = make_resnet18()
    out = m(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)
    # ~11M params for torchvision ResNet18 (head replaced).
    assert 10_000_000 < _param_count(m) < 12_000_000
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_models_zoo.py -q 2>&1 | tail -5`
Expected: collection error / `ModuleNotFoundError: No module named 'pad_synth_core.eval.models_zoo'`.

- [ ] **Step 3: Implement the factories**

`pad-synth-core/src/pad_synth_core/eval/models_zoo.py`:
```python
"""Model factories for the PAD scaling experiment.

Each factory returns an nn.Module mapping an RGB image batch (B, 3, H, W)
to logits (B, 2) — bonafide=0, attack=1. The factories are deliberately
small and explicit; they exist solely for the Spark capacity sweep.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def make_tiny_cnn() -> nn.Module:
    """The Phase-1 baseline (kept here for sweep symmetry)."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 2),
    )


def make_small_cnn() -> nn.Module:
    """~97k params; the mid-capacity tier."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 2),
    )


def make_resnet18() -> nn.Module:
    """torchvision ResNet18 from scratch; final fc -> Linear(512, 2)."""
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


FACTORIES = {
    "L1": make_tiny_cnn,
    "L2": make_small_cnn,
    "L3": make_resnet18,
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_models_zoo.py -q 2>&1 | tail -3`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/models_zoo.py pad-synth-core/tests/test_models_zoo.py
git commit -m "feat(pad-eval): models_zoo (TinyCNN / SmallCNN / ResNet18 factories)"
```

---

## Task 4: Extend `train_and_cross_domain_eval` with device + model_factory

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py` (signature + body)
- Create: `pad-synth-core/tests/test_baseline_extensions.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-core/tests/test_baseline_extensions.py`:
```python
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pad_synth_core.eval.baseline import train_and_cross_domain_eval
from pad_synth_core.eval.models_zoo import make_small_cnn


def _build_tiny_dataset(root: Path, n_bonafide: int = 4, n_attack: int = 4) -> Path:
    base = root / "face"
    for label_dir, n in (("bonafide", n_bonafide), ("print", n_attack)):
        d = base / label_dir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{i:04d}.jpg")
    return root


def test_default_signature_is_backwards_compatible(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(train_root=root, epochs=1, seed=0)
    assert set(out.keys()) >= {
        "eer_in_domain", "val_accuracy_in_domain", "n_train",
        "n_val_in_domain", "eer_cross_domain",
    }
    assert out["eer_cross_domain"] is None


def test_model_factory_injection(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, model_factory=make_small_cnn,
    )
    assert isinstance(out["eer_in_domain"], float)


def test_device_cpu_explicit(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, device="cpu",
    )
    assert isinstance(out["eer_in_domain"], float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA available")
def test_device_cuda(tmp_path):
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, device="cuda",
    )
    assert isinstance(out["eer_in_domain"], float)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_baseline_extensions.py -q 2>&1 | tail -8`
Expected: tests for `model_factory=` and `device=` keyword args fail with `TypeError: train_and_cross_domain_eval() got an unexpected keyword argument ...`. The backwards-compat default test should still pass.

- [ ] **Step 3: Extend `train_and_cross_domain_eval`**

In `pad-synth-core/src/pad_synth_core/eval/baseline.py`, modify the imports block to add `Callable`:

```python
from typing import Any, Callable
```

Replace the existing `train_and_cross_domain_eval` function (lines 101–162 currently) with this version:

```python
def train_and_cross_domain_eval(
    train_root: Path,
    eval_root: Path | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict[str, Any]:
    """Train on train_root; eval in-domain (held-out 25 percent split) and
    optionally cross-domain (full eval_root if provided).

    Defaults preserve the Phase 1/1.5 behavior (TinyCNN on CPU). Pass
    `device="cuda"` and `model_factory=make_small_cnn` etc. for the
    Spark scaling sweep.

    Returns the same dict shape as before, with all numeric fields finite.
    """
    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")

    train_ds_full = TinyPADDataset(train_root)
    n_val = max(1, len(train_ds_full) // 4)
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

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
    in_eer, in_acc = _eval_loader(model, val_dl, dev)

    cross_eer: float | None = None
    cross_acc: float | None = None
    n_val_cross: int | None = None
    if eval_root is not None:
        cross_ds = TinyPADDataset(eval_root)
        cross_dl = DataLoader(cross_ds, batch_size=batch_size)
        cross_eer, cross_acc = _eval_loader(model, cross_dl, dev)
        n_val_cross = len(cross_ds)

    return {
        "eer_in_domain": in_eer,
        "val_accuracy_in_domain": in_acc,
        "n_train": n_train,
        "n_val_in_domain": n_val,
        "eer_cross_domain": cross_eer,
        "val_accuracy_cross_domain": cross_acc,
        "n_val_cross_domain": n_val_cross,
    }
```

Also extend `_eval_loader` to accept the device (move batches inside the loop). Replace the existing `_eval_loader` (lines 83–98) with:

```python
def _eval_loader(
    model: nn.Module, dl: DataLoader, device: torch.device | None = None
) -> tuple[float, float]:
    """Run a model over a dataloader; return (EER, accuracy)."""
    dev = device or torch.device("cpu")
    scores: list[float] = []
    labels: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            scores.extend(probs)
            labels.extend(y.cpu().tolist())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.numel())
    return compute_eer(scores, labels), correct / max(total, 1)
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_baseline_extensions.py pad-synth-core/tests/test_eval_baseline.py -q 2>&1 | tail -5`
Expected: all pass (including the existing `test_eval_baseline.py` tests — backwards compat).

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: prior count + the new tests added in Tasks 3 & 4 (3 from `test_models_zoo.py` after the smoke `test_factories_exposed` and the 3 model tests = 4 total there, plus 4 from `test_baseline_extensions.py` minus the CUDA one which skips locally = 3 ran, 1 skipped).

- [ ] **Step 6: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_baseline_extensions.py
git commit -m "feat(pad-eval): train_and_cross_domain_eval gains device + model_factory"
```

---

## Task 5: `scripts/spark_sweep.py` — the orchestrator

**Files:**
- Create: `scripts/spark_sweep.py`
- Create: `tests/test_spark_sweep.py`

- [ ] **Step 1: Write the failing test**

`tests/test_spark_sweep.py`:
```python
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _seed_dataset(root: Path, n_bonafide: int, n_attack: int) -> Path:
    base = root / "face"
    for label_dir, n in (("bonafide", n_bonafide), ("print", n_attack)):
        d = base / label_dir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{i:04d}.jpg")
    return root


def test_one_cell_end_to_end_cpu(tmp_path):
    set_a = _seed_dataset(tmp_path / "set_a", n_bonafide=12, n_attack=12)
    set_b = _seed_dataset(tmp_path / "set_b", n_bonafide=12, n_attack=12)
    out_dir = tmp_path / "out"

    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "spark_sweep.py"),
            "--set-a-d1", str(set_a), "--set-b-d1", str(set_b),
            "--set-a-d2", str(set_a), "--set-b-d2", str(set_b),
            "--set-a-d3", str(set_a), "--set-b-d3", str(set_b),
            "--output-dir", str(out_dir),
            "--device", "cpu",
            "--epochs", "1",
            "--batch-size", "4",
            "--cells", "L1:D1:0",
        ],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr

    runs = list((out_dir / "runs").glob("*.json"))
    assert len(runs) == 1
    rec = json.loads(runs[0].read_text())
    assert rec["capacity"] == "L1"
    assert rec["data_level"] == "D1"
    assert rec["seed"] == 0
    for k in ("eer_in_domain", "eer_cross_domain", "train_seconds",
              "git_sha", "torch_version", "device"):
        assert k in rec
    assert 0.0 <= rec["eer_in_domain"] <= 1.0

    summary = list(csv.DictReader((out_dir / "summary.csv").open()))
    assert len(summary) == 1
    assert summary[0]["capacity"] == "L1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_spark_sweep.py -q 2>&1 | tail -5`
Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement the orchestrator**

`scripts/spark_sweep.py`:
```python
"""Run the PAD capacity-x-data factorial sweep.

Cells: capacity L in {L1,L2,L3} x data level D in {D1,D2,D3} x seed in {0,1,2}.
Per cell, train on Set A at the matching D, eval cross-domain on Set B at the
matching D. Writes per-cell JSON to <out>/runs/<L>_<D>_<seed>.json plus a
summary.csv across all completed cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_core.eval.baseline import train_and_cross_domain_eval  # noqa: E402
from pad_synth_core.eval.models_zoo import FACTORIES  # noqa: E402

DATA_LEVELS = ("D1", "D2", "D3")
CAPACITIES = ("L1", "L2", "L3")
SEEDS = (0, 1, 2)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _parse_cells(spec: str | None) -> list[tuple[str, str, int]]:
    if not spec:
        return [(L, D, s) for L in CAPACITIES for D in DATA_LEVELS for s in SEEDS]
    cells = []
    for tok in spec.split(","):
        L, D, s = tok.strip().split(":")
        cells.append((L, D, int(s)))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    for L in ("a", "b"):
        for D in DATA_LEVELS:
            ap.add_argument(f"--set-{L}-{D.lower()}", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--cells",
        default=None,
        help="Comma-separated L:D:seed (e.g. 'L1:D1:0,L2:D2:1'); default = all 27.",
    )
    args = ap.parse_args()

    set_roots = {
        ("a", D): getattr(args, f"set_a_{D.lower()}") for D in DATA_LEVELS
    } | {
        ("b", D): getattr(args, f"set_b_{D.lower()}") for D in DATA_LEVELS
    }

    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha()
    torch_v = torch.__version__
    cuda_v = torch.version.cuda or "none"

    cells = _parse_cells(args.cells)
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["capacity", "data_level", "seed", "eer_in_domain",
                    "eer_cross_domain", "train_seconds"])

    for L, D, seed in cells:
        train_root = set_roots[("a", D)]
        eval_root = set_roots[("b", D)]
        t0 = time.time()
        out = train_and_cross_domain_eval(
            train_root=train_root,
            eval_root=eval_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
            model_factory=FACTORIES[L],
        )
        elapsed = time.time() - t0
        rec = {
            "capacity": L,
            "data_level": D,
            "seed": seed,
            "n_train": out["n_train"],
            "n_val_in_domain": out["n_val_in_domain"],
            "n_val_cross_domain": out["n_val_cross_domain"],
            "eer_in_domain": float(out["eer_in_domain"]),
            "eer_cross_domain": (
                float(out["eer_cross_domain"])
                if out["eer_cross_domain"] is not None
                else None
            ),
            "val_accuracy_in_domain": float(out["val_accuracy_in_domain"]),
            "val_accuracy_cross_domain": (
                float(out["val_accuracy_cross_domain"])
                if out["val_accuracy_cross_domain"] is not None
                else None
            ),
            "train_seconds": elapsed,
            "git_sha": git_sha,
            "torch_version": torch_v,
            "cuda_version": cuda_v,
            "device": args.device,
            "train_root": str(train_root),
            "eval_root": str(eval_root),
        }
        out_path = runs_dir / f"{L}_{D}_{seed}.json"
        out_path.write_text(json.dumps(rec, indent=2))
        with summary_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([
                L, D, seed, rec["eer_in_domain"], rec["eer_cross_domain"],
                f"{elapsed:.2f}",
            ])
        ec = rec["eer_cross_domain"]
        ec_s = f"{ec:.3f}" if ec is not None else "N/A"
        print(
            f"{L} {D} seed={seed}  eer_in={rec['eer_in_domain']:.3f}"
            f"  eer_cross={ec_s}  {elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the integration test passes**

Run: `.venv/bin/python -m pytest tests/test_spark_sweep.py -q 2>&1 | tail -5`
Expected: 1 passed (takes ~20–60 s on CPU for the 1 cell × 1 epoch).

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: previous count + 1.

- [ ] **Step 6: Commit**

```bash
git add scripts/spark_sweep.py tests/test_spark_sweep.py
git commit -m "feat(pad-spark): spark_sweep.py orchestrator + integration test"
```

---

## Task 6: `scripts/setup_spark.sh` — Spark bootstrap

**Files:**
- Create: `scripts/setup_spark.sh`
- Create: `tests/test_setup_spark_syntax.py`

- [ ] **Step 1: Write the failing test**

`tests/test_setup_spark_syntax.py`:
```python
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_setup_spark_script_is_syntactically_valid():
    script = REPO / "scripts" / "setup_spark.sh"
    assert script.exists(), "setup_spark.sh must exist"
    bash = shutil.which("bash")
    assert bash, "bash not on PATH"
    r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_setup_spark_is_executable():
    script = REPO / "scripts" / "setup_spark.sh"
    assert script.stat().st_mode & 0o111, "setup_spark.sh must have executable bit set"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_setup_spark_syntax.py -q 2>&1 | tail -3`
Expected: 2 failed (file doesn't exist).

- [ ] **Step 3: Create the script**

`scripts/setup_spark.sh`:
```bash
#!/usr/bin/env bash
# Bootstrap the DGX Spark for the PAD scaling sweep.
# Idempotent: safe to re-run. Assumes ~/ml/projects/pad-spark/ is the
# project checkout (created by the caller via git clone / rsync).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/ml/projects/pad-spark}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: PROJECT_DIR=$PROJECT_DIR does not exist." >&2
  echo "  Push the repo to the Spark first (git clone / rsync)." >&2
  exit 2
fi

cd "$PROJECT_DIR"

# 1) uv (skip if already installed)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer adds ~/.local/bin to PATH for new shells; export for this one.
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Venv (skip if already created)
if [[ ! -d ".venv" ]]; then
  uv venv --python 3.12
fi

# 3) Install PyTorch nightly with CUDA 12.8 (Blackwell sm_121 support)
#    + the small runtime dep set the sweep needs.
uv pip install --upgrade \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  torch torchvision

uv pip install --upgrade numpy pillow pyyaml pytest

# 4) Freeze the resolved versions for reproducibility (one-shot snapshot;
#    re-runs overwrite to reflect the latest nightly).
uv pip freeze > requirements.spark.txt

# 5) Sanity-check: torch sees CUDA + the GPU is the expected GB10.
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available after install"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
PY

echo "Spark setup OK."
```

Mark executable:
```bash
chmod +x scripts/setup_spark.sh
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_setup_spark_syntax.py -q 2>&1 | tail -3`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_spark.sh tests/test_setup_spark_syntax.py
git commit -m "feat(pad-spark): Spark bootstrap script (uv + torch nightly cu128)"
```

---

## Task 7: Provision the Spark (operational — SSH)

**Files:** none (operations on a remote host).

- [ ] **Step 1: Sync the repo to the Spark**

From the laptop:
```bash
ssh swells@spark-50d2.local 'mkdir -p ~/ml/projects/pad-spark'
rsync -av --delete \
  --exclude='.venv' --exclude='__pycache__' --exclude='datasets' \
  --exclude='.superpowers' --exclude='.git/objects/pack' \
  /Users/stuartwells/test/ \
  swells@spark-50d2.local:~/ml/projects/pad-spark/
```

Expected: rsync completes; final size on Spark < 50 MB.

- [ ] **Step 2: Run the bootstrap script**

```bash
ssh swells@spark-50d2.local 'bash ~/ml/projects/pad-spark/scripts/setup_spark.sh' 2>&1 | tail -20
```

Expected: ends with `Spark setup OK.` and a line like `device: NVIDIA GB10`.
If it fails with a torch wheel resolution error, retry once; if persistent, report BLOCKED with the exact error.

- [ ] **Step 3: Run the PAD package tests on the Spark to confirm the env works**

```bash
ssh swells@spark-50d2.local 'cd ~/ml/projects/pad-spark && .venv/bin/python -m pytest pad-synth-core pad-synth-face tests -q 2>&1 | tail -5'
```

Expected: all PAD + root tests pass. The `defid` tests may also run; that's fine.

- [ ] **Step 4: No commit (operational)**

This task is operational. The `requirements.spark.txt` produced by the bootstrap script lives on the Spark only (not committed — nightly versions change daily; we re-bootstrap when we run the experiment, and the report records the exact pinned versions used).

---

## Task 8: Sync datasets to Spark and run the smoke cell

**Files:** none (operations).

- [ ] **Step 1: rsync the six datasets to the Spark**

```bash
rsync -av --partial \
  /Users/stuartwells/test/datasets/spark_set{a,b}_d{1,2,3}/ \
  swells@spark-50d2.local:~/ml/datasets/  # (the trailing slashes flatten; see below)
```

Use this exact loop instead (safer):
```bash
for d in spark_seta_d1 spark_seta_d2 spark_seta_d3 spark_setb_d1 spark_setb_d2 spark_setb_d3; do
  rsync -av --partial \
    "/Users/stuartwells/test/datasets/${d}/" \
    "swells@spark-50d2.local:~/ml/datasets/${d}/"
done
```

Expected: each directory transfers without error; final remote sizes match local.

- [ ] **Step 2: Run the smoke cell (L1, D1, seed=0)**

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
ssh swells@spark-50d2.local "cd ~/ml/projects/pad-spark && .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 ~/ml/datasets/spark_seta_d1 --set-b-d1 ~/ml/datasets/spark_setb_d1 \
  --set-a-d2 ~/ml/datasets/spark_seta_d2 --set-b-d2 ~/ml/datasets/spark_setb_d2 \
  --set-a-d3 ~/ml/datasets/spark_seta_d3 --set-b-d3 ~/ml/datasets/spark_setb_d3 \
  --output-dir ~/ml/logs/pad-spark/${ts} \
  --device cuda --epochs 10 --batch-size 32 \
  --cells L1:D1:0"
```

Expected: prints one line like `L1 D1 seed=0  eer_in=0.27  eer_cross=0.36  35.4s`.

- [ ] **Step 3: Verify the smoke gate**

```bash
ssh swells@spark-50d2.local "cat ~/ml/logs/pad-spark/${ts}/runs/L1_D1_0.json" | tee /tmp/smoke.json
.venv/bin/python -c "
import json, sys
r = json.load(open('/tmp/smoke.json'))
eer = r['eer_cross_domain']
ok = 0.33 <= eer <= 0.39
print('cross-domain EER =', eer, 'gate', '[0.33, 0.39] ->', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
"
```

Expected: exit code 0, "PASS". If FAIL: stop and diagnose (torch/CUDA determinism, dataset integrity, sensor preset). Do NOT proceed to Task 9.

- [ ] **Step 4: No commit yet (commit after the full grid runs in Task 10)**

---

## Task 9: Run the full grid (operational — Spark)

**Files:** none (operations).

- [ ] **Step 1: Run the remaining 26 cells**

```bash
# Reuse the same ${ts} from Task 8 so the smoke and full results live together.
ssh swells@spark-50d2.local "cd ~/ml/projects/pad-spark && .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 ~/ml/datasets/spark_seta_d1 --set-b-d1 ~/ml/datasets/spark_setb_d1 \
  --set-a-d2 ~/ml/datasets/spark_seta_d2 --set-b-d2 ~/ml/datasets/spark_setb_d2 \
  --set-a-d3 ~/ml/datasets/spark_seta_d3 --set-b-d3 ~/ml/datasets/spark_setb_d3 \
  --output-dir ~/ml/logs/pad-spark/${ts} \
  --device cuda --epochs 10 --batch-size 32 \
  --cells L1:D1:1,L1:D1:2,L1:D2:0,L1:D2:1,L1:D2:2,L1:D3:0,L1:D3:1,L1:D3:2,L2:D1:0,L2:D1:1,L2:D1:2,L2:D2:0,L2:D2:1,L2:D2:2,L2:D3:0,L2:D3:1,L2:D3:2,L3:D1:0,L3:D1:1,L3:D1:2,L3:D2:0,L3:D2:1,L3:D2:2,L3:D3:0,L3:D3:1,L3:D3:2"
```

Expected: 26 lines like `L? D? seed=?  eer_in=0.??  eer_cross=0.??  ??.?s`. ResNet18 on D3 will dominate wall-time (a few minutes per cell); the whole sweep should finish in well under 1 hour. If any cell errors, the runner stops on that cell — diagnose and re-run with `--cells` listing only the unfinished cells.

- [ ] **Step 2: Confirm all 27 runs present**

```bash
ssh swells@spark-50d2.local "ls ~/ml/logs/pad-spark/${ts}/runs/ | wc -l"
```

Expected: `27`.

- [ ] **Step 3: rsync results back to the laptop**

```bash
mkdir -p docs/superpowers/reports/2026-05-22-pad-spark-sweep-results
rsync -av "swells@spark-50d2.local:~/ml/logs/pad-spark/${ts}/" \
  docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/
```

Expected: `runs/` (27 files) + `summary.csv` arrive locally.

- [ ] **Step 4: No commit yet (the report in Task 10 commits everything together)**

---

## Task 10: Author the report; commit results + report

**Files:**
- Create: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`
- Modify: `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` (one-line append)

- [ ] **Step 1: Aggregate results into the heatmaps**

Run this one-shot Python to compute the means and stds (do not commit this script — it's a one-off used by you, the implementer, while drafting):

```bash
.venv/bin/python - <<'PY'
import json, csv, statistics as st
from pathlib import Path
runs = sorted(Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs").glob("*.json"))
rows = [json.loads(p.read_text()) for p in runs]
print(f"Got {len(rows)} cells")
cells = {}
for r in rows:
    cells.setdefault((r["capacity"], r["data_level"]), []).append(r)
for (L, D), group in sorted(cells.items()):
    eer_in = [g["eer_in_domain"] for g in group]
    eer_cr = [g["eer_cross_domain"] for g in group]
    secs = [g["train_seconds"] for g in group]
    print(
        f"{L} {D}  n={len(group)}  "
        f"in: {st.mean(eer_in):.3f} +- {st.stdev(eer_in) if len(eer_in)>1 else 0:.3f}  "
        f"cross: {st.mean(eer_cr):.3f} +- {st.stdev(eer_cr) if len(eer_cr)>1 else 0:.3f}  "
        f"secs(median): {sorted(secs)[len(secs)//2]:.1f}"
    )
PY
```

Use the printed values to fill in the heatmaps in Step 2.

- [ ] **Step 2: Write the report**

`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`:
```markdown
# PAD Spark Scaling Sweep — Results

**Date:** 2026-05-22
**Spec:** [`../specs/2026-05-22-pad-spark-scaling-design.md`](../specs/2026-05-22-pad-spark-scaling-design.md)
**Plan:** [`../plans/2026-05-22-pad-spark-scaling.md`](../plans/2026-05-22-pad-spark-scaling.md)
**Hardware:** NVIDIA GB10 (DGX Spark), CUDA 13.0
**Torch:** <fill from any run's torch_version field>
**Git SHA:** <fill from any run's git_sha field>
**Cells:** 9 (capacity × data) × 3 seeds = 27

## Cross-domain EER (mean ± std across 3 seeds)

|       | D1 (current) | D2 (~8×) | D3 (~64×) |
|-------|--------------|----------|-----------|
| **L1 (TinyCNN)**  | <fill> | <fill> | <fill> |
| **L2 (SmallCNN)** | <fill> | <fill> | <fill> |
| **L3 (ResNet18)** | <fill> | <fill> | <fill> |

## In-domain EER (mean ± std across 3 seeds)

|       | D1 | D2 | D3 |
|-------|----|----|----|
| **L1** | <fill> | <fill> | <fill> |
| **L2** | <fill> | <fill> | <fill> |
| **L3** | <fill> | <fill> | <fill> |

## Median training time per cell (seconds)

|       | D1 | D2 | D3 |
|-------|----|----|----|
| **L1** | <fill> | <fill> | <fill> |
| **L2** | <fill> | <fill> | <fill> |
| **L3** | <fill> | <fill> | <fill> |

## Diagnosis

Smoke cell (L1·D1·seed=0) cross-domain EER: <fill> (gate: [0.33, 0.39]) — <PASS/FAIL>.

**Capacity-axis effect** (L3 − L1 at fixed D, mean across seeds and D levels):
ΔEER_cross = <fill>. Threshold for "drops along the axis": ≥ 0.05 with non-overlapping ±1σ bands. Verdict: <axis fires / flat>.

**Data-axis effect** (D3 − D1 at fixed L, mean across seeds and L tiers):
ΔEER_cross = <fill>. Verdict: <axis fires / flat>.

**Overall diagnosis:** <capacity-limited / data-limited / both / physics-limited>.

## Recommendation update for Phase 2

<2–4 sentences. Reference the decisions/roadmap doc and update the "hybrid Phase 2" recommendation accordingly.>

## Raw results

- Per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs/`](./2026-05-22-pad-spark-sweep-results/runs/)
- Summary CSV: [`./2026-05-22-pad-spark-sweep-results/summary.csv`](./2026-05-22-pad-spark-sweep-results/summary.csv)
```

Fill in every `<fill>` from the Step 1 output. The diagnosis verdicts come from the spec §2 quantitative threshold (≥ 0.05 EER drop along an axis with non-overlapping ±1σ bands; otherwise flat).

- [ ] **Step 3: Append a one-line update to the decisions/roadmap doc**

In `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md`, append at the end of the file:

```markdown

---

## 2026-05-22 update — Spark scaling sweep

The Phase 1.5 open question (capacity- / data- / physics-limited) has been disambiguated by a 3×3×3 sweep on the DGX Spark. **Diagnosis: <capacity-limited / data-limited / both / physics-limited>.** See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) for the heatmaps and the updated Phase 2 recommendation.
```

Fill in the diagnosis verbatim from the report.

- [ ] **Step 4: Commit results + report together**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/ \
        docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md
git commit -m "report(pad-spark): scaling sweep results + decisions-roadmap update"
```

- [ ] **Step 5: Full suite green (final)**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: all pass.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose / decision-matrix-update deliverable | Task 10 (report + roadmap update) |
| §2 question + quantitative threshold (≥0.05 with non-overlapping bands) | Task 10 Step 2 (diagnosis section) |
| §3.1 factorial grid (3×3×3) | Task 5 (`spark_sweep.py` enumerates), Task 9 (runs) |
| §3.1 L1/L2/L3 architectures | Task 3 (`models_zoo.py`) + Task 3 tests |
| §3.1 D1/D2/D3 sample counts | Task 1 (configs) + Task 2 (verification) |
| §3.2 train/eval framing (in-domain + cross-domain) | Task 4 (extended baseline already returns both) + Task 5 (records both in JSON) |
| §3.3 GPU placement, deterministic mode | Task 4 (device kw arg) + Task 7 Step 2 (torch+CUDA verified) |
| §4 deterministic local generation, rsync to Spark | Task 1 (configs), Task 2 (generation), Task 8 Step 1 (rsync) |
| §4 D1 matches phase{1,1.5}_smoke counts (96/128) | Task 2 Step 2 (assert counts), Task 8 Step 3 (cross-domain EER gate) |
| §5 smoke gate [0.33, 0.39] | Task 8 Step 3 (explicit gate test) |
| §6 Spark workflow (uv, torch nightly, layout, outputs) | Task 6 (`setup_spark.sh`), Task 7 (operational) |
| §7.1 per-cell JSON schema | Task 5 (`spark_sweep.py` writes exactly these keys) |
| §7.2 summary CSV columns | Task 5 (CSV writer) |
| §7.3 summary report (heatmaps + diagnosis + recommendation) | Task 10 |
| §8 non-goals (no checkpoint saving, no model integration, no hyperparam sweep) | Honored: nothing in this plan saves checkpoints or integrates models into the CLI |
| §9 success criteria | Task 10 Step 4 (commit) + Task 10 Step 5 (suite green) |

**Placeholder scan:** Every `<fill>` in the plan is inside the *report template* that the implementer populates from real run data — they are explicitly *expected to be filled* by Task 10 Step 1's output, not implementation placeholders. No "TBD/TODO/implement later" anywhere else. All code blocks complete.

**Type consistency:** `device: str | None`, `model_factory: Callable[[], nn.Module] | None`, `FACTORIES: dict[str, callable]` consistent across Tasks 3, 4, 5. JSON keys (`capacity`, `data_level`, `seed`, `eer_in_domain`, `eer_cross_domain`, `train_seconds`, `git_sha`, `torch_version`, `device`) consistent between Task 5 (writer) and Task 5 test (asserts subset present) and Task 10 (reader).
