# PAD B2 Pretrained-Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `make_resnet18_pretrained` (ImageNet weights, fc → Linear(512, 2)) to `FACTORIES["L4"]`, then run the focused 18-cell sweep on the Spark and append the L4 column to the existing 224 report.

**Architecture:** One new factory function (5 lines) reusing the same head-swap pattern as the existing `make_resnet18`. The only delta from L3 is the `weights=ResNet18_Weights.IMAGENET1K_V1` constructor flag. Tests verify the registry entry, forward shape, and one end-to-end smoke through `train_and_cross_domain_eval`. The 2026-05-29 spike already proved one cell (cross-domain EER 0.045 vs 0.291 from-scratch); this plan generalises it to a full sweep and writes the result up.

**Tech Stack:** Python 3.12+, PyTorch, torchvision (already in deps via `make_resnet18`), pytest. ssh + the existing `scripts/spark_sweep.py` (with `--cells` filter; no script changes).

**Spec:** [`../specs/2026-05-30-pad-b2-pretrained-backbone-design.md`](../specs/2026-05-30-pad-b2-pretrained-backbone-design.md)

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `pad-synth-core/src/pad_synth_core/eval/models_zoo.py` | Add `make_resnet18_pretrained()`; add `"L4"` to `FACTORIES` | Modify |
| `pad-synth-core/tests/test_models_zoo.py` | Update `test_factories_exposed` to include L4; add L4 unit tests | Modify |
| `pad-synth-core/tests/test_baseline_extensions.py` | One-cell pretrained smoke through `train_and_cross_domain_eval` (skip-if-no-network) | Modify |
| `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` | Append a "2026-05-30 update — B2 (pretrained backbone)" section after the 2026-05-29 A1 section | Modify |
| `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/` + `runs_mix_224_L4/` | Per-cell JSON (9 each) + summary.csv from the sweep | Create (Task 3 output) |

---

## Task 1: Add `make_resnet18_pretrained` factory + tests

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/models_zoo.py`
- Modify: `pad-synth-core/tests/test_models_zoo.py`

- [ ] **Step 1: Write the failing tests**

Open `pad-synth-core/tests/test_models_zoo.py`. The file currently asserts `set(FACTORIES.keys()) == {"L1", "L2", "L3"}` in `test_factories_exposed`. Update that assertion to include `"L4"`, then append two new tests for L4. The full set of edits:

Find:
```python
def test_factories_exposed():
    assert set(FACTORIES.keys()) == {"L1", "L2", "L3"}
    assert FACTORIES["L1"] is make_tiny_cnn
    assert FACTORIES["L2"] is make_small_cnn
    assert FACTORIES["L3"] is make_resnet18
```
Replace with:
```python
def test_factories_exposed():
    assert set(FACTORIES.keys()) == {"L1", "L2", "L3", "L4"}
    assert FACTORIES["L1"] is make_tiny_cnn
    assert FACTORIES["L2"] is make_small_cnn
    assert FACTORIES["L3"] is make_resnet18
    assert FACTORIES["L4"] is make_resnet18_pretrained
```
Also update the imports at the top of the file from:
```python
from pad_synth_core.eval.models_zoo import (
    FACTORIES,
    make_resnet18,
    make_small_cnn,
    make_tiny_cnn,
)
```
to:
```python
from pad_synth_core.eval.models_zoo import (
    FACTORIES,
    make_resnet18,
    make_resnet18_pretrained,
    make_small_cnn,
    make_tiny_cnn,
)
```

Then append at the END of the test file:

```python
def test_l4_pretrained_forward_returns_2logits():
    """L4 must accept the canonical 224x224 input and produce (B, 2) logits."""
    m = make_resnet18_pretrained()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_l4_shares_resnet18_param_count():
    """L4 has the same total parameter count as L3 (only the weight INIT differs)."""
    l3 = make_resnet18()
    l4 = make_resnet18_pretrained()
    assert _param_count(l3) == _param_count(l4)
    # Sanity: ~11M params (ResNet18 with 2-class head).
    assert 11_000_000 < _param_count(l4) < 12_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_models_zoo.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_resnet18_pretrained' from 'pad_synth_core.eval.models_zoo'`.

- [ ] **Step 3: Add the factory + the L4 registry entry**

In `pad-synth-core/src/pad_synth_core/eval/models_zoo.py`:

(a) Update the top import from:
```python
from torchvision.models import resnet18
```
to:
```python
from torchvision.models import ResNet18_Weights, resnet18
```

(b) Add this function immediately after the existing `make_resnet18`:

```python
def make_resnet18_pretrained() -> nn.Module:
    """ResNet18 with ImageNet-pretrained weights; final fc -> Linear(512, 2).
    Same head-swap pattern as `make_resnet18`; only the weight init differs."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m
```

(c) Extend the `FACTORIES` dict at the bottom of the file from:
```python
FACTORIES = {
    "L1": make_tiny_cnn,
    "L2": make_small_cnn,
    "L3": make_resnet18,
}
```
to:
```python
FACTORIES = {
    "L1": make_tiny_cnn,
    "L2": make_small_cnn,
    "L3": make_resnet18,
    "L4": make_resnet18_pretrained,
}
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_models_zoo.py -v`
Expected: all PASS — the 3 existing L1/L2/L3 tests + the updated `test_factories_exposed` + the 2 new L4 tests.

Note: the first call to `make_resnet18_pretrained()` downloads `resnet18-f37072fd.pth` (~45 MB) into `~/.cache/torch/hub/checkpoints/`. The Spark already has it cached from the 2026-05-29 spike. Locally this is a one-time download; subsequent test runs hit the cache.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/models_zoo.py pad-synth-core/tests/test_models_zoo.py
git commit -m "feat(pad-core/eval): make_resnet18_pretrained -> FACTORIES[L4] (ImageNet weights)"
```

---

## Task 2: Pretrained smoke through `train_and_cross_domain_eval`

A one-cell end-to-end smoke confirming `model_factory=make_resnet18_pretrained` runs through the existing training/eval path and returns finite metrics. Skipped cleanly when the pretrained weights can't be downloaded (network-gated).

**Files:**
- Modify: `pad-synth-core/tests/test_baseline_extensions.py`

- [ ] **Step 1: Write the failing test**

Append to `pad-synth-core/tests/test_baseline_extensions.py` (the file already imports `Path`, `np`, `pytest`, `torch`, `Image`, `IMAGE_SHAPE`, `train_and_cross_domain_eval`, `make_small_cnn`, and defines `_build_tiny_dataset`):

```python
from pad_synth_core.eval.models_zoo import make_resnet18_pretrained


@pytest.fixture(scope="module")
def _pretrained_available():
    """Skip the pretrained-smoke test if the weights download fails (e.g. no
    network in CI). Warms the cache once per session."""
    try:
        make_resnet18_pretrained()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pretrained weights unavailable: {e}")


def test_pretrained_resnet18_factory_runs_through_train_and_eval(
    _pretrained_available, tmp_path,
):
    """One-cell smoke: pretrained ResNet18 trained on a tiny fixture dataset,
    1 epoch on CPU, returns finite EER. Locks the path that the Spark sweep
    will exercise at scale."""
    root = _build_tiny_dataset(tmp_path / "t")
    out = train_and_cross_domain_eval(
        train_root=root, epochs=1, seed=0, device="cpu",
        model_factory=make_resnet18_pretrained,
    )
    assert isinstance(out["eer_in_domain"], float)
    assert 0.0 <= out["eer_in_domain"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails (or skips)**

Run: `.venv/bin/python -m pytest "pad-synth-core/tests/test_baseline_extensions.py::test_pretrained_resnet18_factory_runs_through_train_and_eval" -v`

Expected: PASS if `make_resnet18_pretrained` is already implemented (after Task 1) and the cache has the weights (true on this laptop after Task 1's test run); otherwise SKIPPED with a clear "pretrained weights unavailable" message.

(There's no traditional "fails first then passes" red-green cycle here because Task 1 already added the factory and `pytest` will resolve the import. The TDD intent of this task is to lock the integration path with a smoke test that explicitly exercises pretrained-ResNet18 through `train_and_cross_domain_eval`, distinct from the unit tests in Task 1 that only construct the module.)

- [ ] **Step 3: Commit**

```bash
git add pad-synth-core/tests/test_baseline_extensions.py
git commit -m "test(pad-core/eval): pretrained ResNet18 smoke through train_and_cross_domain_eval"
```

---

## Task 3: Run the 18-cell L4 sweep on the Spark + append report

The factory is in place. Sync code to the Spark, run the two 9-cell sweeps (mask and mix), pull results back, compute the mean±std + comparison-vs-L3 tables, append the report section, commit + push.

**Files:**
- Create: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/runs/*.json` + `summary.csv` (9 cells)
- Create: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/runs/*.json` + `summary.csv` (9 cells)
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (append new section after the 2026-05-29 A1 section)

- [ ] **Step 1: Sync the updated code to the Spark**

```bash
rsync -az --exclude='.venv' --exclude='.git' --exclude='datasets' --exclude='__pycache__' --exclude='*.pyc' \
  ./ swells@spark-50d2.local:~/ml/projects/pad-spark/
```
Expected: completes with no errors. The Spark already has the 12 mask+mix datasets and the digiface_224 dir from the 2026-05-29 A1 sweep — no dataset re-sync needed.

- [ ] **Step 2: Smoke-check the factory works on the Spark before launching the sweep**

```bash
ssh -o BatchMode=yes swells@spark-50d2.local '
cd ~/ml/projects/pad-spark
.venv/bin/python -c "
import sys; sys.path.insert(0, \"pad-synth-core/src\")
from pad_synth_core.eval.models_zoo import FACTORIES
m = FACTORIES[\"L4\"]()
print(\"L4 loaded; params=\", sum(p.numel() for p in m.parameters()))
"
'
```
Expected: prints `L4 loaded; params= 11177538`. The Spark already has the cached ResNet18 weights from the 2026-05-29 spike, so this is fast.

- [ ] **Step 3: Launch the mask-only L4 sweep in background on the Spark**

```bash
ssh -o BatchMode=yes swells@spark-50d2.local '
cd ~/ml/projects/pad-spark
CELLS=$(python3 -c "print(\",\".join(f\"L4:{D}:{s}\" for D in (\"D1\",\"D2\",\"D3\") for s in (0,1,2)))")
mkdir -p docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4
rm -f docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/summary.csv
nohup .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mask_seta_d1 --set-b-d1 datasets/mask_setb_d1 \
  --set-a-d2 datasets/mask_seta_d2 --set-b-d2 datasets/mask_setb_d2 \
  --set-a-d3 datasets/mask_seta_d3 --set-b-d3 datasets/mask_setb_d3 \
  --set-a-d4 datasets/mask_seta_d3 --set-b-d4 datasets/mask_setb_d3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4 \
  --cells "$CELLS" --device cuda > /tmp/sweep_mask_224_L4.log 2>&1 &
echo "MASK_L4_PID: $!"
'
```
Capture the printed PID for use in Step 4.

- [ ] **Step 4: Poll mask sweep until done (9 cells)**

Poll progress in 90-second intervals up to ~7.5 minutes (the mask sweep at 224 took 7.5 min for 27 cells on the 2026-05-29 run; 9 cells should be ~3 min):
```bash
for i in 1 2 3 4 5; do
  sleep 90
  echo "=== poll $i ==="
  ssh -o BatchMode=yes swells@spark-50d2.local '
    SUM=~/ml/projects/pad-spark/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/summary.csv
    [ -f "$SUM" ] && n=$(($(wc -l < "$SUM") - 1)) || n=0
    echo "cells_done=$n/9"
    [ $n -ge 9 ] && echo "DONE" && exit 0 || echo still-running
  '
done
```
Expected: cells_done reaches 9 within ~3-5 min. Do NOT proceed to Step 5 until "DONE" is printed.

- [ ] **Step 5: Launch the integrated (mix) L4 sweep + poll**

```bash
ssh -o BatchMode=yes swells@spark-50d2.local '
cd ~/ml/projects/pad-spark
CELLS=$(python3 -c "print(\",\".join(f\"L4:{D}:{s}\" for D in (\"D1\",\"D2\",\"D3\") for s in (0,1,2)))")
mkdir -p docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4
rm -f docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/summary.csv
nohup .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mix_seta_d1 --set-b-d1 datasets/mix_setb_d1 \
  --set-a-d2 datasets/mix_seta_d2 --set-b-d2 datasets/mix_setb_d2 \
  --set-a-d3 datasets/mix_seta_d3 --set-b-d3 datasets/mix_setb_d3 \
  --set-a-d4 datasets/mix_seta_d3 --set-b-d4 datasets/mix_setb_d3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4 \
  --cells "$CELLS" --device cuda > /tmp/sweep_mix_224_L4.log 2>&1 &
echo "MIX_L4_PID: $!"
'
# Then poll, same pattern as Step 4
for i in 1 2 3 4 5; do
  sleep 90
  ssh -o BatchMode=yes swells@spark-50d2.local '
    SUM=~/ml/projects/pad-spark/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/summary.csv
    [ -f "$SUM" ] && n=$(($(wc -l < "$SUM") - 1)) || n=0
    echo "cells_done=$n/9"
    [ $n -ge 9 ] && echo "DONE" && exit 0 || echo still-running
  '
done
```

- [ ] **Step 6: Pull results back from the Spark**

```bash
rsync -az \
  swells@spark-50d2.local:~/ml/projects/pad-spark/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4 \
  swells@spark-50d2.local:~/ml/projects/pad-spark/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4 \
  docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/
```
Verify both `summary.csv` files have 10 lines (1 header + 9 data) and `runs/*.json` directories contain 9 files each.

- [ ] **Step 7: Compute the mean±std tables and comparison-vs-L3**

```bash
.venv/bin/python - <<'PY'
import csv, statistics as st
from collections import defaultdict

# The 2026-05-29 L3 baseline (from the prior sweep, immutable).
L3_BASELINE = {
    ("mask", "D1"): 0.583, ("mask", "D2"): 0.292, ("mask", "D3"): 0.291,
    ("mix",  "D1"): 0.323, ("mix",  "D2"): 0.264, ("mix",  "D3"): 0.225,
}

def load_L4(path):
    cells = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            cells[r["data_level"]].append({
                "eer_in": float(r["eer_in_domain"]),
                "eer_cross": float(r["eer_cross_domain"]),
                "acer": float(r["acer_cross_domain"]),
            })
    return cells

for which, path in (("mask", "docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/summary.csv"),
                    ("mix",  "docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/summary.csv")):
    print(f"\n{'='*60}\n{which.upper()}-only L4 (pretrained ResNet18) @ 224\n{'='*60}")
    c = load_L4(path)
    print(f"\n{'D':<4} {'in-domain':<18} {'cross-domain':<18} {'ACER':<18} {'L3 cross':<10} {'Δ':<8}")
    for D in ("D1","D2","D3"):
        vs = c[D]
        in_m = st.mean(v["eer_in"] for v in vs); in_s = st.pstdev(v["eer_in"] for v in vs)
        cr_m = st.mean(v["eer_cross"] for v in vs); cr_s = st.pstdev(v["eer_cross"] for v in vs)
        ac_m = st.mean(v["acer"] for v in vs); ac_s = st.pstdev(v["acer"] for v in vs)
        l3 = L3_BASELINE[(which, D)]
        delta = cr_m - l3
        print(f"{D:<4} {in_m:.3f}±{in_s:.3f}     {cr_m:.3f}±{cr_s:.3f}     {ac_m:.3f}±{ac_s:.3f}     {l3:.3f}      {delta:+.3f}")
PY
```
Capture the output — you'll paste it into the report section in Step 8.

- [ ] **Step 8: Write the report section**

Append after the existing "2026-05-29 update — A1 resolution bump" section in `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`:

```markdown
---

## 2026-05-30 update — B2 (pretrained ResNet18 backbone) result

The 2026-05-29 A1 sweep diagnosed model under-capacity at 224×224, and a same-day one-cell capacity spike showed ImageNet-pretrained ResNet18 cuts mask-only cross-domain EER from 0.291 (L3 from-scratch) to **0.045** at the same cell — better than the 64×64 baseline. This sweep generalises that finding to the full 18 L4 cells (9 mask + 9 mix). Code SHA: `<commit-sha>`. Wall-time: ~<MM> min mask + ~<MM> min mix on GB10.

### L4 (pretrained ResNet18) cross-domain EER @ 224 (mean ± std across 3 seeds)

[paste the mask cross-domain row from Step 7's output as a markdown table]
[paste the mix cross-domain row similarly]

### L4 in-domain EER @ 224 (mean ± std)

[paste in-domain values]

### L4 ISO 30107-3 ACER @ 224 (threshold fixed on dev at APCER ≤ 5%)

[paste ACER values]

### L4 vs L3 from-scratch (same dataset, same epochs, same trainer)

[paste the L4 cross-domain vs L3 baseline comparison rows from Step 7]

### Headline finding

[Write 2-3 sentences based on the actual numbers: did the spike generalise (L4·D3 < 0.10 → big win), partially (0.10-0.20 → real but smaller than spike), or not (~0.27 → spike was a fluke)? Update the lever-queue recommendation accordingly.]

### Raw results

- Mask-only L4 per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/runs/`](./2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/runs/) (9 files); summary CSV: [`./2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/summary.csv`](./2026-05-22-pad-spark-sweep-results/runs_mask_224_L4/summary.csv)
- Integrated L4 per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/runs/`](./2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/runs/) (9 files); summary CSV: [`./2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/summary.csv`](./2026-05-22-pad-spark-sweep-results/runs_mix_224_L4/summary.csv)
- Configs unchanged from the A1 sweep (same `mask_*` / `mix_*` configs); only the model factory differs.
```

Fill in the `<MM>`, `<commit-sha>` (the eventual merge commit), and the bracketed table-paste / headline-finding sections from the actual Step 7 numbers.

- [ ] **Step 9: Commit and push**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4 \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4
git status --short | grep -iE "datasets/|\.jpg$" || echo "OK: no datasets/images staged"
git commit -m "report(b2-l4): pretrained ResNet18 sweep at 224 -- <headline-from-data>"
git push origin <feature-branch>   # if this still runs on a branch; otherwise push main after merge
```

Verify `git status --short` shows no `datasets/*.jpg` files staged (only the docs/reports tree should be staged).

---

## Self-review notes

- **Spec coverage:** §3 file list → Tasks 1-3; §4 factory → Task 1; §5 training config (Adam lr=1e-3, no normalisation, 8 epochs, batch 32) → Task 3 uses the existing `train_and_cross_domain_eval` defaults, unchanged; §6 sweep invocation → Task 3 Steps 3-5; §7 report format → Task 3 Step 8; §8 testing → Tasks 1 + 2.
- **Type consistency:** the factory `make_resnet18_pretrained` name appears identically in `models_zoo.py`, `test_models_zoo.py` (Task 1) and `test_baseline_extensions.py` (Task 2). The `"L4"` registry key is used identically in Tasks 1, 2, and 3.
- **No placeholders:** every code block contains the actual code; the report section in Step 8 has placeholders for *data* (the numbers and the headline) — those are intentionally not pre-filled because the run hasn't happened yet. The bracketed `[paste …]` / `[Write 2-3 sentences …]` instructions are for the engineer to populate from Step 7's actual output, not for the plan to invent.
- **Out of scope:** ImageNet normalisation, per-model LR, ResNet50, A2 sensor expansion, re-running L1/L2/L3 — all explicitly deferred per spec §10.
