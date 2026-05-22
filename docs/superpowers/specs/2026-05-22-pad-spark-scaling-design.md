% PAD Spark Scaling Experiment — Design
% Capacity × Data factorial on the DGX Spark to disambiguate the Phase 1.5 open question
% 2026-05-22

---

## 1. Purpose and audience

A focused experiment on the DGX Spark (`spark-50d2.local`, NVIDIA GB10) that produces a **decision-matrix update** for the PAD synthetic-dataset roadmap. The Phase 1.5 cross-domain proxy returned an EER of 0.36 (multi-seed mean ≈ 0.39) — interpretable as the detector being weak, but not as to **why**. The decisions/roadmap report identifies three hypotheses: *physics-limited*, *capacity-limited*, *data-limited*. This experiment disambiguates them.

The deliverable is the answer plus a written report; no trained detector is kept, no model is integrated into the `pad-synth-face` CLI (those are out of scope — see §8).

Audience: future maintainers picking up Phase 2 prioritization; the Phase 2 spec author.

## 2. The question this experiment answers

For each cell of a `capacity × data` factorial trained on the same Phase 1.5 data-generating process (physics held constant):

| Observed effect on cross-domain EER | Diagnosis | Phase 2 implication |
|---|---|---|
| EER drops along the **L** (capacity) axis but flat across **D** (data) | capacity-limited | Phase 2 should include a model upgrade in addition to physics |
| EER drops along **D** but flat across **L** | data-limited | Phase 2 should scale generation and revisit real-data integration |
| EER drops along **both** | both | Phase 2 should address both; weight by slope |
| EER drops along **neither** | physics-limited | Confirms the hybrid Phase 2 (print physics + mask attack); a model swap won't help |

The hybrid Phase 2 is already the recommended direction in the decisions/roadmap report. This experiment either confirms it or sharpens it.

**Quantitative threshold for "drops along an axis":** a cross-domain EER reduction of **≥ 0.05** at the axis extreme (L3 vs L1 at fixed D, or D3 vs D1 at fixed L) — measured as a mean across the 3 seeds, with the means separated by more than one std (i.e. the bands don't overlap). Under that threshold, the axis is considered flat. This is the rule used to populate the diagnosis table above.

## 3. Experimental design

### 3.1 Factorial grid (27 runs)

3 capacities × 3 data levels × 3 seeds (`seed ∈ {0, 1, 2}`).

**Capacities:**

- **L1 — TinyCNN**: the existing `pad_synth_core.eval.baseline.TinyCNN` (3→8→16 conv with AdaptiveAvgPool, Linear(16, 2); a few hundred parameters). This is the floor — the existing baseline.
- **L2 — SmallCNN**: a new ~100k-parameter CNN. Architecture (exact, no placeholder):
  ```
  Conv2d(3,  16, k=3, p=1) -> ReLU -> MaxPool2d(2)
  Conv2d(16, 32, k=3, p=1) -> ReLU -> MaxPool2d(2)
  Conv2d(32, 64, k=3, p=1) -> ReLU -> MaxPool2d(2)
  Conv2d(64, 128, k=3, p=1) -> ReLU -> AdaptiveAvgPool2d(1)
  Flatten -> Linear(128, 2)
  ```
  ≈ 97k parameters.
- **L3 — ResNet18**: `torchvision.models.resnet18(weights=None)` with the final `fc` replaced by `Linear(512, 2)`. Trained from scratch — no pretrained weights, because real-image-pretrained features would inject a domain prior that confounds a synthetic-only experiment. ≈ 11M parameters.

**Data levels** (controlled by `bonafide.samples_per_bonafide` in the run config):

| Level | `samples_per_bonafide` | Set A total (8 IDs) | Set B total (16 IDs) |
|---|---|---|---|
| **D1** | 6 / 4 (matches current Phase 1 / Phase 1.5) | 96 | 128 |
| **D2** | 32 / 32 | 512 | 1024 |
| **D3** | 256 / 256 | 4096 | 8192 |

(D1 keeps the existing Phase 1.5 baseline exactly, so the L1·D1 cell is a regression check on the 0.36 number.)

### 3.2 Training and evaluation framing

Each cell trains on Set A at its data level and evaluates twice via the existing `pad_synth_core.eval.baseline.train_and_cross_domain_eval`:

- **In-domain EER** — held-out 25% split from Set A. Comparable to the Phase 1 metric (0.29 in-domain).
- **Cross-domain EER** — entire Set B at the matching data level. Comparable to the Phase 1.5 metric (0.36 cross-domain).

Training hyperparameters are held constant across all cells: Adam lr=1e-3, CrossEntropyLoss, **10 epochs** (matches the published Phase 1.5 baseline that produced the 0.36 cross-domain number), batch_size = 32. The experiment isolates capacity × data; hyperparameter tuning is out of scope (§8). The existing `train_and_cross_domain_eval` already returns both EERs in one call.

### 3.3 GPU placement

All training and inference run on the GB10. The existing baseline is CPU-only; the runner must move `model` and each batch to `cuda` (deterministic flag enabled — `torch.use_deterministic_algorithms(True)` where possible; document the per-seed variance band as the multi-seed std in the report).

## 4. Dataset generation

Generated **locally** (laptop) using the existing tested `pad-synth-face generate` CLI, then `rsync`'d to the Spark. Reasons: deterministic byte-identical regeneration from any machine; the Spark only needs torch + numpy + Pillow; the generator stack stays where it is tested.

Six new configs under `configs/runs/`:

| File | Set | samples_per_bonafide | Seed | Sensor preset | Bonafide root |
|---|---|---|---|---|---|
| `spark_seta_d1.yaml` | A | 6 | 20260522 | `mobile-front-2024` | `datasets/_fixtures/digiface` |
| `spark_seta_d2.yaml` | A | 32 | 20260522 | `mobile-front-2024` | `datasets/_fixtures/digiface` |
| `spark_seta_d3.yaml` | A | 256 | 20260522 | `mobile-front-2024` | `datasets/_fixtures/digiface` |
| `spark_setb_d1.yaml` | B | 4 | 20260523 | `webcam-1080p` | `datasets/_fixtures/extended_fixture` |
| `spark_setb_d2.yaml` | B | 32 | 20260523 | `webcam-1080p` | `datasets/_fixtures/extended_fixture` |
| `spark_setb_d3.yaml` | B | 256 | 20260523 | `webcam-1080p` | `datasets/_fixtures/extended_fixture` |

`samples_per_bonafide=4` for Set B D1 matches the existing `phase15_setb.yaml` exactly so the D1 cell is a regression check.

Generation output: `datasets/spark_set{a,b}_d{1,2,3}/` with the same manifest+provenance shape as existing datasets. `rsync -av --partial datasets/spark_set*_d* swells@spark-50d2.local:~/ml/datasets/`.

## 5. Smoke-test first cell (workflow gate)

Before running the full 27-cell grid, run **one** end-to-end cell to validate the Spark workflow: `(L1 TinyCNN, D1, seed=0)`. Gate to pass before continuing:

- Cross-domain EER ∈ [0.33, 0.39] (Phase 1.5's known 0.36 ± 0.03; allows for small CUDA-vs-CPU numerical differences).
- Smoke cell wall-time < 5 minutes.

If the gate fails, diagnose (torch/CUDA, dataset rsync integrity, determinism flag) before spending wall-time on 26 more runs.

## 6. Spark-side workflow

- **Project root on Spark**: `~/ml/projects/pad-spark/` (a git checkout of the local repo via SSH push to a bare repo at `~/ml/projects/pad-spark.git`, or `rsync -av` of the working tree).
- **Python environment**: install `uv` on the Spark (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then in the project root: `uv venv && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 numpy Pillow pyyaml`. Nightly wheel for Blackwell sm_121 support. Pin the resolved versions to `requirements.spark.txt` after install for reproducibility.
- **Datasets**: `~/ml/datasets/spark_set{a,b}_d{1,2,3}/` (rsync'd from the laptop).
- **Sweep runner**: a new top-level `scripts/spark_sweep.py` that takes the grid as input and runs each cell. Calls into `pad_synth_core.eval.baseline.train_and_cross_domain_eval` for L1; calls into a new sibling module `pad-synth-core/src/pad_synth_core/eval/models_zoo.py` (factories `make_small_cnn()`, `make_resnet18()`) for L2/L3. The runner must accept a model factory argument and an optional CUDA device, so the existing `train_and_cross_domain_eval` is extended (backwards-compatible default keeps the current behavior).
- **Outputs on Spark**: `~/ml/logs/pad-spark/<UTC-timestamp>/runs/<L>_<D>_<seed>.json` (one file per cell run) + `summary.csv` (the 27-row table) + `summary.md` (the report).
- **Outputs back**: `rsync -av swells@spark-50d2.local:~/ml/logs/pad-spark/<timestamp>/ docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/` (the directory becomes the committed report artifact). The committed report `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` is the top-level summary; raw per-cell JSON lives under it.

## 7. Result artifact

### 7.1 Per-cell JSON (one per run, on Spark; rsync'd back)

```json
{
  "capacity": "L1|L2|L3",
  "data_level": "D1|D2|D3",
  "seed": 0,
  "samples_per_bonafide_a": 6,
  "samples_per_bonafide_b": 4,
  "n_train": 72,
  "n_val_in_domain": 24,
  "n_val_cross_domain": 128,
  "eer_in_domain": 0.29,
  "eer_cross_domain": 0.36,
  "val_accuracy_in_domain": 0.71,
  "val_accuracy_cross_domain": 0.64,
  "train_seconds": 12.3,
  "git_sha": "<hex>",
  "torch_version": "2.x.x+cu128",
  "cuda_version": "13.0",
  "device": "cuda:0"
}
```

### 7.2 Summary CSV

Columns: `capacity, data_level, seed, eer_in_domain, eer_cross_domain, train_seconds`. 27 rows.

### 7.3 Summary report (committed `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`)

- 3×3 heatmap of cross-domain EER (mean ± std across seeds)
- 3×3 heatmap of in-domain EER (mean ± std across seeds)
- Per-cell training time (one number, the median)
- Diagnosis: which axis (L, D, both, neither) explains the variance — written in plain English
- Recommendation update for Phase 2 — a paragraph linked from the living decisions/roadmap report
- A one-line PR-style update appended to `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` linking the new report

## 8. Explicit non-goals (YAGNI)

- **No** model checkpoint saving, no model card, no model integration into the `pad-synth-face eval` CLI (that's the deferred "deliverable B"; revisit after Phase 2's physics changes land).
- **No** real-data integration (no DigiFace-1M wiring beyond the existing fixture stand-in).
- **No** deepfake / generator-zoo Phase 2.5 work.
- **No** hyperparameter sweep (Adam lr=1e-3, batch_size=32, 8 epochs are held constant across cells).
- **No** Spark-side ollama or LLM-agent integration (a separate use case for the box).
- **No** changes to `defid-pkg/` or `defid-demo-pkg/` (different project).

## 9. Success criteria

- 27 cells (9 grid cells × 3 seeds) run to completion on the Spark, producing 54 EER readings (27 in-domain + 27 cross-domain), summarized as 18 (mean, std) pairs across the 9 grid cells.
- Smoke cell cross-domain EER in [0.33, 0.39].
- A written diagnosis (capacity-/data-/physics-limited / both) in the new report.
- All artifacts committed to `main`: the 6 new configs, `scripts/spark_sweep.py`, `pad-synth-core/src/pad_synth_core/eval/models_zoo.py` (and any backwards-compatible extension to `baseline.py`), the rsync'd results directory, and the summary report.
- Existing `defid` / `defid-demo` / `pad-synth-face` suites continue to pass unchanged.
