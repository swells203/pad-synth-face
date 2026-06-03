# PAD — B1 Synth-Pretrain → Real-Finetune Curve (design)

**Date:** 2026-06-03
**Status:** approved (brainstorm) → ready for implementation plan
**Branch:** `feat/pad-b1-finetune-curve`

## 1. Problem & motivation

The PAD models are trained purely on synthetic data. The **hybrid hypothesis**
— the architecture commercial PAD systems actually use — is *pretrain on
synthetic, finetune on a small amount of real data*. B1 (Lever B1 in memory
`pad-next-sub-projects`) quantifies this: real-test EER as a function of the
number of real finetune samples `N` (0/50/200/1000). N=0 is the synth-only
baseline; the curve's slope answers "how much real data buys how much EER."

**Data reality:** only the n=55 AxonData free-sample pilot is staged
(`datasets/_real_attack/axondata`), and it is licensed **CC-BY-NC-4.0
(NonCommercial)** — research-only, like DigiFace/DFDC (see memory
`pad-commercial-licensing`). n=55 cannot support the full curve (finetuning on
N=50 leaves ~5 for test). So this sub-project **builds the harness and
validates it mechanically on n=55**; the real N=0/50/200/1000 curve runs later
on purchased/larger real data — the same inert-scaffolding pattern as the DFDC
and commercial-bonafide harnesses.

## 2. Goal & non-goals

**Goal:** A reusable capability + runner that, given a synthetic pretrain set
and a real set, pretrains once, finetunes on N real samples for each N in a
list, and reports real-test EER (+ ISO metrics) per N — producing the hybrid
curve. Mechanically proven on the n=55 pilot; turnkey for real data.

**Non-goals:**
- No real scientific curve now (n=55 is plumbing-grade; only N=0 + ~pool-size
  reachable).
- No plotting/matplotlib (a text table suffices).
- No multi-seed averaging in v1 (single seed; multi-seed is a later flag).
- No new model factories (reuse the `L4` pretrained ResNet18).
- No change to `train_and_cross_domain_eval` or the existing sweep — B1 is two
  new functions beside it + one runner script.

## 3. Decisions (locked in brainstorm)

| Decision | Choice |
|---|---|
| Deliverable | **Build harness, defer real curve** (validate on n=55) |
| Finetune mode | **Configurable** `full` (default) / `head` (freeze backbone) |
| Architecture | **Approach 1** — new functions in `baseline.py` + runner script |

## 4. Architecture

Three units, max reuse of existing eval helpers (`TinyPADDataset`,
`subject_disjoint_split`, `_score_dataset`, `compute_eer`, `threshold_at_apcer`,
`apcer_bpcer_acer`, the `FACTORIES` model zoo).

```
synth_root ──► pretrain_on_synth() ──► trained model ──► state_dict snapshot
                                                              │ (once)
real_root ──► subject_disjoint_split ──► (finetune_pool, real_test)
                                                              │
for N in n_list:  finetune_and_eval_on_real(state_dict, pool[:N], real_test, mode)
                                                              │
                                              <output-dir>/runs/N<n>_seed<s>.json
                                                              │
                                              curve_summary.json + printed table
```

### 4.1 `pretrain_on_synth(...)` — `pad_synth_core/eval/baseline.py`

```
pretrain_on_synth(
    synth_root: Path, model_factory: Callable[[], nn.Module],
    epochs: int = 8, lr: float = 1e-3, batch_size: int = 8,
    seed: int = 0, device: str | None = None,
) -> nn.Module
```
Trains a fresh `model_factory()` model on the full synthetic root (no val split
needed — pretraining uses all of it), Adam at `lr`, CrossEntropyLoss, the same
epoch loop as `train_and_cross_domain_eval`. Returns the trained model. The
runner snapshots `model.state_dict()` once so every N forks from one pretrain.

### 4.2 `finetune_and_eval_on_real(...)` — `pad_synth_core/eval/baseline.py`

```
finetune_and_eval_on_real(
    pretrained_state: dict, model_factory: Callable[[], nn.Module],
    finetune_pool: Dataset, real_test_root: Path, n_real: int,
    mode: str = "full", epochs: int = 8, lr: float = 1e-4,
    batch_size: int = 8, seed: int = 0, device: str | None = None,
    target_apcer: float = 0.05,
) -> dict
```
- Builds `model_factory()`, loads `pretrained_state`.
- `mode="head"` → set `requires_grad=False` on every param **not** under the
  final classifier module `fc` (the ResNet head); `mode="full"` → all params
  trainable. The optimizer is built over
  `filter(lambda p: p.requires_grad, model.parameters())` at `lr` (the lower
  finetune LR). **`head` mode assumes a ResNet-style `.fc` head** — i.e. the
  `L3`/`L4` resnet factories (the production backbones). If a model exposes no
  `fc` attribute, `head` mode raises a clear `ValueError` rather than silently
  freezing the whole network; `full` mode works for any factory.
- Takes the first `n_real` items of `finetune_pool` (the pool is pre-shuffled
  deterministically by the runner). **`n_real == 0` → skip the finetune loop
  entirely** → evaluates the pretrained model as-is (the synth-only baseline).
- Evaluates on `TinyPADDataset(real_test_root)` via `_score_dataset` +
  `compute_eer` + the ISO threshold/APCER/BPCER/ACER path (threshold fixed on
  the finetune set when it has PAI metadata, else None — same convention as the
  existing function).
- Returns a dict with the **same metric keys the sweep already emits** (so it is
  report/aggregation-compatible) plus `n_real` and `mode`. The **cross-domain
  keys** (`eer_cross_domain`, `acer_cross_domain`, …) hold the **real-test**
  numbers — these are the curve's y-values. `n_train` holds `n_real`. The
  in-domain keys reflect the finetune-set fit (or are `None` when `n_real == 0`,
  since there is no finetune set in the synth-only baseline).

### 4.3 Real-data split (reuse `subject_disjoint_split`)

The runner splits the real set ONCE into a **fixed** subject-disjoint
`(finetune_pool, real_test)` via `subject_disjoint_split(real_ds,
val_fraction=test_fraction, seed)` — the same val fraction sits on the test
side. Fixed across all N → the curve is fair (identical test set), and the test
subjects never appear in finetune data.

**Granularity caveat:** `TinyPADDataset.subjects` = `bonafide_source.id`. For
the AxonData pilot those are per-selfie filenames, so the split is effectively
*sample*-disjoint (no real subject grouping). No leakage either way; real
subject grouping applies automatically once a properly-structured purchased
dataset (with true subject IDs) is used. The harness logic is unchanged.

**Class-balance guard:** because the split is subject-blind to label, the runner
asserts the `real_test` partition contains **both** classes (≥1 bonafide and ≥1
attack); EER is undefined otherwise. If a split is degenerate (tiny data), it
raises a clear `ValueError` naming the problem rather than emitting a
meaningless EER.

### 4.4 Runner — `scripts/b1_finetune_curve.py`

- **Args:** `--synth-root` (required), `--real-root` (required), `--n-list`
  (default `0,50,200,1000`), `--finetune-mode` (`full`|`head`, default `full`),
  `--test-fraction` (default `0.3`), `--model` (FACTORIES key, default `L4`),
  `--pretrain-epochs` (default 8), `--finetune-epochs` (default 8),
  `--finetune-lr` (default 1e-4), `--batch-size` (default 8), `--seed` (default
  0), `--output-dir` (required).
- **Flow:** load real set → `subject_disjoint_split` into `(pool, test)` →
  class-balance guard → deterministically shuffle the pool (seeded) →
  `pretrain_on_synth` once → snapshot `state_dict` → for each N in `--n-list`:
  if `N <= len(pool)` run `finetune_and_eval_on_real`, write
  `<output-dir>/runs/N<n>_seed<s>.json`; else **`log` "requested N=<n>, pool has
  <p> — skipped"** (no silent capping).
- **Summary:** write `<output-dir>/curve_summary.json` (list of
  `{n_real, eer, acer, skipped}`) and print a table `N | real-test EER | ACER`
  plus a one-line readout: does finetuning help (EER@max-N vs EER@N=0). Pure
  aggregation over the per-N JSONs — no separate verdict script.
- **CLI:** core logic in importable functions (`split_real`, `run_curve`,
  `_render_curve`) so tests don't shell out; `main(argv) -> int`.

### 4.5 Tests + validation + docs

- **Unit tests** (`pad-synth-core/tests/test_b1_finetune.py`, tiny generated
  fixtures, no real faces): `head` mode leaves backbone params unchanged after
  finetune while `fc` changes; `full` mode changes backbone; `n_real=0` skips
  finetune and equals the pretrained eval; the class-balance guard raises on a
  single-class test split; N>pool is skipped+logged not capped.
- **Runner test** (`tests/test_b1_finetune_curve.py`): `run_curve` over a tiny
  synth + tiny real fixture writes per-N JSON with the expected keys and a
  summary; `main` returns 0.
- **Mechanical validation on n=55** (operational, like the commercial dry-run):
  run the curve locally with a small synthetic pretrain set + the AxonData
  pilot, `--n-list 0,<pool>`, confirm end-to-end run + well-formed JSON. EER
  values meaningless at this scale — plumbing proof only.
- **`docs/b1-finetune-curve.md`** runbook: the split discipline, the one-command
  curve invocation (local validation + Spark for the real curve), and the
  honesty note (real N≥50 needs purchased/larger real data; the AxonData free
  sample is CC-BY-NC-4.0 research-only).
- **Report:** append a B1 section to
  `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (the curve
  table once a real run exists).

## 5. File manifest

| File | Action |
|---|---|
| `pad-synth-core/src/pad_synth_core/eval/baseline.py` | **Modify** (add `pretrain_on_synth`, `finetune_and_eval_on_real`; no change to existing functions) |
| `scripts/b1_finetune_curve.py` | **Create** |
| `pad-synth-core/tests/test_b1_finetune.py` | **Create** |
| `tests/test_b1_finetune_curve.py` | **Create** |
| `docs/b1-finetune-curve.md` | **Create** |
| `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` | **Modify** (append B1 section — only once a real run exists; deferred) |

## 6. Test strategy

TDD per new function: failing test → implement → green. Fixtures are tiny
generated PAD trees (bonafide + one attack type, a handful of images each) — no
real or licensed faces enter the repo (`datasets/` is gitignored regardless).
The pretrain/finetune loops run on CPU in the tests at 1 epoch.

## 7. Success criteria

- `pretrain_on_synth` + `finetune_and_eval_on_real` implemented with the
  `full`/`head` mode branch and the `n_real=0` baseline path; unit tests green.
- The runner produces per-N JSON (sweep-compatible keys) + a curve summary, caps
  nothing silently, and guards against degenerate (single-class) test splits.
- End-to-end mechanical validation on n=55 runs and emits well-formed JSON.
- `train_and_cross_domain_eval` and all existing tests are unchanged/green.
- The harness is inert/ready: the real curve is one command away once larger
  real data is staged.
