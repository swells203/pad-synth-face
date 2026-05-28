% PAD Evaluation-Metrics Upgrade Design (ISO 30107-3 + Subject-Disjoint Splits)
% Make the ruler trustworthy: APCER/BPCER/ACER at a dev-fixed operating point + subject-disjoint in-domain splits — the prerequisite to optimising anything downstream
% 2026-05-28

---

## 1. Purpose and audience

The 2026-05-27 synth→real pilot (`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`) made the trust problem explicit: every prior cross-domain number is a single threshold-free EER, the in-domain split leaks identities (`random_split` over images can land the same bonafide identity in both train and val), and there is no operating-point reporting. Before optimising any data or training lever, the ruler itself has to be trustworthy.

This spec adds:

- **ISO 30107-3 metrics** (`APCER`, `BPCER`, `ACER`, with per-PAI APCER breakdown) reported at a **threshold fixed on the in-domain dev split at a target APCER = 5%** and then applied to the cross-domain test set. This is the honest measure under domain shift: the threshold cannot peek at the test distribution.
- **Subject-disjoint in-domain splits** driven by the dataset's `manifest.jsonl` (group by `bonafide_source.id`, split identities disjointly), with the current random split kept as a fallback when no manifest is present (back-compat).

Existing EER reporting and sweep columns are **kept as-is**; new metrics are **additive**, so prior reports remain comparable.

Audience: maintainers, future sub-projects that need a trustworthy baseline (DFDC-grounded synthesis, capture-domain randomisation, synth-pretrain → real-finetune curve — see `pad-next-sub-projects` memory).

## 2. The question this answers

| Behaviour on the next synth→real run | What it tells us |
|---|---|
| EER unchanged from prior reports; subject-disjoint in-domain EER drops slightly vs the old random-split value | Old in-domain numbers were modestly inflated by identity leakage. New numbers are honest. |
| Cross-domain `ACER ≈ 0.5` and per-PAI `APCER` near 1.0 at the dev-fixed threshold | Confirms the synth→real pilot: the model collapses to "predict bonafide" at any reasonable operating point under shift. Sets the baseline every later lever must beat. |
| Per-PAI `APCER` shows wildly different values across attack types | Pinpoints which PAI species (print/replay/mask/...) generalises worst — actionable signal for data/physics work. |

Decision rule: the upgrade is "done" when the same 27-cell sweep, on the same data, reports all of `{eer_*, apcer_cross_domain, bpcer_cross_domain, acer_cross_domain, threshold, apcer_per_pai}` deterministically (same-seed reproducibility), AND the existing `eer_in_domain`/`eer_cross_domain` numbers move only by what the subject-disjoint split change should plausibly produce (a small in-domain shift; cross-domain numerically identical because the cross-domain eval set is unchanged).

## 3. Architecture and files

One new pure-functions module, additive modifications to the eval baseline and the sweep writer, plus tests. No new external dependencies.

| Change | File | Note |
|---|---|---|
| New metrics module | `pad-synth-core/src/pad_synth_core/eval/metrics.py` | Pure functions: `compute_eer`, `threshold_at_apcer`, `apcer_bpcer_acer` |
| Move existing `compute_eer` | `pad-synth-core/src/pad_synth_core/eval/baseline.py` → re-export from `metrics.py` | Keep the import path live for back-compat (`from ... eval.baseline import compute_eer`) |
| Manifest-aware dataset attributes | `eval/baseline.py` (`TinyPADDataset`) | Each item additionally exposes `subject: str | None` and `attack_type: str | None`, populated from `<root>/manifest.jsonl` when present |
| Subject-disjoint split helper | `eval/baseline.py` | `subject_disjoint_split(dataset, val_fraction, seed) -> (train_subset, val_subset)`; random fallback if any item has `subject is None` |
| Integrate metrics + split | `eval/baseline.py` (`train_and_cross_domain_eval`) | Use subject-disjoint split when available; compute the new ISO metrics; return additive keys |
| Surface new metrics in the sweep | `scripts/spark_sweep.py` | Add CSV columns (`acer_cross_domain`, `apcer_cross_domain`, `bpcer_cross_domain`, `threshold`) and JSON fields (incl. `apcer_per_pai`) |
| Tests | `pad-synth-core/tests/test_eval_metrics.py`, additions to existing eval tests | Hand-computed cases; split correctness; integration |

## 4. Metric definitions (ISO 30107-3)

Score conventions match today's code: `score = P(attack)`, classify as attack if `score ≥ threshold`.

- **APCER per PAI species `s`**: fraction of attack presentations of species `s` *incorrectly classified as bona fide* (i.e. with `score < threshold`).
  `APCER_s = #{attacks of type s with score < threshold} / #{attacks of type s}`.
- **APCER (overall)**: `APCER = maxₛ APCER_s` — ISO worst-case PAI. (We do **not** report mean-over-PAI in this cycle.)
- **BPCER**: fraction of bona fide presentations *incorrectly classified as attacks* (`score ≥ threshold`).
  `BPCER = #{bonafide with score ≥ threshold} / #{bonafide}`.
- **ACER**: `(APCER + BPCER) / 2`.
- **`threshold_at_apcer(scores, labels, attack_types, target_apcer=0.05)`**: scan unique scores; return the threshold that produces the largest overall APCER value not exceeding `target_apcer` (using as much of the APCER budget as the data allows). At equal APCER, prefer the **higher** threshold — at the same attack-miss rate, a higher threshold flags fewer bonafide samples (lower BPCER), the better operating point under the same budget. Returns `(threshold, achieved_apcer)`.
- **`compute_eer(scores, labels)`**: unchanged numerically from today's implementation; moved into `metrics.py` and re-exported.

All functions are pure: inputs are `list[float]` / `list[int]` / `list[str | None]` and a scalar threshold; no I/O, no globals.

## 5. Subject-disjoint split (manifest-driven, with fallback)

`TinyPADDataset.__init__` does what it does today (glob image paths under `face/{bonafide,<x>}/*.jpg`), and *additionally*:

1. If `<root>/manifest.jsonl` exists, load it once and build a map `output_path (str) → (subject, attack_type)` where `subject = record["bonafide_source"]["id"]` and `attack_type = record["attack_type"]` (None for bonafide). Each `__getitem__` still returns `(tensor, label)` to preserve the existing PyTorch interface; the per-item `subject` and `attack_type` are exposed via parallel attributes `self.subjects: list[str | None]` and `self.attack_types: list[str | None]`, indexed identically to `self.items`.
2. If the manifest is missing, both attribute lists are filled with `None` and the dataset behaves exactly as today.

`subject_disjoint_split(dataset, val_fraction, seed)`:

- If every `subject` is `None`, return `random_split` (today's behaviour).
- Otherwise group sample indices by subject, deterministically permute the **identity** order with `seed`, take a prefix totalling ≥ `val_fraction × len(dataset)` as the val set, and the rest as train. Subjects never cross the boundary.

`train_and_cross_domain_eval` calls this helper for the in-domain split. Cross-domain eval still uses the full `eval_root` dataset (unchanged).

## 6. Threshold policy: fixed on dev, applied to test

After training:

1. Score every dev sample → `dev_scores`. Compute `(t_dev, apcer_dev) = threshold_at_apcer(dev_scores, dev_labels, dev_attack_types, target_apcer=0.05)`.
2. Score every cross-domain sample → `test_scores`. Compute `apcer_bpcer_acer(test_scores, test_labels, test_attack_types, threshold=t_dev)` → `(apcer_per_pai, apcer_max, bpcer, acer)`.
3. Threshold-free EER on both sets is computed and returned exactly as today.

If `eval_root` is `None`, the cross-domain block is `None` (matches today's return-shape).

The target APCER (default `0.05`) is an argument to `train_and_cross_domain_eval` so future protocols can sweep it.

## 7. Sweep output (additive, backward-compatible)

`scripts/spark_sweep.py` keeps every existing column and adds:

- CSV columns: `acer_cross_domain`, `apcer_cross_domain`, `bpcer_cross_domain`, `threshold`. Old `eer_in_domain` / `eer_cross_domain` columns stay in the same positions.
- Per-cell JSON adds the four scalars above plus `apcer_per_pai: dict[str, float]` and `target_apcer: 0.05`.

Old report tables remain comparable since the existing columns are untouched and numerically identical (cross-domain EER doesn't depend on the new threshold).

## 8. Testing

`pad-synth-core/tests/test_eval_metrics.py`:

- **`compute_eer` regression**: hand-built scores/labels where the EER is known by inspection (or asserted against today's value to verify the move didn't change behaviour).
- **`apcer_bpcer_acer` hand-computed**: small fixture of `(scores, labels, attack_types)` with a fixed threshold; assert `apcer_per_pai`, `apcer_max`, `bpcer`, `acer` to exact values. Includes the all-bonafide and all-attack edge cases.
- **`threshold_at_apcer` correctness**: known scores with two PAI species; assert the chosen threshold actually achieves `APCER ≤ target` and that no higher threshold would also satisfy the constraint.
- **PAI grouping**: a record with `attack_type=None` (bonafide) must never contribute to `APCER`; a record without a matching PAI key must not raise.

Subject-disjoint split tests (in `pad-synth-face/tests/`):

- **No identity leakage**: build a fixture dataset with known subjects, run `subject_disjoint_split`, assert `set(train_subjects) ∩ set(val_subjects) == ∅`.
- **Random fallback**: dataset without a manifest → split still works and produces non-empty train/val.
- **Determinism**: same seed → same partition.

Integration test:

- Run `train_and_cross_domain_eval(train_root=<synthetic fixture dataset>, eval_root=<real fixture from build_fixture_real_attack + ingest>, epochs=1, device="cpu")`; assert the return dict contains finite numeric values for every new key, `apcer_per_pai` has an entry per attack type in `eval_root`, and `threshold` is a finite float.

## 9. Out of scope (this cycle)

- **Leave-one-dataset-out cross-domain protocol** — agreed deferred to its own cycle. It's a thin orchestration layer on top of the metrics + dataset changes shipped here.
- **Sweeping multiple operating points** (e.g. APCER ∈ {1%, 5%, 10%}) — the API supports it via the new argument, but the sweep script reports a single operating point this cycle. Adding columns is additive and can ship later.
- **Per-condition slicing** (lighting/capture/device sub-metrics) — requires richer eval-set metadata than the harness currently captures.
- **APCER mean-over-PAI** — explicitly chosen against; only the ISO worst-case (max) is reported this cycle.
