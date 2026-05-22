% PAD Spark Scaling — D4 Extension Design
% Add a 16k+/32k+ data tier and re-measure to confirm or qualify the data-limited diagnosis
% 2026-05-22

---

## 1. Purpose

The just-merged Spark sweep diagnosed the Phase 1.5 detector weakness as **data-limited**: cross-domain EER dropped 0.12–0.17 along the data axis at every model capacity, while the capacity axis stopped helping at D3 (TinyCNN actually beat ResNet18). The parent design recommended scaling generation as the first-class Phase 2 deliverable. This spec extends the existing 3 × 3 × 3 factorial with a **fourth data tier D4 (Set A = 16,384 samples; Set B = 32,768 samples)** to answer one question: **does the data axis continue dropping at 4× D3, plateau, or reverse?**

Parent spec: [`./2026-05-22-pad-spark-scaling-design.md`](./2026-05-22-pad-spark-scaling-design.md). All non-goals from the parent inherit unchanged (no checkpoint saving, no model integration into the CLI, no hyperparameter tuning, no real-data integration, no model architecture changes).

## 2. The question this extension answers

| D4 cross-domain EER vs D3 | Interpretation | Phase 2 implication |
|---|---|---|
| Drops by ≥ 0.05 with non-overlapping ±1σ bands | Data axis still steep at D4 | Push beyond D4 (D5 follow-up justified); generation-scale remains the dominant lever |
| Flat (within 0.05) or non-overlapping bands fail | Data-limited has saturated near D3 | Diagnosis qualifies to "data-limited up to ~10k samples; beyond that, physics is the next lever." Promotes the print-physics work in the hybrid recommendation. |
| Rises (worse than D3) | Unexpected — possible synthetic-distribution artifact (e.g. attack diversity exhausted) | Stop and investigate before generating further |

Same quantitative threshold rule as the parent spec §2: a meaningful change is **≥ 0.05 EER** with **non-overlapping ±1σ bands** across the 3 seeds.

## 3. New artifacts

**Configs (2):**

| File | Set | samples_per_bonafide | Seed | Sensor preset | Bonafide root | Total samples |
|---|---|---|---|---|---|---|
| `configs/runs/spark_seta_d4.yaml` | A | 1024 | 20260522 | mobile-front-2024 | `./datasets/_fixtures/digiface` | 16,384 |
| `configs/runs/spark_setb_d4.yaml` | B | 1024 | 20260523 | webcam-1080p | `./datasets/_fixtures/extended_fixture` | 32,768 |

Seeds match D3 exactly (Set A: 20260522, Set B: 20260523), so D4 is a same-distribution superset of D3's draw, not an independent re-roll. This isolates the "more data, same generator" axis cleanly.

**No new code modules** beyond a small script extension (§4).

## 4. Script extension: `scripts/spark_sweep.py`

- Add `"D4"` to the existing `DATA_LEVELS = ("D1", "D2", "D3", "D4")` tuple.
- Add two new argparse args: `--set-a-d4` and `--set-b-d4`, both required `Path` — same shape as the existing six.
- The implicit `--cells` default (all combinations of `CAPACITIES × DATA_LEVELS × SEEDS`) now expands from 27 to **36 cells**. Backwards compatibility: anyone re-running only the original 27 explicitly passes `--cells L1:D1:0,L1:D1:1,...` (one-time switch; we don't need to do this since the prior runs are already committed as JSONs).
- Per-cell JSON schema unchanged. Existing test `tests/test_spark_sweep.py` continues to pass (it uses `--cells L1:D1:0` explicitly, so the default-expansion change is invisible to it). One new value should also be exercised: the test must additionally accept the new `--set-a-d4` / `--set-b-d4` args (since they're required).
- The CSV-truncation gotcha from the parent project (each invocation rewrites the header) is handled the same way: regenerate the combined CSV from JSONs after the run, as a Task-10-style step.

## 5. Run plan (operational)

1. Generate the two new D4 datasets locally via `pad-synth-face generate`.
2. Verify counts: Set A D4 = 16,384, Set B D4 = 32,768.
3. rsync both datasets to the Spark.
4. Run **9 new cells** on the Spark with the extended sweep: `--cells L1:D4:0,L1:D4:1,L1:D4:2,L2:D4:0,L2:D4:1,L2:D4:2,L3:D4:0,L3:D4:1,L3:D4:2`. Same hyperparameters as the parent (10 epochs, batch size 32, `--device cuda`).
5. rsync the 9 new JSONs into the existing report directory `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs/` (alongside the existing 27).
6. Regenerate the combined `summary.csv` from all 36 JSONs.
7. Append a "**2026-05-22 update — D4 result**" section to the existing report with: a fourth column on both EER heatmaps and the training-time heatmap; the data-axis effect of D3 → D4 per capacity tier; an updated diagnosis (drops/flat/rises); and an updated Phase 2 recommendation paragraph.
8. Append a one-line update to the living `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` linking back to the updated report.

## 6. Cost budget

- Generation (laptop): rough estimate ~6 minutes total (4× D3 time per set × 2 sets).
- Training (Spark, ResNet18 dominates): ~60 s per L3·D4 seed × 3 = ~3 minutes. L1+L2 cells fast. **Total D4 sweep wall-time: ~5 minutes.**
- Disk: ~240 MB across both new datasets, ~30 MB rsync to Spark + new JSONs back.

## 7. Success criteria

- Both D4 configs committed; both datasets generate cleanly with the exact counts in §3.
- `spark_sweep.py` extension passes its existing integration test plus a one-line update if the test's required args list changes.
- 9 new D4 cell JSONs arrive in the report directory; combined CSV has 36 rows.
- The results report has a populated D4 column (in-domain heatmap, cross-domain heatmap, training-time heatmap) and a written diagnosis verdict + Phase 2 recommendation update.
- Full test suite green (137 passed, 1 skipped) — no regressions to existing tests.
- Source code of unrelated packages provably unmodified: `git diff main -- defid-pkg defid-demo-pkg pad-synth-face/src pad-synth-core/src` empty after the branch (only `pad-synth-face/tests/test_spark_configs.py` is extended with the two new D4 rows; only `scripts/spark_sweep.py` is extended; no other source changes).
