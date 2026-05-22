% PAD Print Attack — Physics v2.1 Design (halftone jitter)
% Break the deterministic-halftone watermark by adding per-sample jitter
% 2026-05-22

---

## 1. Purpose and audience

The just-merged v2 sweep numerically "fired" (cross-domain EER drops of +0.156 to +0.250) across all 9 cells, BUT 6/9 cells hit exactly 0.000 EER — diagnosed as a **generator-fingerprint artifact**: the v2 halftone uses a fully deterministic screen geometry (fixed angles, fixed cell-size formula, no translation), so two samples with the same `print_dpi` produce visually identical halftone patterns. The detector trivially learns "this exact halftone screen" and the pattern is identical in Set A and Set B.

v2.1 fixes the watermark by adding **per-sample jitter** to the halftone screen geometry. The goal is for two print attacks at the same `print_dpi` to have *visually different* halftone signatures — mirroring real-printer variability. No new physics layer is added; the halftoning algorithm is reparameterized to be probabilistic instead of deterministic.

Audience: future maintainers, the Phase 2 author. This is a follow-on to the just-merged v2 work. Mask attack module is still a separate next sub-project.

## 2. The question this answers

Does adding sub-pixel offset / angle / cell-size jitter to halftoning break the v2 watermark while preserving the cross-domain gain over v1?

| v2.1 cross-domain EER vs v1 and v2 | Diagnosis | Phase 2 implication |
|---|---|---|
| **No cell hits 0.000**, AND at least one D3 cell shows v2.1 ≤ v1 − 0.05 with non-overlapping bands | Watermark broken; physics gain survives jitter. | Ship v2.1 as production print attack; proceed to mask-attack sub-project. |
| **No cell hits 0.000** but v2.1 ≈ v1 (no meaningful improvement at D3) | Jitter broke the watermark AND erased the gain — the v2 "improvement" was entirely the artifact. | Physics axis confirmed as not the lever; escalate to real-data integration as the dominant Phase 2 candidate. |
| **Some cell still hits 0.000** | The jitter wasn't enough; another deterministic feature is leaking. | Stop and audit — try heavier jitter or add a fourth source (dot-shape categorical) before another sweep. |

The decision rule for "broken watermark" is binary: **no v2.1 cross-domain cell ≤ 0.001 EER** (since EER bins on this dataset are quantized, exact 0.0 vs 0.001 distinguishes "perfect" from "near-perfect, real-data-like"). The decision rule for "gain survives" is the same as v2's: **≥ 0.05 drop with non-overlapping ±1σ bands** at D3 vs v1.

## 3. Jitter mechanisms (concrete)

Three per-channel per-sample jitter sources, all driven by the RNG already passed into `PrintAttack.simulate`:

| Source | Distribution | Sampled per |
|---|---|---|
| Sub-pixel offset of screen origin `(dx_c, dy_c)` | uniform `[−cell_px/2, +cell_px/2]` each axis | channel `c ∈ {C,M,Y,K}`, per sample |
| Angle jitter `Δθ_c` added to base angle | normal `μ=0, σ=3.0°` | channel `c`, per sample |
| Cell-size jitter multiplier `k_c` | uniform `[0.90, 1.10]` | channel `c`, per sample |

Effective per-channel screen parameters become `cell_px_c = base_cell * k_c` and `angle_c = base_angle_c + Δθ_c`, with the screen evaluated at translated coords `(x − dx_c, y − dy_c)`.

`base_cell` is the existing v2 formula: `max(2.0, round(8.0 × 150.0 / float(print_dpi)))`.

The jitter draws cost **16 random numbers per sample** total (4 channels × {dx, dy, Δθ, k} = 16). Negligible cost.

Skipped (deferred to a hypothetical v2.2 if needed): dot-shape categorical variation (round/elliptical/euclidean), dot-gain multiplicative noise (already provided by the existing `_paper_texture` step).

## 4. API change — minimal

`_apply_halftone(rgb, print_dpi)` → `_apply_halftone(rgb, print_dpi, rng)`. Internal helpers `_dot_screen` and `_halftone_channel` gain optional jitter parameters with sensible defaults that preserve the deterministic behavior when no rng is provided (so existing unit tests covering the deterministic case continue to work).

`PrintAttack.simulate` already passes `rng` to other helpers (`_paper_texture`, `_perspective_warp`); one extra `rng` argument to `_apply_halftone`. No other changes to `simulate`.

`_apply_icc`, `_PAPER_TINTS`, `_paper_texture`, `_perspective_warp`, `_apply_cutout` — all unchanged.

## 5. Ontology version bump

`ontology/face/print.yaml` `version` bumped from `"2026-05-22"` to `"2026-05-23"`.

**No new axes.** Jitter parameters (the σ=3.0°, the uniform ranges) are algorithm-level constants, like the existing `_paper_texture` `scale=0.03`. They are not user-tunable ontology knobs and should not be sampled per-sample as additional axes — the per-sample variability comes from the rng draws inside `_apply_halftone`, not from sampling distribution parameters.

The version bump's purpose is provenance: manifests stamp `ontology_version=2026-05-23` (modulo the deferred pipeline.py hardcode — see §8) so v2 vs v2.1 datasets are self-identifying.

## 6. Determinism preserved

Each sample receives a per-sample RNG seeded from the master seed via the existing `derive_sample_seed(seed, modality, attack_type, counter)` flow. Same master seed → same per-sample RNG state → same jitter draws → same output. The full pipeline remains byte-deterministic.

`tests/golden/golden_hashes.json` MUST be regenerated as part of this work — the 4 print hashes will change (replay hashes unchanged due to per-attack RNG isolation, same as v2→v2.1's golden diff).

## 7. Measurement plan

After implementation:

1. **Generate 6 v2.1 datasets** at D1–D3 only (skip D4): `datasets/v21_seta_d{1,2,3}/` and `datasets/v21_setb_d{1,2,3}/`. Six new configs under `configs/runs/v21_*.yaml`. Same seeds as the v1 and v2 sweeps (20260522 / 20260523).
2. **rsync** the 6 datasets to the Spark.
3. **Run 27-cell sweep** on the Spark using v2.1 datasets: 3 capacities × 3 data levels × 3 seeds, same hyperparameters as v2 (10 epochs, batch 32, `--device cuda`).
4. **rsync results back**, regenerate combined CSV (separate `summary_v21.csv`), append a **"v2.1 result"** section to the existing results report with three columns per cell (v1, v2, v2.1 cross-domain EER + verdict).
5. One-line decisions/roadmap update with the verdict (artifact broken / not broken; gain survives / lost).

## 8. Architecture / component boundaries

- **Modified in place:** `pad-synth-face/src/pad_synth_face/attacks/print.py` — extend `_apply_halftone`, `_dot_screen`, `_halftone_channel` signatures and bodies; one-line change in `simulate` to pass `rng` to halftone.
- **Modified in place:** `ontology/face/print.yaml` — version bump only.
- **Modified in place:** `tests/golden/golden_hashes.json` — regenerated.
- **New tests** under `pad-synth-face/tests/`: at minimum, `test_print_halftone_jitter.py` asserting (a) two different RNG states produce different halftone outputs (the load-bearing invariant — without this the watermark survives), (b) same RNG state produces identical output (determinism), and (c) jittered cells still scale with `print_dpi` on average (the v2 DPI-scaling test must still pass for typical samples, accepting some per-sample noise).
- **Existing tests** `test_print_halftone.py` (v2's 5 tests): some MAY need adjustment since the determinism test calls `_apply_halftone` without an rng — keep the no-rng path deterministic (defaulted to no-jitter); the existing tests then continue to pass unchanged. The DPI scaling test (`test_halftone_changes_dot_count_with_dpi`) operates with no rng, so it tests the deterministic baseline only — that's fine.
- **Existing tests** `test_print_v2_integration.py` (v2's 5 tests): keep the existing assertions. The "low DPI has more dot structure than high DPI" test should still pass on average — but with jitter, single-sample comparisons could be noisy. If that test becomes flaky, use a deterministic test rng (e.g., `sample_rng(7)`) so the assertion remains stable.
- **New configs (×6)** under `configs/runs/v21_*.yaml`.
- **Append** updates to the existing results report + decisions/roadmap.

ICC code, paper-texture, perspective warp, cutout, replay attack, all `pad-synth-core` modules — unchanged.

## 9. Explicit non-goals

- **No** new ontology axes (jitter is algorithm-level).
- **No** changes to `_apply_icc` or the ICC parameter table.
- **No** changes to `_PAPER_TINTS`, `_paper_texture`, `_perspective_warp`, `_apply_cutout`.
- **No** new attack types (mask attack remains a separate sub-project).
- **No** real-data integration (still Phase 2.5+).
- **No** replay-attack changes.
- **No** dot-shape categorical jitter (deferred to v2.2 if v2.1's three jitter sources don't break the watermark).
- **No** dot-gain noise added — the existing `_paper_texture` step already provides multiplicative noise.
- **No** `pipeline.py` ontology_version hardcode fix (still deferred — see v2 report's note).
- **No** D4 regeneration (D4 is the plateau region; v2.1 measurement uses D1–D3 only, identical scope to v2).

## 10. Success criteria

- New jitter unit tests pass; the existing v2 tests in `test_print_halftone.py` and `test_print_v2_integration.py` still pass (no-rng default preserves determinism for tests that don't pass an rng).
- After golden regen, the full suite returns to its previous green count (154 passed, 1 skipped).
- Two v2.1 print attacks at the same `print_dpi` but different per-sample seeds produce **byte-different output** (the watermark-breaking test).
- v2.1 27-cell sweep: **no cell hits 0.000 cross-domain EER** (watermark broken).
- At least one D3 cell shows v2.1 cross-domain EER ≤ v1 EER by ≥ 0.05 with non-overlapping ±1σ bands (the v2 gain survives jitter).
- Report committed with v1 / v2 / v2.1 three-way comparison and a verdict; one-line decisions/roadmap update.
- `defid-pkg`, `defid-demo-pkg`, `pad-synth-core/src`, `pad-synth-face/src/pad_synth_face/{sensor,bonafide,pipeline,cli}.py`, `replay.py`, `base.py` — provably unmodified by branch diff.
