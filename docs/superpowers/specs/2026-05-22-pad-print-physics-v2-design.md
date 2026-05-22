% PAD Print Attack — Physics v2 Design
% Halftoning + ICC profile simulation; the first lever after the data axis plateaued
% 2026-05-22

---

## 1. Purpose and audience

The D4 sweep (just merged) showed the data axis plateaued at D3 cross-domain EER ≈ 0.20–0.26 across all model capacities. The diagnosis: the **synthetic generator's distribution is the binding constraint, not its scale.** This spec is the first physics-axis intervention: upgrade the print attack to v2 by adding two missing physics layers that the original `print.py` docstring explicitly deferred — **halftoning** (printer dot-pattern simulation) and **ICC profile simulation** (printer→paper color-space transform).

Audience: future maintainers, the Phase 2 author. This is the first of the hybrid Phase 2 sub-projects; the **mask attack module is a separately scoped follow-up** that does not block this work.

## 2. The question this answers

Does physics improvement move the cross-domain EER number that data scaling could not?

| v2 cross-domain EER at D3 vs v1 cross-domain EER at D3 (mean ± std, 3 seeds) | Diagnosis | Phase 2 implication |
|---|---|---|
| **Drops ≥ 0.05** with non-overlapping ±1σ bands | Physics IS the lever. Confirms the hybrid recommendation. | Proceed to mask-attack sub-project; revisit data scaling on v2 physics. |
| **Flat** (Δ < 0.05 or overlapping bands) | Physics is not enough on its own at this scale | Escalate: real-data integration becomes the dominant lever; mask attack may still help but is no longer expected to be transformative. |
| **Rises** | v2 introduces an artifact the detector latches onto on Set A but not Set B | Stop and audit the new physics; do not ship as the production print attack. |

Same `≥ 0.05 with non-overlapping ±1σ bands` rule as the parent sweep spec §2.

## 3. Halftoning

**Approach:** per-channel AM (amplitude-modulated) screening at the standard print rosette angles. RGB → CMYK conversion (simple math, no real profile), threshold each channel against a rotated dot screen at a frequency tied to `print_dpi`, recombine to RGB. The canonical 4-color offset/laser-print process.

### 3.1 Concrete algorithm

1. Compute K (black) from min channel: `K = 1 − max(R, G, B)`; then `C = (1 − R − K) / (1 − K)` etc. (standard undercolor extraction; no profile-specific math).
2. For each of C, M, Y, K, threshold against a rotated cosine dot screen at angle `θ_c ∈ {15°, 75°, 0°, 45°}` respectively.
3. The dot-screen cell size is `cell_px = max(2, round(8 × 150 / print_dpi))`. On a 64×64 image this gives **8 px cells at 150 dpi** (chunky, visible rosette) and **2 px cells at 1200 dpi** (fine, near-invisible). Intermediate values: 300 dpi → 4 px; 600 dpi → 2 px.
4. The dot pattern is `screen[y, x] = 0.5 + 0.5 × cos(2π (x' / cell_px) × cos(θ_c) + 2π (y' / cell_px) × sin(θ_c))` (the rotated cosine grid); compare channel value to `screen` to produce a binary dot.
5. Recombine C, M, Y, K → RGB via the inverse of step 1.

The screen pattern is **deterministic** — no RNG calls during halftoning. The full pixel-grid screen is built once per (cell_px, angle) tuple.

### 3.2 Citations

- Roetling 1976, "Halftone method for printing reproduction" (the canonical AM-screening reference).
- Pereira et al., "LBP-TOP based countermeasure against face spoofing attacks," ACCV 2012 Workshops (already cited in the existing `print_dpi` axis — establishes halftone artifacts as a PAD-relevant signal).

## 4. ICC profile simulation

**Approach:** a parameterized print-profile transform keyed by the existing `paper_type` axis. **No real ICC files** — three small parameter sets approximating the published behavior of common print profiles. Models the color-space distortion a real printer→paper chain introduces, without dragging in `littlecms` or external profile data.

### 4.1 Concrete parameters per paper_type

Each `paper_type` selects a tuple `(gamut_compression: float, white_point_shift: (Δx, Δy), tone_gamma: float)`:

| paper_type | gamut_compression | white_point_shift (Δx, Δy) | tone_gamma | qualitative effect |
|---|---|---|---|---|
| matte | 0.12 | (+0.012, +0.008) | 1.10 | warm shift, more compression, slight darken |
| glossy | 0.05 | (+0.002, +0.001) | 0.95 | near-neutral, mild lift |
| photo | 0.03 | (−0.003, −0.002) | 0.92 | slight cool shift, minimal compression, mild lift |

### 4.2 Transform pipeline (per pixel, after halftoning)

The whole §4 transform operates in **sRGB float [0, 1]** — the parameter values in §4.1 are chosen for sRGB-space operation, not linear-light. Sufficient for the parameterized-approximation scope; converting to linear-light would require gamma round-trips that don't add fidelity at this approximation level.

1. **Gamut compression**: `rgb_out = (1 − c) · rgb_in + c · 0.5` where `c = gamut_compression × icc_profile_strength` (the strength axis from §5 scales the effect 0.5–1.0×).
2. **White-point shift** modeled as a small additive bias in sRGB: `rgb += (Δx · 0.5, Δy · 0.5, −(Δx + Δy) · 0.25)` (rough chromaticity-to-RGB approximation; clipped to [0, 1]).
3. **Tone curve** as a power: `rgb = rgb ** tone_gamma`.

The 3×3 matrix path is intentionally avoided in favor of the 3-parameter scalar tuple per paper_type — it's the smallest physics-defensible parameter set that captures gamut, white point, and tone independently. Adding a full matrix is a YAGNI follow-up if a future audit shows missing fidelity.

### 4.3 Citations

- Lukac & Plataniotis (eds.), *Color Image Processing: Methods and Applications*, ch. on printer color characterization, for the parameter ranges.
- Marini, Rizzi (2000), "A Computational Approach to Color Adaptation Effects" (white-point handling).

## 5. Ontology changes

**File:** `ontology/face/print.yaml`.

- `version` bumped from `"2026-05-11"` to `"2026-05-22"`.
- `print_dpi` axis: same values/weights/provenance as v1, but the **active comment** now removed (no longer informational — consumed by halftoning per §3).
- New axis `icc_profile_strength`:
  ```yaml
  icc_profile_strength:
    type: uniform
    low: 0.5
    high: 1.0
    provenance:
      paper: "Lukac & Plataniotis (eds.), Color Image Processing: Methods and Applications, CRC Press 2007"
      doi: null
      url: null
  ```
  Models inter-print variability of color-profile fidelity (some prints are well-calibrated, some drift).
- `paper_type`, `tilt_degrees`, `holder_present`, `cutout` axes: unchanged.

The ontology lint (mandatory `provenance` per axis) continues to apply.

## 6. Backwards compatibility — bump and forward

No config flag, no dual-path code, no `print_v2.py` sibling module. `pad-synth-face/src/pad_synth_face/attacks/print.py` is **modified in place**: the `PrintAttack` class's `__init__` and `sample`/`simulate` API are unchanged; only internals change (new helpers `_to_cmyk`, `_halftone_channel`, `_inv_cmyk`, `_icc_transform`). The ontology version bump is the single signal that the physics has changed.

**Consequences:**
- `tests/golden/golden_hashes.json` will need to be **regenerated** as part of this work. The regeneration is the canonical signal that physics+ontology changed. Document the regen in the commit message; treat it as deliberate.
- Existing committed datasets (`datasets/phase1_smoke/`, `datasets/phase15_setb/`, `datasets/spark_set*_d{1,2,3,4}/`) are not modified on disk. They remain v1 hashes; their `manifest.jsonl` rows already record `ontology_version: "2026-05-11"`. They are self-identifying as v1.
- The published Phase 1 in-domain EER (0.29) and Phase 1.5 cross-domain mean (0.39) are now "v1 physics" numbers. The Phase 2 measurement (§7) explicitly compares to those v1 numbers as the baseline.

## 7. Measurement plan

After implementation lands:

1. **Regenerate** the six Spark sweep datasets at D1–D3 only (skip D4 — the plateau was the trigger, and a 4-level v1-vs-v2 head-to-head doubles compute without adding much information). New datasets: `datasets/v2_seta_d{1,2,3}/` and `datasets/v2_setb_d{1,2,3}/` (separate names so v1 and v2 coexist on disk for direct comparison).
2. **Six new configs** under `configs/runs/`: `v2_seta_d{1,2,3}.yaml` and `v2_setb_d{1,2,3}.yaml`. Identical to the existing `spark_*` configs except `name`/`output` (the bumped ontology version is picked up automatically from the YAML file the configs point at). Same seeds as the v1 sweep (20260522 / 20260523). Because `icc_profile_strength` is appended at the *end* of the axes list, all pre-existing axes (`paper_type`, `print_dpi`, `tilt_degrees`, `holder_present`, `cutout`) draw the same values as v1 — the only differences are (a) the new physics interprets `print_dpi` differently (now active for halftoning), and (b) one new appended RNG draw for `icc_profile_strength`.
3. **rsync** the 6 v2 datasets to the Spark.
4. **Run a 27-cell sweep** on the Spark using the v2 datasets: 3 capacities × 3 data levels × 3 seeds, same hyperparameters as the parent (10 epochs, batch_size 32, `--device cuda`).
5. **rsync results back**, regenerate combined CSV (now 27 v1 + 27 v2 = 54 rows OR a separate v2 summary; whichever the plan picks). Append a "**v2 physics result**" section to the existing results report with the v1 vs v2 cross-domain EER comparison per cell and the diagnosis from §2.
6. Append a one-line update to the decisions/roadmap doc.

The v2 sweep is the deliverable that decides whether physics moves the needle.

## 8. Architecture / component boundaries

- **Modified in place:** `pad-synth-face/src/pad_synth_face/attacks/print.py` — new internal helpers; public class API unchanged.
- **Modified in place:** `ontology/face/print.yaml` — version bump + new axis.
- **Modified in place:** `tests/golden/golden_hashes.json` — regenerated.
- **New tests** under `pad-synth-face/tests/`: at minimum, `test_print_halftone.py` (the halftone screen tiles deterministically, dot count scales with print_dpi) and `test_print_icc.py` (the ICC transform's qualitative direction matches the table in §4.1).
- **New configs (×6)** under `configs/runs/` for the v2 measurement sweep.
- **Append-only** updates to the existing results report + decisions/roadmap.

No other files modified. `defid-pkg/`, `defid-demo-pkg/`, `pad-synth-core/`, `pad-synth-face/src/pad_synth_face/attacks/replay.py` — all untouched.

## 9. Explicit non-goals

- **No** anisotropic specular highlights (deferred follow-up — modest payoff, easy to add later).
- **No** mask attack module (separate next sub-project; brainstormed independently).
- **No** real-data integration / DigiFace-1M wiring.
- **No** changes to the replay attack physics.
- **No** new sensor presets.
- **No** D4 regeneration in the measurement sweep (D1–D3 head-to-head is enough; D4 is the plateau region, not the discriminative scale).
- **No** real ICC profile files / `littlecms` dependency — the 3-parameter approximation is the deliberate scope.
- **No** v1/v2 dual-mode code path.

## 10. Success criteria

- `pad-synth-face/tests/test_print_halftone.py` and `test_print_icc.py` pass; existing tests still pass after golden regen.
- Generated v2 print attacks visibly exhibit halftone dot structure at low DPI and smooth out at high DPI (a one-line eyeball check in the test, e.g., the per-pixel autocorrelation peak at the screen frequency).
- The 27-cell v2 Spark sweep completes; 27 result JSONs land in a new directory under `docs/superpowers/reports/<date>-pad-print-v2-results/`.
- The report has a populated v1-vs-v2 cross-domain heatmap and a written verdict per the §2 rule.
- One-line decisions/roadmap update with the verdict.
- The whole feature branch's `git diff main -- defid-pkg defid-demo-pkg pad-synth-core/src pad-synth-face/src/pad_synth_face/attacks/replay.py pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/src/pad_synth_face/pipeline.py pad-synth-face/src/pad_synth_face/bonafide.py` is **empty** — only `print.py`, the ontology YAML, the golden hashes JSON, configs, tests, and reports change.
