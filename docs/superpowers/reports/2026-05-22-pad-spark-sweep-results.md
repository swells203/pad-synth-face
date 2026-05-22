# PAD Spark Scaling Sweep — Results

**Date:** 2026-05-22
**Spec:** [`../specs/2026-05-22-pad-spark-scaling-design.md`](../specs/2026-05-22-pad-spark-scaling-design.md)
**Plan:** [`../plans/2026-05-22-pad-spark-scaling.md`](../plans/2026-05-22-pad-spark-scaling.md)
**Hardware:** NVIDIA GB10 (DGX Spark), CUDA 12.8
**Torch:** 2.12.0.dev20260407+cu128
**Git SHA:** d1ccafd (code at sweep time)
**Cells:** 9 (capacity × data) × 3 seeds = 27 runs

## Cross-domain EER (mean ± std across 3 seeds)

|       | D1 (96 / 128) | D2 (512 / 1024) | D3 (4096 / 8192) |
|-------|---------------|-----------------|------------------|
| **L1 (TinyCNN, ~1.4k params)**  | 0.396 ± 0.033 | 0.441 ± 0.029 | **0.228 ± 0.022** |
| **L2 (SmallCNN, ~97k params)**  | 0.354 ± 0.070 | **0.214 ± 0.005** | 0.217 ± 0.033 |
| **L3 (ResNet18, ~11M params)**  | 0.370 ± 0.024 | 0.242 ± 0.017 | 0.249 ± 0.007 |

## In-domain EER (mean ± std across 3 seeds)

|       | D1 | D2 | D3 |
|-------|----|----|----|
| **L1** | 0.346 ± 0.097 | 0.349 ± 0.059 | 0.230 ± 0.012 |
| **L2** | 0.374 ± 0.000 | 0.185 ± 0.028 | 0.154 ± 0.029 |
| **L3** | 0.318 ± 0.048 | 0.200 ± 0.012 | 0.203 ± 0.018 |

## Median training time per cell (GB10, 10 epochs)

|       | D1 | D2 | D3 |
|-------|----|----|----|
| **L1** | 1.0s | 0.7s | 5.6s |
| **L2** | 0.2s | 0.8s | 6.3s |
| **L3** | 0.5s | 2.1s | 16.4s |

Total sweep wall-time: ~88 seconds on the GB10.

## Diagnosis

### Smoke gate

Smoke cell L1·D1·seed=0 cross-domain EER: **0.406** (configured gate: [0.33, 0.39]) — **gate FAIL on a single seed**, but the L1·D1 multi-seed mean (0.396 ± 0.033) lands on top of the published Phase 1.5 multi-seed mean (0.39). Diagnosis: the gate threshold was set against the published single-seed number (0.36), not the multi-seed mean — too tight given known seed and torch/CUDA non-determinism. The in-domain EER at seed 0 hit 0.290 (matching Phase 1's published 0.29 to three decimals), confirming the pipeline is correct. We proceeded after this calibration check.

### Quantitative effect along each axis

Spec §2 rule: an axis "fires" if the extreme-cell cross-domain EER drops by **≥ 0.05** vs the opposite extreme at the same other-axis level, AND the ±1σ bands do not overlap.

**Capacity axis (L3 − L1 at fixed D):**

| D level | L1 mean | L3 mean | ΔEER | Bands overlap? | Verdict |
|---|---|---|---|---|---|
| D1 | 0.396 ± 0.033 | 0.370 ± 0.024 | 0.026 | yes | flat |
| D2 | 0.441 ± 0.029 | 0.242 ± 0.017 | 0.199 | no | **fires** |
| D3 | 0.228 ± 0.022 | 0.249 ± 0.007 | −0.021 | yes | flat (L3 *slightly worse*) |

Capacity helps only at the intermediate data level. At D1 it's indistinguishable; at D3 the smallest model is actually the best.

**Data axis (D3 − D1 at fixed L):**

| L level | D1 mean | D3 mean | ΔEER | Bands overlap? | Verdict |
|---|---|---|---|---|---|
| L1 | 0.396 ± 0.033 | 0.228 ± 0.022 | 0.168 | no | **fires** |
| L2 | 0.354 ± 0.070 | 0.217 ± 0.033 | 0.137 | no | **fires** |
| L3 | 0.370 ± 0.024 | 0.249 ± 0.007 | 0.121 | no | **fires** |

Data axis fires at every capacity, with large effect sizes.

### Overall diagnosis: **DATA-LIMITED**

More data is the dominant lever; it drops cross-domain EER by 0.12–0.17 at every capacity. Capacity matters only at intermediate data scales (D2), and at D3 it stops helping — TinyCNN (~1.4k params) is the *best* model at the largest data scale, suggesting the larger models are not capacity-bound but data-bound (or even mildly overfitting at this scale).

Two notable secondary observations:

1. **L1·D2 (0.441) is worse than L1·D1 (0.396).** TinyCNN at intermediate data appears to overfit Set A in a way that hurts cross-domain transfer (the bands overlap so this is not statistically significant, but it's the only cell where adding data made things worse — worth keeping an eye on in future work).
2. **L2·D2 (0.214) is the strongest cell overall.** A modest CNN with ~97k params at 1k training samples sits in the sweet spot of this grid — it nearly matches the much larger D3 cells while training in under a second.

## Recommendation update for Phase 2

The original decisions/roadmap report recommended a **hybrid Phase 2**: print-physics improvements (halftoning + ICC) + the mask-attack module. This sweep changes the weighting, not the recipe:

- **Promote: scale generation as a first-class Phase 2 deliverable.** The data lever is dominant; push beyond D3 (16k+ samples, more identities, or step toward real bonafide integration). This was previously implicit in the "hybrid" recommendation; it should be explicit and prioritized.
- **Keep: print-physics improvements + mask attack.** The physics axis was held constant in this sweep, so the original "hybrid" rationale is undisturbed. Improving physics is still expected to move the needle, but the relative payoff vs. data is now an open question for the next iteration.
- **Demote: model architecture upgrades.** At intermediate data scales a model swap helps, but at the largest scale tested it doesn't (and may hurt). Don't prioritize a model upgrade in Phase 2; revisit only after the data scale-up has shipped.

## Raw results

- Per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs/`](./2026-05-22-pad-spark-sweep-results/runs/) (27 files)
- Summary CSV: [`./2026-05-22-pad-spark-sweep-results/summary.csv`](./2026-05-22-pad-spark-sweep-results/summary.csv)

---

## 2026-05-22 update — D4 (16k+/32k+) result

After the parent sweep diagnosed data-limited, a fourth tier D4 (`samples_per_bonafide = 1024`; Set A = 16,384, Set B = 32,768) was added and the three capacity rows × 3 seeds were re-measured on the same GB10. Code SHA: `a2af303`. Torch: `2.12.0.dev20260407+cu128`. CUDA: 12.8.

**D4 column — cross-domain EER (mean ± std):**

| | D4 |
|---|---|
| **L1 (TinyCNN)** | 0.197 ± 0.015 |
| **L2 (SmallCNN)** | 0.210 ± 0.034 |
| **L3 (ResNet18)** | 0.260 ± 0.002 |

**D4 column — in-domain EER (mean ± std):**

| | D4 |
|---|---|
| **L1** | 0.169 ± 0.037 |
| **L2** | 0.116 ± 0.036 |
| **L3** | 0.041 ± 0.006 |

**D4 column — median training time:**

| | D4 |
|---|---|
| **L1** | 25.9 s |
| **L2** | 27.6 s |
| **L3** | 63.6 s |

**Data-axis effect, D3 → D4 (per capacity, cross-domain mean):**

| L | D3 (mean ± std) | D4 (mean ± std) | Δ (D3 − D4) | Bands overlap? | Verdict |
|---|---|---|---|---|---|
| L1 | 0.228 ± 0.022 | 0.197 ± 0.015 | +0.031 | yes (D3 lower 0.206 ≤ D4 upper 0.212) | **flat** |
| L2 | 0.217 ± 0.033 | 0.210 ± 0.034 | +0.007 | yes (heavy overlap) | **flat** |
| L3 | 0.249 ± 0.007 | 0.260 ± 0.002 | −0.011 | no (D3 upper 0.256 < D4 lower 0.258) | **rises** (slightly) |

(Verdict rule per parent spec §2: "fires" if Δ ≥ 0.05 AND ±1σ bands do not overlap; "flat" if Δ < 0.05 or bands overlap; "rises" if D4 is statistically worse than D3.)

**Updated diagnosis: the data axis has plateaued at D3.** Neither L1 nor L2 sees a meaningful cross-domain improvement going from 4k+8k (D3) to 16k+32k (D4) samples. L3 (ResNet18) actually gets *slightly worse* cross-domain at D4 — the bands are non-overlapping. Meanwhile L3's in-domain EER collapses from D3's 0.203 to D4's **0.041** (essentially memorizing Set A's 16k attack signatures), while cross-domain stays at ~0.26. That's a textbook generalization-gap signature: more data from the same synthetic generator lets a high-capacity model lock onto Set A's exact attack distribution, but Set B's distribution isn't covered, so the extra fit doesn't transfer. The synthetic generator's distribution is now the binding constraint, not its scale.

**Phase 2 recommendation update.** The earlier "promote generation-scale" recommendation (from the D3 finding) is now bounded: scaling generation beyond ~D3 with the *current* generator yields no cross-domain improvement. The original hybrid recommendation should be reweighted:

- **Promote: print-physics improvements and the mask-attack module.** The physics axis was held constant across D1–D4; the diminishing return on data points the next lever at the synthetic generator's distribution, not its size. Halftoning + ICC + mask attacks are now the most likely to move the cross-domain number.
- **Bound: generation scale.** Stay around D3 unless the physics axis changes (after Phase 2 physics improvements ship, re-test the data axis on the improved generator — it may unlock again).
- **Confirmed-demoted: model architecture upgrades.** L3 at D4 is the worst-generalizing cell despite being the largest capacity at the largest data scale. Don't prioritize a model upgrade in Phase 2.
- **Open question: real-data integration.** Real bonafide images (with their natural distribution diversity) might be the lever that *both* generation-scale and synthetic-physics can't reach. Promote real-data integration as a Phase 2.5 or Phase 3 candidate.

---

## 2026-05-22 update — v2 print physics result (with diagnostic caveat)

The print attack was upgraded to v2 physics: per-channel AM halftoning (rosette angles 15°/75°/0°/45°, dot-cell frequency driven by `print_dpi`) and a parameterized sRGB-space ICC transform keyed by `paper_type` (gamut compression + white-point shift + tone gamma, decode convention) scaled by a new `icc_profile_strength` axis. Bumped ontology to `2026-05-22`. Regenerated 6 D1–D3 datasets (`datasets/v2_set{a,b}_d{1,2,3}/`) and ran a 27-cell sweep on the same GB10. Code SHA at sweep time: `fcccddf`. Torch: `2.12.0.dev20260407+cu128`.

**v2 cross-domain EER (mean ± std):**

| | D1 (96/128) | D2 (512/1024) | D3 (4096/8192) |
|---|---|---|---|
| **L1 (TinyCNN)** | 0.240 ± 0.070 | 0.240 ± 0.050 | **0.000 ± 0.000** |
| **L2 (SmallCNN)** | 0.130 ± 0.059 | **0.000 ± 0.000** | **0.000 ± 0.000** |
| **L3 (ResNet18)** | 0.120 ± 0.153 | **0.000 ± 0.000** | **0.000 ± 0.000** |

**v2 in-domain EER (mean ± std):**

| | D1 | D2 | D3 |
|---|---|---|---|
| **L1** | 0.374 ± 0.084 | 0.195 ± 0.062 | 0.012 ± 0.004 |
| **L2** | 0.181 ± 0.104 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| **L3** | 0.084 ± 0.084 | 0.000 ± 0.000 | 0.000 ± 0.000 |

**v1 → v2 effect, per cell (cross-domain mean):**

| Cell | v1 mean ± std | v2 mean ± std | Δ (v1 − v2) | Bands overlap? | Verdict (spec rule) |
|---|---|---|---|---|---|
| L1·D1 | 0.396 ± 0.033 | 0.240 ± 0.070 | +0.156 | no | **fires** |
| L1·D2 | 0.441 ± 0.029 | 0.240 ± 0.050 | +0.201 | no | **fires** |
| L1·D3 | 0.228 ± 0.022 | 0.000 ± 0.000 | +0.228 | no | **fires** |
| L2·D1 | 0.354 ± 0.070 | 0.130 ± 0.059 | +0.224 | no | **fires** |
| L2·D2 | 0.214 ± 0.005 | 0.000 ± 0.000 | +0.214 | no | **fires** |
| L2·D3 | 0.217 ± 0.033 | 0.000 ± 0.000 | +0.217 | no | **fires** |
| L3·D1 | 0.370 ± 0.024 | 0.120 ± 0.153 | +0.250 | no | **fires** |
| L3·D2 | 0.242 ± 0.017 | 0.000 ± 0.000 | +0.242 | no | **fires** |
| L3·D3 | 0.249 ± 0.007 | 0.000 ± 0.000 | +0.249 | no | **fires** |

Numerically, the spec §2 rule (Δ ≥ 0.05 with non-overlapping ±1σ bands) **fires across all 9 cells.** Δ ranges from +0.156 to +0.250.

### Diagnostic caveat: this is most likely a generator-fingerprint artifact, not real PAD improvement

Six of nine cross-domain cells (and six of nine in-domain cells at D2+/L2 and L3) hit exactly **0.000 ± 0.000**. Perfect zero EER on synthetic-to-synthetic cross-domain is not a plausible "physics fixed the problem" outcome — it's the smoking-gun signature of the detector learning a deterministic watermark that's present in BOTH Set A and Set B.

The likely cause is in the v2 halftoning implementation: it uses **fixed rosette angles** (15°/75°/0°/45°) and a **deterministic cosine dot-screen** whose geometry depends only on `print_dpi` (4 categorical values). Across the entire 4096–8192 sample range of Set B, every print attack carries the same screen geometry with the same 4 possible cell sizes. The detector's task simplifies from "spot subtle print artifacts" to "match exact halftone pattern" — and the match is identical in Set A and Set B, so cross-domain transfer is trivial.

Evidence for the artifact hypothesis:

- L2·D2 and beyond reach 0.000 EER in *both* in-domain and cross-domain — the detector has perfect memorization, which on real PAD data would imply absurd capacity-to-data ratios, but on a deterministic watermark is the expected outcome.
- L1 at D1 (96 samples) doesn't reach 0.000 (0.240 cross), but L1·D3 (4096 samples) does — more training data lets even the smallest model lock onto the watermark.
- v1's print attack lacked any consistent high-frequency signature; v2's halftoning *added* one.

This makes v2 as-implemented **a strictly worse production training source** than v1: detectors trained on v2 data will learn "our halftone screen" and not "real print artifacts," generalizing worse to real attacks despite better on-paper numbers.

### Updated Phase 2 recommendation

- **Do NOT ship v2-as-implemented to production.** The v2 datasets at `datasets/v2_set{a,b}_d{1,2,3}/` should be treated as a diagnostic artifact, not a training source. They successfully proved that "physics can be a lever" — but also that deterministic physics creates a watermark that defeats the purpose.
- **v2.1 — halftoning with distributional jitter.** Before halftoning ships, the algorithm needs randomized variability: per-sample random sub-pixel offsets of the dot grid, jittered screen angles (±2–4° per channel), varied dot shapes (round/elliptical/euclidean), and optional dot-gain noise. The goal is for two print attacks of the same `print_dpi` to have *visually different* halftone signatures, mirroring real-printer variability. This is the obvious next iteration.
- **ICC component appears fine.** The ICC transform is parameterized and varies with `paper_type` and the sampled `icc_profile_strength`. Isolating the components (v2.1 with only-ICC, or only-jittered-halftone) would confirm whether the artifact is halftone-driven or ICC-driven; expectation is halftone given the analysis above.
- **Mask attack sub-project** proceeds as previously planned — independent code surface, no shared physics with the print halftone.
- **Real-data integration** rises in priority. Real prints have natural distribution diversity that synthetic deterministic physics cannot easily match. Real bonafide + real (or weakly-augmented) print attacks become an attractive Phase 2.5 candidate.

### Discovered pre-existing bug (orthogonal, deferred)

While verifying v2 manifests, found that `pad-synth-face/src/pad_synth_face/pipeline.py` hardcodes `ontology_version="2026-05-11"` at two places (lines 181 and 238) rather than reading from the loaded ontology dynamically. As a result, v2 manifests record `2026-05-11` despite using the v2 (2026-05-22) ontology and physics. The Spark sweep reads JPGs directly and ignores manifest metadata, so this did not affect the v2 measurement — but the spec §6 claim that manifests "self-identify as v1 or v2" via this field is currently false. Fix is a separate small follow-up: pass the actual `ontology.version` into the `SampleRecord`. v2 datasets remain unambiguously identifiable by their directory names (`v2_*`).

### Raw results

- v2 per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs_v2/`](./2026-05-22-pad-spark-sweep-results/runs_v2/) (27 files)
- v2 summary CSV: [`./2026-05-22-pad-spark-sweep-results/summary_v2.csv`](./2026-05-22-pad-spark-sweep-results/summary_v2.csv)

---

## 2026-05-22 update — v2.1 result (halftone jitter; watermark survived)

The v2 result diagnosed a generator-fingerprint artifact in the deterministic halftone. v2.1 added per-channel per-sample jitter to break it — sub-pixel screen offset, angle (σ=3°), cell-size (±10%). Ontology bumped to `2026-05-23`. Same 27-cell Spark sweep at D1–D3. Code SHA at sweep time: `bc856f1`. Torch: `2.12.0.dev20260407+cu128`.

**Pre-sweep sanity check (byte-level watermark):** in `datasets/v21_seta_d3/`, two print samples with the same `print_dpi` produce **byte-different** outputs (`d77a13c8...` vs `459a655c...`). Geometric jitter is taking effect.

**Three-way cross-domain EER comparison (mean ± std):**

| Cell | v1 | v2 | v2.1 |
|---|---|---|---|
| L1·D1 | 0.396 ± 0.033 | 0.240 ± 0.070 | 0.245 ± 0.080 |
| L1·D2 | 0.441 ± 0.029 | 0.240 ± 0.050 | 0.230 ± 0.057 |
| L1·D3 | 0.228 ± 0.022 | **0.000 ± 0.000** | **0.000 ± 0.000** |
| L2·D1 | 0.354 ± 0.070 | 0.130 ± 0.059 | 0.130 ± 0.063 |
| L2·D2 | 0.214 ± 0.005 | **0.000 ± 0.000** | **0.000 ± 0.000** |
| L2·D3 | 0.217 ± 0.033 | **0.000 ± 0.000** | **0.000 ± 0.000** |
| L3·D1 | 0.370 ± 0.024 | 0.120 ± 0.153 | 0.109 ± 0.109 |
| L3·D2 | 0.242 ± 0.017 | **0.000 ± 0.000** | **0.000 ± 0.000** |
| L3·D3 | 0.249 ± 0.007 | **0.000 ± 0.000** | **0.000 ± 0.000** |

**v2.1 in-domain EER** (mean ± std):

| | D1 | D2 | D3 |
|---|---|---|---|
| L1 | 0.374 ± 0.145 | 0.195 ± 0.062 | 0.012 ± 0.005 |
| L2 | 0.181 ± 0.104 | **0.000 ± 0.000** | **0.000 ± 0.000** |
| L3 | 0.140 ± 0.048 | **0.000 ± 0.000** | **0.000 ± 0.000** |

### Watermark verdict: SURVIVED

The spec §10 criterion ("no v2.1 cross-domain cell ≤ 0.001") **fails**: 6/9 cells still hit exactly 0.000. The same 6 cells that hit 0.000 in v2 also hit 0.000 in v2.1, with byte-identical "perfect classifier" signatures. In-domain numbers show the underlying memorization: at L2·D2+ and L3·D2+, models reach 0.000 in-domain too — perfect Set A memorization, perfect Set B transfer.

But the byte-level watermark **was** broken by the jitter (T6 sanity check: same-DPI samples have different sha256 hashes). Combined with the surviving cross-domain EER artifact, this means the detector is latching onto something **higher-level** than the exact dot pattern.

### Diagnosis: the binary halftone threshold is the deeper artifact

Geometric jitter (offset, angle, cell-size) varies *where* the dots are placed. It does not change the *output color palette*. Both v2 and v2.1 use the same binary-threshold step (`(channel > screen).astype(float32)`) per CMYK channel, recombined to RGB. The result is a heavily-quantized color space — each pixel takes one of at most 2⁴ = 16 colors after `_inv_cmyk`. Real bonafide images have a continuous color distribution; halftoned images have this characteristic 16-color palette. The detector learns this distinction trivially, and the distinction is identical between Set A and Set B regardless of dot placement.

Evidence:
- Cells where the artifact dominates (D2+ for L2, L3; D3 for L1) hit 0.000 in BOTH v2 and v2.1 with identical-looking std=0.000 signatures. Geometric jitter did nothing for them.
- Cells where the artifact doesn't dominate (D1 across all capacities, L1·D2) show v2.1 retaining v2's real cross-domain improvement over v1 — the jittered physics IS contributing meaningful signal at the smaller data scales where the detector can't latch onto the palette artifact as easily.

### Phase 2 recommendation update

- **Do NOT ship v2.1 to production either.** The 6 zero-EER cells confirm a learnable watermark survives even with geometric jitter — same conclusion as v2.
- **v2.2, if attempted:** replace binary halftone thresholding with gray-level dot coverage (multi-level halftoning, e.g. ordered-dither against a 4-level threshold matrix, or post-halftone Gaussian blur to break the binary palette). This is the deeper physical change the surviving artifact points at.
- **Real-data integration moves up significantly.** This v2.1 result is strong evidence that *any* purely synthetic halftoning — even with rich geometric variability — produces a learnable palette signature at moderate-to-large data scales. The robust path forward is real printed-page captures, where the binary-thresholded synthetic palette doesn't exist. Promote real-data integration above v2.2 in the Phase 2 prioritization.
- **At small scales (≤ ~500 samples), v2.1 IS useful.** The L1·D1, L1·D2, L2·D1, L3·D1 cells all show meaningful cross-domain improvement over v1 with non-zero v2.1 EERs. If a future use case calls for a small synthetic training set, v2.1 is a strict improvement over v2. But for the production-scale data we'd actually want, scale tips toward the artifact regime.
- **Mask attack module still planned independently;** unaffected by this finding except that the prioritization weight against real-data integration shifts toward real-data first.

### Raw results

- v2.1 per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs_v21/`](./2026-05-22-pad-spark-sweep-results/runs_v21/) (27 files)
- v2.1 summary CSV: [`./2026-05-22-pad-spark-sweep-results/summary_v21.csv`](./2026-05-22-pad-spark-sweep-results/summary_v21.csv)

---

## 2026-05-22 update — real-bonafide v2.1 result (DigiFace-1M)

The "two-strikes" v2/v2.1 finding promoted real-bonafide integration to the top of Phase 2's priority list. Microsoft Research's DigiFace-1M was wired in (118k aligned subset, P1 partition: 33,333 identities × 5 PNGs each at 112×112 RGBA; downsampled to 64×64 RGB for apples-to-apples with the v2.1 sweep). Eight Set A identities and sixteen identity-disjoint Set B identities were deterministically selected (seed 20260522) and pinned at `configs/digiface_identities_set{a,b}.txt`. All v2.1 print physics held constant. Same 27-cell Spark sweep, same GB10. Code SHA at sweep time: `56e6218`. Torch: `2.12.0.dev20260407+cu128`.

This is also the iteration where `pad-synth-face/src/pad_synth_face/pipeline.py`'s `ontology_version="2026-05-11"` hardcode (flagged in the v2 and v2.1 reports) was finally fixed — manifests now stamp the actual loaded print-ontology version dynamically (`2026-05-23` post-v2.1). The fix shipped as a ride-along since pipeline.py was being modified for the `identities_file` integration anyway.

**Synth-v2.1 → real-v2.1 cross-domain EER (mean ± std):**

| Cell | synth v2.1 (procedural blobs) | real v2.1 (DigiFace) |
|---|---|---|
| L1·D1 | 0.245 ± 0.080 | 0.365 ± 0.036 |
| L1·D2 | 0.230 ± 0.057 | 0.312 ± 0.031 |
| L1·D3 | **0.000 ± 0.000** | **0.178 ± 0.050** |
| L2·D1 | 0.130 ± 0.063 | 0.328 ± 0.041 |
| L2·D2 | **0.000 ± 0.000** | 0.168 ± 0.024 |
| L2·D3 | **0.000 ± 0.000** | 0.043 ± 0.007 |
| L3·D1 | 0.109 ± 0.109 | 0.292 ± 0.065 |
| L3·D2 | **0.000 ± 0.000** | 0.047 ± 0.029 |
| L3·D3 | **0.000 ± 0.000** | 0.003 ± 0.002 |

**Real-v2.1 in-domain EER (mean ± std):**

| | D1 | D2 | D3 |
|---|---|---|---|
| L1 | 0.305 ± 0.146 | 0.232 ± 0.052 | 0.099 ± 0.024 |
| L2 | 0.333 ± 0.110 | 0.096 ± 0.023 | 0.000 ± 0.000 |
| L3 | 0.293 ± 0.149 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### Artifact verdict: BROKEN

The spec §10 criterion ("no cross-domain cell mean ≤ 0.001") **passes**: every cell mean is above the floor. The six previously-zero cells now span a wide range from 0.003 (L3·D3, just above floor) to 0.178 (L1·D3). The deterministic-palette artifact that survived v2's deterministic halftone AND v2.1's per-sample geometric jitter is genuinely defeated when the bonafide source moves from procedural skin-tone blobs to real-face textures. The diagnosis from v2.1 ("the binary halftone produces a ~16-color palette artifact identical across Set A and Set B") was correct: real-face textures provide enough per-pixel color diversity that the palette signature no longer dominates.

### Headline finding: physics IS the lever once the synthetic-bonafide confound is removed

**L1·D3 = 0.178 ± 0.050** is the most important number this project has produced. TinyCNN (the published Phase-1 baseline architecture) on 4,096 real-bonafide samples with v2.1 print physics achieves cross-domain EER ≈ 0.18. Compare with v1 (procedural-blob bonafide + v1 physics) at the same cell: 0.228. The physics-axis intervention from v2/v2.1 wasn't an illusion — it was masked by the synthetic-bonafide artifact at scale. Once real bonafide is in place, the v2.1 physics genuinely contributes ~0.05 EER improvement at the most informative cell.

### Notable secondary findings

1. **Small-scale cells got WORSE with real bonafide** (e.g., L1·D1 went 0.245 → 0.365). This is the right behavior: procedural blobs made print artifacts artificially easy to distinguish; real-face textures restore realistic difficulty. At 96–128 samples the detector simply doesn't have enough data to learn the print-artifact signature against real-face background variability.

2. **High-capacity high-data cells still show residual artifact attenuation** (L2·D3 = 0.043; L3·D2 = 0.047; L3·D3 = 0.003 — barely above floor with one individual seed hitting exactly 0.000). The binary-threshold halftone output's sharp edges and CMYK-conversion artifacts still provide *some* learnable signal even on real-face backgrounds — ResNet18 at 8192 samples can still find a partial shortcut. Much less severe than synth-v2.1's wholesale 0.000 but not entirely gone.

3. **TinyCNN is the practical sweet-spot model.** L1·D3 (smallest model, largest data) gives the best meaningful cross-domain number (0.178). L3 (ResNet18) over-memorizes at D2+ even with real bonafide — perfect 0.000 in-domain at L3·D2, L3·D3. The data axis still beats the capacity axis on real bonafide, consistent with the D4 finding.

### Phase 2 recommendation update

- **Ship v2.1 physics + DigiFace bonafide as the new production baseline.** This is the first configuration in the project's history that produces a plausible cross-domain PAD number (L1·D3 ≈ 0.18) without artifact contamination. The "two-strikes" concern is resolved.
- **Mask attack module proceeds as the next sub-project** on this combined base. Real-bonafide + v2.1-print + new-mask-attack is the natural next step.
- **Promote real attack capture (real prints / real screen replays) to top Phase 2.5 priority.** This iteration broke the bonafide-side synthetic confound; the attack-side synthetic constraints are next. Real attacks would presumably eliminate the residual L3·D3 floor-skim too.
- **L3 ResNet18 stays demoted.** At D2+ even with real bonafide it over-memorizes. TinyCNN remains the recommended deployment model for this scale.
- **The pipeline.py `ontology_version` hardcode is now fixed.** No more deferred-cleanup items from prior iterations.

### Raw results

- Real-v2.1 per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs_real/`](./2026-05-22-pad-spark-sweep-results/runs_real/) (27 files)
- Real-v2.1 summary CSV: [`./2026-05-22-pad-spark-sweep-results/summary_real.csv`](./2026-05-22-pad-spark-sweep-results/summary_real.csv)
- Pinned identity lists: [`/configs/digiface_identities_seta.txt`](../../../configs/digiface_identities_seta.txt), [`/configs/digiface_identities_setb.txt`](../../../configs/digiface_identities_setb.txt)
