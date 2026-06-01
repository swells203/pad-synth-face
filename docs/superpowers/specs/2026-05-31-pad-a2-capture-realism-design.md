% PAD A2 — Capture-Domain Realism (sensor.py expansion) Design
% Replace `sensor.py`'s minimal post-attack chain with a physically-ordered capture pipeline that adds **radial lens distortion, motion blur, shot+read sensor noise, and a multi-pass JPEG recompression chain** — each rng-driven and per-sample jittered so no fingerprint is introduced. Default-on in the existing MOBILE/WEBCAM presets; single combined sweep vs the L4 baseline measures the delta.
% 2026-05-31

---

## 1. Purpose and audience

The 2026-05-30 B2 sweep (commit `67110ba`) gave the project its first deployable PAD configuration: pretrained ResNet18 at 224×224 on the v2.1+DigiFace+mask base, mask-only cross-domain EER 0.060 ± 0.012 (mix L4·D3 = 0.059 ± 0.015; ACER@5%APCER 6 %/11 %). That detector learns from richer 224×224 inputs but the **capture chain it's trained against is still minimal** — `sensor.py` today does only vignette / white-balance / single-Gaussian noise / single-pass JPEG. Real captures pass through camera optics (lens distortion), motion (handheld blur), physical sensor physics (shot + read noise), and a multi-stage compression chain (capture → upload → server re-encode). Synthetic samples that skip those layers leave the detector under-prepared for them.

This spec rebuilds `sensor.py`'s post-attack chain into a physically-ordered capture pipeline that adds four effects — **radial lens distortion, motion blur, shot+read sensor noise, multi-pass JPEG** — each rng-driven and per-sample jittered. The existing presets (`MOBILE_FRONT_2024`, `WEBCAM_1080P`) extend with the new parameter ranges and stay the only API surface; `apply_sensor(img, preset, rng) → (img, params)` is unchanged externally.

This is incremental on top of B2, not a replacement. The next sub-projects in the queue (`pad-next-sub-projects` memory) — DFDC sweep, B1 synth-pretrain→real-finetune curve, Tier-B real benchmark — all consume this new sensor as the production capture chain.

## 2. The question this answers

| Behaviour on the next 18-cell L4+A2 sweep | What it tells us |
|---|---|
| Cross-domain EER **drops** vs L4 baseline (mask 0.060, mix 0.059) | A2 adds learnable capture realism on top of B2. New production baseline → A2 ships, queue continues to DFDC / B1 / Tier-B. |
| Cross-domain EER **flat** vs L4 baseline | The detector already saturates on the existing sensor's signal; capture-realism doesn't unlock more *within synthetic*. Real value will only show under **synth→real** evaluation (which the DFDC / Tier-B sweeps test). A2 still ships as the production capture chain, but its impact is deferred to the real-data sweeps. |
| Cross-domain EER **worsens** | A new generator fingerprint was introduced (the v2/v2.1 trap, applied to the sensor layer). Apply the same anti-fingerprint playbook: identify which effect is too deterministic, jitter it harder, or remove it. |
| Any cell collapses to 0.000 | Catastrophic fingerprint at the sensor layer. Bisect across the 4 new effects; one of them is producing a learnable invariant across Set A / Set B. |

Decision rule (mirrors every prior sweep): `no cross-domain cell mean ≤ 0.001` = artifact-free.

## 3. Architecture and files

| Change | File | Note |
|---|---|---|
| Extend `SensorPreset` | `pad-synth-face/src/pad_synth_face/sensor.py` | Add `motion_blur_px_range: tuple[int,int]`, `jpeg_passes_range: tuple[int,int]`, `lens_k1_range: tuple[float,float]` |
| Extend the two presets | same file | `MOBILE_FRONT_2024` and `WEBCAM_1080P` gain default ranges (§4) |
| Replace `_noise` body | same file | Same signature `_noise(img, iso, rng) -> ndarray`; body becomes shot + read instead of pure Gaussian |
| New helper `_lens_distort` | same file | Radial (`k1`-only) forward warp via `cv2.remap`; per-sample `k1` |
| New helper `_motion_blur` | same file | Directional line kernel; per-sample length + angle; `cv2.filter2D` |
| New helper `_jpeg_chain` | same file | 1–3 encode→decode passes, per-pass jittered QF; replaces single-pass `_jpeg_roundtrip` in `apply_sensor` |
| Reorder `apply_sensor` | same file | Physical pipeline: lens → motion → noise → vignette → WB → JPEG-chain |
| Per-effect tests | `pad-synth-face/tests/test_sensor_a2.py` (new) | Determinism, jitter, range, non-no-op, integration via `apply_sensor` |
| Determinism golden | `tests/golden/golden_hashes.json` | Regenerated — sensor changes affect every sample |
| Run configs | `configs/runs/*.yaml` | **Unchanged** (sensor preset names are unchanged) |
| Sweep outputs | `runs_mask_224_L4_A2/` + `runs_mix_224_L4_A2/` | 18 L4 cells with the new sensor |
| Report | `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` | Append "2026-05-31 update — A2 capture-realism" section with L4+A2 vs L4-only comparison |

No changes to: `attacks/*.py`, `eval/baseline.py`, `eval/metrics.py`, `eval/models_zoo.py`, the L4 factory, ontologies, or `scripts/spark_sweep.py`.

## 4. Effect specifications

All effects are pure functions of `(img, params..., rng)`. Every per-sample parameter is drawn from `rng` inside `apply_sensor` and recorded in the returned `sensor_params` dict (existing pattern). Outputs are clipped to `[0, 255]` `uint8` at the end of `apply_sensor` (existing convention).

### 4.1 Radial lens distortion (`_lens_distort`)

Single-parameter radial distortion via the `k1` coefficient of the Brown-Conrady model. For a normalised radius `r`, the distorted radius is `r' = r * (1 + k1 * r²)`. `k1 > 0` is pincushion, `k1 < 0` is barrel.

Per-sample: `k1 ~ U(lens_k1_range)`. Apply via `cv2.remap` with a precomputed sample map.

**Preset defaults:**
- `MOBILE_FRONT_2024`: `lens_k1_range = (-0.10, 0.10)` (consumer mobile lenses are mildly barrel/pincushion-corrected; small residual).
- `WEBCAM_1080P`: `lens_k1_range = (-0.05, 0.05)` (webcams have wider FOV correction; smaller residual).

`k1 = 0` is the identity transform; the range straddles zero so ~50 % of samples are essentially undistorted (matches the real distribution where some lenses are well-corrected).

### 4.2 Motion blur (`_motion_blur`)

Directional line kernel of length `L` (px) at angle `θ` (rad). Real handheld captures have a directional motion-blur trail; a Gaussian blur is the wrong shape and gives away the synthesizer.

Per-sample: `L ~ U(motion_blur_px_range)` (integer; `L = 1` is identity), `θ ~ U(0, π)`. Build a `(L, L)` kernel with a 1-px-thick line at angle `θ`, normalise to sum 1. Apply via `cv2.filter2D`.

**Preset defaults:**
- `MOBILE_FRONT_2024`: `motion_blur_px_range = (1, 7)` (handheld phone selfies, often significant motion).
- `WEBCAM_1080P`: `motion_blur_px_range = (1, 4)` (tripod / fixed-mount, less motion).

Including `L = 1` in the range keeps the no-blur case in distribution — real captures aren't always blurry.

### 4.3 Shot + read noise (`_noise` replacement)

Replace the current pure-Gaussian `_noise` (signature `_noise(img, iso, rng) -> ndarray`) with a physically motivated shot + read model.

```
signal = img.astype(float32)
shot_sigma = sqrt(max(signal, 1.0)) * (iso / 800.0) * 0.5
shot = rng.normal(0.0, shot_sigma, size=signal.shape)
read = rng.normal(0.0, 1.5, size=signal.shape)
return clip(signal + shot + read, 0, 255).astype(uint8)
```

Shot noise (Poisson-approximated via signal-dependent Gaussian, `sigma = sqrt(signal)`) scales with ISO. Read noise is a small fixed-magnitude Gaussian (sensor electronics floor). No preset field changes — driven by the existing `iso_range`.

(Using a signal-dependent Gaussian rather than true Poisson is the standard simulator approximation — visually indistinguishable and ~10× faster than `rng.poisson` on a 224×224×3 array.)

### 4.4 JPEG recompression chain (`_jpeg_chain`)

Real social-media captures pass through multiple JPEG passes (capture → app encode → server re-encode → CDN re-encode). A single roundtrip leaves a distinguishable single-encode signature.

Per-sample: `n_passes ~ randint(jpeg_passes_range)` (1, 2, or 3 typically). For each pass: `qf ~ randint(jpeg_qf_range)` (drawn fresh per pass), encode + decode. Replaces the single `_jpeg_roundtrip` call in `apply_sensor`.

**Preset defaults:**
- `MOBILE_FRONT_2024`: `jpeg_passes_range = (1, 3)`.
- `WEBCAM_1080P`: `jpeg_passes_range = (1, 2)` (less of an upload-chain story than mobile).

`n_passes = 1` is the current behaviour — kept in distribution.

## 5. Pipeline order (`apply_sensor`)

```
attack-output uint8 (H, W, 3)
  → _lens_distort       (lens optics)
  → _motion_blur        (exposure-time motion)
  → _noise              (sensor: shot + read)
  → _vignette           (ISP: optical/sensor falloff)
  → _white_balance      (ISP: color temperature)
  → _jpeg_chain         (encode + recompression)
→ uint8 (H, W, 3) out + sensor_params dict
```

Ordering rationale: each effect simulates the physical stage at its real position in a camera pipeline. The current order (vignette → wb → noise → jpeg) gets reshuffled to put sensor effects before ISP processing; this is more physical and means noise sees the unprocessed signal levels (correct shot-noise behaviour).

`sensor_params` dict gains the new per-sample parameters: `lens_k1`, `motion_blur_L`, `motion_blur_theta`, `jpeg_passes`, `jpeg_qf_per_pass: list[int]`. Recorded in the manifest exactly like `iso`/`kelvin`/`qf` today.

## 6. Artifact discipline (the v2/v2.1 lesson, applied to the sensor layer)

Every new effect carries the discipline that the attack physics taught us:

- **No deterministic patterns.** Every per-sample parameter (`k1`, `L`, `θ`, `n_passes`, per-pass `qf`) is drawn from `rng` each sample. Two samples with the same preset must produce byte-different outputs.
- **No quantisation beyond JPEG's standard.** All intermediate math is `float32`; only the final `uint8` cast and JPEG's standard 8-bit roundtrip discretize. (`_noise`, `_lens_distort`, `_motion_blur` are continuous.)
- **Identity options stay in distribution.** `k1 = 0`, `L = 1`, `n_passes = 1` are all reachable so the "no-effect" case isn't excluded — important for sensor variance modelling and for not creating a "synthetic always has effect X" tell.
- **Byte-level sanity check** in the sweep prep: confirm two same-preset samples differ (existing pattern from prior sub-projects).
- **The new ISO ACER metric** (from the eval-metrics upgrade) catches any operating-point degradation that EER alone would hide.

## 7. Sweep + report

1. Locally regenerate all 12 `mask_*` / `mix_*` datasets at 224 with the new sensor.
2. Rsync code + datasets to the Spark.
3. Run 9 mask + 9 mix L4 cells (using `make_resnet18_pretrained`, same as 2026-05-30 sweep) with A2-enabled sensor → `runs_mask_224_L4_A2/` + `runs_mix_224_L4_A2/`.
4. Pull results back, compute mean±std + per-cell ΔEER vs the 2026-05-30 L4 baseline.
5. Append "2026-05-31 update — A2 capture-realism" report section with the headline numbers and the decision-matrix verdict from §2.

The L4 baseline (commit `67110ba`) stays immutable as the pre-A2 reference. No re-run of the L1/L2/L3 cells.

## 8. Testing

`pad-synth-face/tests/test_sensor_a2.py` (new):

- **Per-effect determinism** — same seed → byte-identical output (each helper).
- **Per-effect jitter** — different seeds → measurably different output (covers the anti-fingerprint invariant; this is the analogue of the mask anti-watermark test that has caught regressions before).
- **Per-effect shape/dtype/range** — `(H, W, 3) uint8`, values in `[0, 255]`.
- **Per-effect non-no-op** — output differs from input on a non-degenerate input (the existing degenerate-image guard handles all-uniform).
- **`apply_sensor` integration** — full chain on a noise-padded fixture with `MOBILE_FRONT_2024`; assert shape/dtype/QC-pass; assert `sensor_params` dict contains the 5 new keys.
- **Byte-level anti-watermark** — two `apply_sensor` calls with different seeds on the same input produce byte-different outputs.

Existing `test_sensor.py` continues to assert the contract (the new `apply_sensor` chain is a superset of the old one in terms of effects but the function signature and output shape are unchanged).

Determinism golden regenerated (`PAD_SYNTH_UPDATE_GOLDEN=1 pytest tests/test_determinism_golden.py`).

## 9. Compute

- **Local dataset regen:** ~10–15 min for the 12 datasets at 224 (the new effects are pure NumPy/cv2; cost scales linearly with samples).
- **Spark sweep:** ~3 min mask + ~4.5 min mix on GB10 (matches the 2026-05-30 L4 sweep timing — the model and resolution are identical, only inputs differ).
- **No new dependencies.** `cv2` already in deps (used by `attacks/print.py` and `attacks/replay.py`); `_motion_blur` and `_lens_distort` use `cv2.filter2D` / `cv2.remap` respectively.

## 10. Out of scope

- **Per-effect ablations** — single combined sweep this cycle. Ablation becomes a follow-up only if the combined delta is unclear or negative.
- **Replay recapture chain (display → camera response)** — replay-attack-specific; belongs in `attacks/replay.py`, not the general sensor. Deferred as a separate sub-project (A3).
- **Per-preset parameter tuning** — sensible defaults shipped; sweep over `motion_blur_px_range` upper bound (etc.) is YAGNI for the first measure.
- **Architecture changes** — L4 stays the model; this is purely a data-side change.
- **Re-running L1/L2/L3 cells** — those numbers are immutable in prior report sections.
- **Color/WB expansion** — existing `_white_balance` (Kelvin sweep) already covers it.
- **DFDC sweep / B1 finetune / Tier-B benchmark** — separate sub-projects that consume the post-A2 sensor as their production capture chain.
