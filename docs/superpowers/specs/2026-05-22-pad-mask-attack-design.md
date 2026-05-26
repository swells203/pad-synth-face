% PAD Mask-Attack Module Design
% A third face-attack module (paper / silicone / resin masks) on the v2.1 + DigiFace baseline, with the v2/v2.1 artifact lesson designed in
% 2026-05-22

---

## 1. Purpose and audience

The real-bonafide (DigiFace-1M) iteration broke the synthetic-bonafide palette confound and produced the project's first plausible cross-domain PAD number (L1·D3 ≈ 0.178 with print physics). That result promoted the **mask-attack module** to the next sub-project on the combined v2.1-print + DigiFace-bonafide base.

This spec adds a third face-attack module — `mask` — alongside the existing `print` and `replay` modules. It models worn face masks across a material range (paper cutout → flexible silicone → rigid resin) using the same 2D image-space procedural approach as the existing modules. Crucially, it bakes in the hard-won v2/v2.1 lesson — **no deterministic generator fingerprint** — from the first line of code rather than discovering it through a failed sweep.

Audience: future maintainers and the Phase 2.5 author (real-attack capture). This module is the natural next attack surface once the bonafide-side confound was removed.

## 2. The question this answers

The deliverable is a clean **mask-only cross-domain EER** number, directly comparable to print's 0.178 headline, plus an integrated print+replay+mask number.

| Mask-only cross-domain EER | Diagnosis | Implication |
|---|---|---|
| **No cell ≤ 0.001**, lands in a plausible band (e.g. 0.05–0.35) | The mask physics is sound and artifact-free — the designed-in jitter/continuous-color discipline held. | Ship mask as a production attack class; proceed to the integrated sweep and Phase 2.5 real-attack capture. |
| **Some cells hit 0.000 / ≤ 0.001** | A mask-specific generator fingerprint slipped in despite the discipline (a fixed shading gradient, specular template, or seam geometry shared across Set A/B). | Isolate and jitter the offending stage; re-sweep. Same playbook as v2 → v2.1. |
| **All cells near 0.5 (chance)** | The mask cues are too subtle at 64×64 to separate from bonafide, or wash out under the sensor stage. | Revisit material-bundle strength / which cues dominate; the physics is too weak, not artifacted. |

Decision rule (mirrors the print sweeps): `no cross-domain cell mean ≤ 0.001` = artifact-free.

## 3. Architecture and files

The module follows the established `FaceAttackModule` pattern exactly. Net new surface is one source file, one ontology file, and a one-line registry edit.

| Change | File | Note |
|---|---|---|
| New module | `pad-synth-face/src/pad_synth_face/attacks/mask.py` | `MaskAttack` class |
| New ontology | `ontology/face/mask.yaml` | `attack_type: mask`, version `2026-05-22` |
| Registry edit | `pad-synth-face/src/pad_synth_face/pipeline.py` | add `"mask": MaskAttack` to `_ATTACK_REGISTRY` |
| Robustness fix | `pad-synth-face/src/pad_synth_face/pipeline.py` | §7 — canonical-version derivation |

**No eval changes.** `pad_synth_core.eval.baseline.TinyPADDataset` labels every `face/<subdir>/` except `bonafide` as attack (label 1). A mask attack writing to `face/mask/` is auto-discovered. `scripts/spark_sweep.py` is attack-agnostic (it trains on whatever lives in the Set-A dataset dir and evaluates the Set-B dir), so it needs no changes either.

`MaskAttack` implements the protocol from `attacks/base.py`:

```python
class MaskAttack:
    name = "mask"
    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "mask"
        self.ontology = ontology
    def sample_params(self, rng) -> dict[str, Any]:
        return self.ontology.sample_params(rng)
    def simulate(self, bonafide, params, rng) -> np.ndarray: ...
```

## 4. Ontology axes (`ontology/face/mask.yaml`)

One categorical material selector plus six continuous, jitter-friendly physics axes. Every axis carries literature provenance (enforced by `load_ontology`).

| Axis | Type | Range / values | Role |
|---|---|---|---|
| `mask_type` | categorical | `[paper, silicone, resin]`, weights `[0.30, 0.45, 0.25]` | selects the in-code material bundle (§5) |
| `light_azimuth_deg` | uniform | `[-180, 180]` | shading + specular highlight direction |
| `light_elevation_deg` | uniform | `[10, 80]` | shading + specular highlight direction |
| `specular_strength` | uniform | `[0.0, 1.0]` | gloss amount, scaled by material |
| `aperture_misalignment_px` | uniform | `[0.0, 4.0]` | eye/mouth hole misregistration vs. underlying face |
| `surface_warp` | uniform | `[0.0, 1.0]` | non-rigid drape of the mask over the face |
| `seam_visibility` | uniform | `[0.0, 1.0]` | mask-perimeter edge intensity |

Provenance anchors (assigned per-axis in the plan):

- Erdogmus & Marcel, "Spoofing Face Recognition with 3D Masks", IEEE TIFS 2014 (3DMAD database) — mask geometry, lighting, aperture cues. DOI `10.1109/TIFS.2014.2322255`.
- Manjani et al., "Detecting Silicone Mask-Based Presentation Attack via Deep Dictionary Learning", IEEE TIFS 2017 (SMAD) — silicone material appearance. DOI `10.1109/TIFS.2017.2676720`.
- Liu et al., "CASIA-SURF / HiFiMask: A Large-Scale High-Fidelity Mask Dataset", 2021 — material range (paper/resin/silicone) and weighting. DOI `10.1109/CVPR46437.2021.00616`.

## 5. Material bundle (in code, keyed by `mask_type`)

Approach A: a module-level dict mirrors `PrintAttack`'s `_PAPER_TINTS` / `_ICC_PARAMS`. Each entry bundles the material-dependent constants; the continuous axes (§4) modulate them per sample.

```
_MASK_MATERIALS: dict[str, MaskMaterial] = {
    # color_cast_rgb, specular_scale, texture_loss_sigma, subsurface_tint
    "paper":    flat / matte    / high texture-loss / no subsurface
    "silicone": waxy desaturated/ glossy            / mid texture-loss / warm translucent subsurface
    "resin":    cool, hard      / glossiest         / highest texture-loss (smoothest) / opaque
}
```

Exact numeric values are pinned in the implementation plan. The bundle holds constants only; all per-sample variation comes from the jittered axes, never from the bundle.

## 6. `simulate()` pipeline (2D image-space, continuous, fully jittered)

Operates on the `(64, 64, 3)` bonafide array in float `[0,1]`, returns `uint8`. The downstream sensor stage is applied by `pipeline.py` exactly as for print/replay.

1. **Texture-loss** — Gaussian low-pass (masks lack skin-pore detail); σ from the material bundle plus small per-sample jitter.
2. **Material color cast** — continuous multiplicative tint + subsurface additive tint from the bundle.
3. **Pseudo-3D shading** — an analytic elliptical-dome shading gradient (an image-space parametric field, *not* a per-face depth map), Lambertian term lit by `light_azimuth_deg` / `light_elevation_deg`. Image-space, consistent with replay's sheen overlay.
4. **Specular highlight** — a soft Blinn-Phong highlight whose position follows the per-sample light direction (`light_azimuth_deg`/`light_elevation_deg`); intensity = `specular_strength × material gloss` with per-sample rng jitter.
5. **Aperture mismatch** — soft darkened/offset eye and mouth regions (reuses print's region geometry, but soft-edged and offset by `aperture_misalignment_px`, not a hard cutout).
6. **Surface warp** — mild perspective/elastic warp for the drape, magnitude from `surface_warp` (reuses the `_perspective_warp` style).
7. **Seam** — a subtle darkened elliptical perimeter ring, intensity = `seam_visibility`.

## 7. Artifact discipline (the v2/v2.1 lesson, designed in)

This is a first-class requirement, not an afterthought:

- **No binary thresholding and no color quantization anywhere.** The v2 halftone's `(channel > screen)` binary step produced the ~16-color palette watermark that survived even v2.1's geometric jitter. The mask pipeline stays in continuous float until the final `uint8` cast.
- **Every spatial pattern is per-sample jittered.** Shading orientation, specular position/size, aperture offset, warp, and seam geometry all draw from `rng` each sample, so Set A and Set B never share a fixed geometry the detector can memorize.
- **Byte-level sanity check.** The sweep workflow asserts two same-`mask_type` samples produce byte-different outputs (mirrors the v2.1 T6 check) before trusting any EER.

## 8. Required robustness fix (surfaced by this work)

`pipeline.py:112` currently derives the canonical manifest version as:

```python
_ontology_version = attack_modules["print"].ontology.version
```

A **mask-only** generation config (§9, deliverable 1) has no `print` attack, so this raises `KeyError`. Fix: derive the canonical version deterministically from whatever attacks are present — a fixed priority order (`print` → `replay` → `mask`, falling back to the first attack name alphabetically) — and document why. This is a targeted in-flight improvement to code this work touches, not unrelated refactoring.

## 9. Deliverables

**Deliverable 1 — mask-only cross-domain sweep.**

New configs under `configs/runs/`: `mask_set{a,b}_d{1,2,3}.yaml`, each with the `mask` attack only, on the DigiFace 64×64 bonafide base, reusing the pinned Set A / Set B identity lists (`configs/digiface_identities_set{a,b}.txt`) and the v2.1 sensor/seed conventions. Run the existing 27-cell `scripts/spark_sweep.py` on the GB10 → a mask-only cross-domain EER table.

**Deliverable 2 — integrated multi-attack sweep.**

Configs adding `mask` alongside `print` + `replay` (weights pinned in the plan), Set A / Set B → a blended-detector cross-domain EER, to compare against the per-attack numbers.

**Report.** Both result tables are appended as new dated sections to `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (continuing the running log).

## 10. Testing

New `pad-synth-face/tests/test_mask_attack.py`:

- Output shape `(64,64,3)` and dtype `uint8` preserved.
- **Determinism:** same seed → byte-identical output.
- **Jitter:** different seeds → different output.
- **Anti-palette assertion:** output has far more than 16 distinct colors (guards the v2 mistake directly).
- **Material differentiation:** the three `mask_type` bundles produce measurably different image statistics.

Plus:

- `ontology/face/mask.yaml` is auto-covered by `tests/test_ontology_files.py` (lints all ontology YAMLs incl. provenance) — confirm it passes.
- Add `mask` to a pipeline end-to-end test (`pad-synth-face/tests/test_pipeline_e2e.py`).
- Add a mask determinism-golden entry to `tests/test_determinism_golden.py`.

## 11. Out of scope

- Real 3D geometry / morphable-model rendering (image-space approximation only — confirmed in brainstorming).
- Per-mask-type branched pipelines or a compositional layer framework (Approach A chosen).
- v2.2 print-physics changes (separate sub-project).
- Real mask capture (Phase 2.5).
