# PAD Synthetic Dataset — Phase 1.5 Design Spec

**Date:** 2026-05-11
**Status:** Draft (pending user review)
**Purpose:** Establish a cross-domain eval baseline that produces a useful signal for choosing Phase 2's "deep vs wide" direction, without the dataset-acquisition friction of a full real-eval integration.

---

## 1. Why Phase 1.5

Phase 1 ended with synthetic-trained → synthetic-evaluated EER 0.37 on a balanced 96-sample smoke dataset. That number doesn't tell us *why* the detector is only weakly learning: physics too simple, network too small, dataset too small, or all three. Without disambiguation, Phase 2's "deep vs wide" choice is a coin flip.

The spec's headline metric is synthetic-trained → **real-evaluated** EER. Wiring up real PAD data (CelebA-Spoof, MSU-MFSD, etc.) requires academic-license registration and multi-GB downloads — fundamentally slow. Phase 1.5 substitutes a **synthetic cross-domain proxy**: train on Set A, evaluate on a *deliberately different* Set B. The number this produces isn't the real-world metric, but it does answer the proximate question — *does the detector generalize across distributions at all?* — which is enough to set Phase 2 priorities.

---

## 2. Goals and Non-Goals

### 2.1 Goals
- Generate Set B: a synthetic dataset that differs from Set A in bonafide source and sensor preset, sharing the same attack ontologies for a controlled experiment.
- Add a `webcam-1080p` sensor preset (was already a Phase 2 deliverable).
- Add a `train_and_cross_domain_eval` entry point that returns both in-domain and cross-domain EER.
- Log the cross-domain EER and write `LIMITATIONS.md` capturing what the number does and does not represent.
- Use that number to set Phase 2 direction with evidence.

### 2.2 Non-goals
- Real PAD dataset integration (CelebA-Spoof, MSU-MFSD, etc.). Real-data eval is a Phase 2 sub-task.
- New attack types (mask, deepfake) — Phase 2 main scope.
- Physics fidelity improvements (halftoning, ICC, subpixel models, refresh-rate banding) — Phase 2 (depends on this phase's outcome).
- Voice modality — Phase 3.
- Network architecture changes beyond what the cross-domain eval entry-point needs.

### 2.3 Hard constraints
- All 55 existing tests must continue to pass.
- License posture preserved (no new license-restricted data introduced).
- Determinism golden test still green at the end (regeneration only if pipeline outputs change for Set A — they should not).
- The cross-domain eval runs alongside the in-domain eval; both numbers logged.

---

## 3. Approach

Phase 1.5 keeps all existing modules unchanged and adds three small things, no new packages, no architectural changes:

```
              pad-synth-core
              │
              ├── eval/baseline.py
              │     + train_and_cross_domain_eval()   ← new wrapper
              │
              └── (everything else unchanged)

              pad-synth-face
              │
              ├── sensor.py
              │     + WEBCAM_1080P preset             ← new
              │
              ├── _fixtures.py
              │     + build_extended_fixture_bonafide()  ← new fixture variant
              │
              ├── pipeline.py
              │     register WEBCAM_1080P in _SENSOR_REGISTRY  ← 1-line change
              │
              └── (everything else unchanged)

              configs/runs/
              ├── phase1_smoke.yaml         (existing — Set A)
              └── phase15_setb.yaml         (new — Set B)
```

### 3.1 The two domains

| Aspect | Set A (existing) | Set B (new) |
|---|---|---|
| Bonafide source | Procedural fixture (8 identities, uniform-color blobs + small noise) | **Extended procedural fixture**: 16 identities, oval face-shaped silhouettes with skin-tone color stats, larger native resolution downsized to 64×64 |
| Sensor preset | `mobile-front-2024` | `webcam-1080p` |
| Attack ontologies | Print + replay (literature-cited) | Same ontologies (controlled experiment) |
| Resolution at output | 64×64×3 RGB | 64×64×3 RGB (resized) |
| Sample counts | 8 identities × `samples_per_bonafide` | 16 identities × `samples_per_bonafide` |
| Seed | 20260511 | 20260512 |

**Deliberate choice — change bonafide source and sensor; hold attack ontologies stable.** This isolates the question we need answered: *does the detector generalize across bonafide and sensor distribution shift?* If we also varied ontologies we'd be confounding three change axes at once.

### 3.2 Why "extended procedural fixture" instead of real DigiFace-1M

The original spec said Set B's bonafide should come from real DigiFace-1M (1.2M synthetic faces, MIT license). That's a stronger signal but requires the user to download ~6 GB and pin a SHA-256 manifest. **Phase 1.5's whole purpose is fast feedback — we are deliberately not the right phase to introduce that download.** Real DigiFace-1M ingestion is a Phase 2 sub-task; this phase uses a richer procedural fixture for zero acquisition friction.

The extended procedural fixture creates real distribution shift from Set A:
- 16 vs 8 identities (more variety in the negative class)
- Skin-tone color statistics instead of uniform random RGB (matches what a real face image's color distribution looks like, even though the geometry is procedural)
- Oval silhouette with a darker eye-region and lighter lower-face area (basic face-like structure rather than a uniform blob)
- Different seed so per-identity colors don't overlap with Set A

This is acknowledged as a synthetic-to-synthetic proxy, not a synthetic-to-real test. `LIMITATIONS.md` documents the gap explicitly.

---

## 4. Components

### 4.1 `WEBCAM_1080P` sensor preset

`pad-synth-face/src/pad_synth_face/sensor.py`:

```python
WEBCAM_1080P = SensorPreset(
    name="webcam-1080p",
    iso_range=(200, 1600),       # webcams typically work in lower light → higher ISO
    jpeg_qf_range=(70, 92),       # webcams typically lossier JPEG than phones
    wb_k_range=(3200, 6000),     # broader indoor lighting variation
    vignette_strength=0.20,       # less vignette than a tight phone front cam
)
```

Provenance for the spec's ontology citations: aggregated from Logitech and Razer webcam teardown specs (USB-class sensors, OmniVision OV2710 / Sony IMX179 ranges). No new ontology axes — these are sensor preset constants, same shape as `MOBILE_FRONT_2024`.

Pipeline registration: one-line add to `_SENSOR_REGISTRY` dict in `pipeline.py`.

### 4.2 `build_extended_fixture_bonafide(root)`

New function in `pad-synth-face/src/pad_synth_face/_fixtures.py` (alongside existing `build_fixture_bonafide`):

```python
def build_extended_fixture_bonafide(root: Path) -> Path:
    """Set B's bonafide fixture: 16 identities with skin-tone color stats
    and an oval face-shaped silhouette. Procedural, deterministic, no license."""
    # 16 identities, each emits 4 PNG samples (different per-sample noise).
    # Skin tones: per-identity base RGB drawn from a Fitzpatrick-1-to-6 palette
    # (literature-derived rough color centers, not personally identifying).
    # Silhouette: oval mask (Gaussian falloff from center) at darker base color,
    # with a 'forehead/cheek' lighter region and 'eye/mouth' darker regions.
```

The function is purely procedural (no downloads, no external data). Output layout matches the existing `DigiFaceLoader` shape, so no new loader needed.

### 4.3 `train_and_cross_domain_eval` in `eval/baseline.py`

```python
def train_and_cross_domain_eval(
    train_root: Path,
    eval_root: Path | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a TinyCNN on train_root. Evaluate twice:
    - in-domain: held-out split of train_root (existing behavior)
    - cross-domain (if eval_root is given): full eval_root as held-out set

    Returns:
        {
            "eer_in_domain": float,
            "val_accuracy_in_domain": float,
            "n_train": int, "n_val_in_domain": int,
            "eer_cross_domain": float | None,
            "val_accuracy_cross_domain": float | None,
            "n_val_cross_domain": int | None,
        }
    """
```

The existing `train_and_eval_tiny_cnn` becomes a thin wrapper that calls this with `eval_root=None` and returns the in-domain fields under the original names, preserving its public API.

### 4.4 Set-B smoke config

`configs/runs/phase15_setb.yaml`:

```yaml
run:
  name: phase15_setb
  output: ./datasets/phase15_setb
  seed: 20260512
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_fixtures/extended_fixture
  samples_per_bonafide: 4
  splits: {train: 0.0, dev: 0.0, test: 1.0}  # entirety of Set B is held-out eval

attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml

sensor_preset: webcam-1080p
```

Note `splits: {train: 0.0, dev: 0.0, test: 1.0}` — all of Set B is held out. The eval reads the whole thing.

### 4.5 Cross-domain eval CLI

Extend the existing CLI with a new `eval` subcommand:

```
pad-synth-face eval --train-root datasets/phase1_smoke --eval-root datasets/phase15_setb
```

Output is the dict from `train_and_cross_domain_eval` written to stdout as JSON, and appended to `datasets/phase1_smoke/qc/cross_domain_eval/history.jsonl`.

---

## 5. Data Flow

```
1. Build extended procedural fixture
   → datasets/_fixtures/extended_fixture/  (16 identity dirs)

2. Generate Set A (existing)
   pad-synth-face generate --config configs/runs/phase1_smoke.yaml
   → datasets/phase1_smoke/

3. Generate Set B (new)
   pad-synth-face generate --config configs/runs/phase15_setb.yaml
   → datasets/phase15_setb/

4. Run cross-domain eval
   pad-synth-face eval --train-root datasets/phase1_smoke \
                       --eval-root  datasets/phase15_setb
   → stdout JSON + appended to datasets/phase1_smoke/qc/cross_domain_eval/history.jsonl
```

---

## 6. Tests

- All 55 existing tests pass unchanged.
- New tests:
  - `test_webcam_1080p_preset_applies_sensor`
  - `test_extended_fixture_builds_16_identities`
  - `test_train_and_cross_domain_eval_in_domain_mode` (eval_root=None → matches `train_and_eval_tiny_cnn` output shape)
  - `test_train_and_cross_domain_eval_two_roots` (cross-domain runs end-to-end on two minimal fixtures, returns both EERs)
  - `test_pipeline_handles_all_test_split` (loader's identity-disjoint-split correctly assigns all 16 identities to test when ratios=(0,0,1))
- Determinism golden test: should still pass without regeneration since Set A's pipeline (Phase 1 smoke config, mobile-front sensor, existing procedural fixture) is unchanged.

---

## 7. Success Criteria and Phase 2 Decision Matrix

Phase 1.5 succeeds when the cross-domain EER is logged with a meaningful interpretation.

The number lands in one of three regimes, each suggesting a different Phase 2 priority:

| Cross-domain EER | What it means | Phase 2 recommendation |
|---|---|---|
| **< 0.30** (better than in-domain ~0.37) | Detector generalized; physics signal is strong and the additional Set B variety actually helped (more bonafide diversity). The architecture is working. | **Go wide**: add mask and deepfake. Surface coverage is the missing piece. |
| **0.30–0.45** (similar to in-domain) | Detector is doing its job at a consistent baseline; neither overfit nor especially robust. | **Hybrid**: pick one of {improve print/replay physics, add one new attack type} based on which is cheaper. |
| **> 0.45** (worse than in-domain) | Detector overfit to Set A specifics. Even mild distribution shift breaks it. | **Go deep**: physics fidelity is the binding constraint. No new attack types until print/replay produce a detector that generalizes. |

---

## 8. Risks

| Risk | Probability | Mitigation |
|---|---|---|
| `int(round(n * 0.0))` boundary case in `identity_disjoint_split` for (0, 0, 1) ratios | Low–Medium | Add explicit test; current impl uses `int(round(...))` which should yield 0 for both, putting everyone in test. Verify and document. |
| Cross-domain EER is itself noisy at this dataset size (32 eval samples) | High | Report seed; consider running across 3 seeds and reporting min/median/max. |
| User reads cross-domain EER as "real-world EER" | Medium | `LIMITATIONS.md` is explicit; the CLI output prefixes both numbers with `in_domain` / `cross_domain` keys. |
| Extended procedural fixture is *too* similar to Set A's fixture | Medium | Distribution shift comes from three axes (16 vs 8 IDs, skin-tone vs uniform-random colors, oval vs blob geometry, plus sensor swap). If cross-domain EER ≈ in-domain EER, the fixture differences weren't enough → run a quick post-hoc analysis comparing pixel-stat histograms. |
| Determinism golden test breaks unexpectedly | Low | Set A's pipeline didn't change; if golden fails, investigate before regenerating. |

---

## 9. Out-of-Scope Items Explicitly Tracked for Later

The following are noted *not* fixed in Phase 1.5:

- **Real DigiFace-1M ingestion** — Phase 2 sub-task. The `DigiFaceLoader` already works on the real layout; what's missing is the data and a fetch script.
- **CelebA-Spoof / MSU-MFSD / Idiap Replay-Attack integration** — Phase 2 sub-task. Required for the spec's true headline metric.
- **Multi-seed eval reporting** — small enhancement, can be added when the eval is actually informative enough to need it.
- **The 5 Phase-2 deferral items from the Phase 1 final review** (ruff pin, QCResult unification, source-image variety, fixture bootstrap script, `_ATTACK_REGISTRY` type annotation) — Phase 2 main scope cleanup. Phase 1.5 only touches files that are directly related to the cross-domain eval.

---

## 10. Limitations to Ship as `LIMITATIONS.md`

```markdown
# Cross-domain eval limitations (Phase 1.5)

The cross-domain EER reported by `pad-synth-face eval --train-root A --eval-root B`
is a synthetic-to-synthetic generalization test, not a synthetic-to-real-world test.

What it does measure:
- Whether the PAD detector overfits to a specific synthetic distribution
- Robustness to bonafide-source and sensor-preset variation, holding attack
  ontologies stable

What it does NOT measure:
- Real-world deployment performance
- Generalization to actual print attacks captured on actual cameras
- Generalization to attack types not present in either set

Real-data eval (CelebA-Spoof, MSU-MFSD, etc.) is tracked as a Phase 2 sub-task.
Until that lands, treat the cross-domain EER as a *floor* indicator: if it's
bad, real-world performance will also be bad. If it's good, real-world
performance is undetermined.
```
