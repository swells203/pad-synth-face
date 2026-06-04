# Proposal — Mask-attack synthesis fidelity (3D/material rendering)

**Status:** PROPOSED future sub-project — not started. No spec/plan yet; this is
the pre-design capture. Promote to a spec via `superpowers:brainstorming` if/when
greenlit.
**Date:** 2026-06-04
**Origin:** diagnosed in the 2026-06-04 mask root-cause investigation (report
`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` §per-PAI; memory
`pad-next-sub-projects`).

## 1. The diagnosed problem

The single PAD detector transfers to real **print** attacks (per-PAI synth→real
EER 0.38) but **fails on real masks** (EER **0.67**, worse than chance). Systematic
debugging found the cause is **not** a wiring bug — every param in
`pad_synth_face/attacks/mask.py` renders measurably (toggling them moves pixels by
4.8–24.6 /255; the 8 seed-renders differ pairwise by mean 24/255). The cause is a
**model-fidelity limitation**: the mask synthesis is a 2D image-space
approximation whose own docstring states *"No real 3D geometry: the '3D-ness' is
faked with an analytic elliptical-dome shading field."* Every synthetic mask
therefore carries a shared synthetic **signature** — ~24% global darkening + a soft
texture-loss blur + the analytic dome "glow" — that real silicone/3D masks do not
have (rigid material edges, real specular, seams, true geometry). The detector
learns that synthetic tell, which is absent in real masks → no transfer.

**Net:** the synthetic mask attack does not *look like* a real mask. Closing the
mask half of the synth→real gap needs a higher-fidelity mask model (or real mask
data), not a parameter fix.

## 2. Goal & non-goals

**Goal:** make synthetic mask attacks resemble real worn masks closely enough that
the detector's per-PAI synth→real **mask** EER moves materially toward the print
level (≈0.38) on a real mask benchmark — without re-introducing a new fixed
synthetic fingerprint (the v2-style artifact trap; see `spark_dgx_workflow`).

**Non-goals:** print/replay generators (print already transfers; replay is a
separate, smaller gap); the detection model; the eval harness.

## 3. Approaches (sketch — decide at brainstorm time)

1. **Real 3D mask rendering (highest fidelity, biggest lift).** Fit a 3D morphable
   face mesh, drape a mask geometry over it, and render with a real material/BRDF
   shader (silicone gloss/translucency, resin, paper) under sampled lighting, then
   composite onto the bonafide. Captures true geometry, specular, edges, seams.
   Cost: a rendering dependency (e.g. a differentiable/mesh renderer) + 3DMM
   fitting; the largest engineering lift.
2. **2D-plus, texture-driven (moderate, incremental on current code).** Keep the
   image-space pipeline but replace the *analytic* dome/specular with
   **measured material textures** (real silicone/resin albedo + specular maps,
   real seam/edge crops), sharper rigid edges, and displacement, so the output
   stops reading as a smooth analytic glow. Cheaper; risk: still an approximation,
   may not fully close the gap.
3. **Real mask data (pragmatic, sidesteps synthesis).** Acquire real mask-attack
   samples (CelebA-Spoof has Face/3D mask types; commercial vendors sell silicone/
   3D-mask sets) and train/finetune on them directly. Not a *synthesis* fix, but
   likely the fastest route to a working real-mask detector — and consistent with
   the project's broader "real data is the higher-leverage path" finding.

## 4. Acceptance criteria

- On a **real mask benchmark** (CelebA-Spoof mask types, or a commercial 3D/
  silicone-mask set), per-PAI synth→real mask EER improves materially from the
  current ≈0.67 (target: approach the print transfer level, ≈0.4 or better).
- **No new artifact:** in-synth cross-domain EER does not collapse to ≈0.000 on
  mask cells (the fixed-fingerprint trap); mask renders remain per-sample jittered.
- Visual check: rendered masks show distinguishable material/specular/edge cues
  (the inverse of the 2026-06-04 `docs/figures/mask-samples.png` shared-glow look).

## 5. Sequencing — do NOT start yet

This sub-project is **gated on a real mask test set** — without one you cannot
measure whether a fidelity improvement actually closes the real gap (you'd be
optimising against a synthetic mirror, exactly the trap that produced the 0.67).
So the prerequisite is the **CelebA-Spoof run** (`docs/celeba-spoof-b1.md`), which
provides real mask samples and confirms/kills the per-PAI hypothesis at scale.

Priority is **below** the current data-gated queue (CelebA-Spoof / commercial
sourcing): given synthetic-only already fails to transfer (synth→real EER 0.40)
and real data is the higher-leverage path, invest in mask synthesis fidelity only
if (a) a real mask benchmark confirms the mask gap matters for the target use case
**and** (b) the decision is to keep pushing the synthetic route rather than
train/finetune on real masks (Approach 3).
