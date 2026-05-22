# PAD Synthetic Dataset — Decisions, Trade-offs, and Roadmap

**Date:** 2026-05-11
**Status:** Living document
**Audience:** Future maintainers, future Claude sessions picking up this work, anyone reviewing the project's strategic posture
**Scope:** Synthesizes the decisions made across Phase 1, Phase 1 closure (eval loop), and Phase 1.5 design. Forward-looks to Phase 2 and beyond.

---

## Executive Summary

This project builds a synthetic, license-clean dataset pipeline for training Presentation Attack Detection (PAD) models. As of 2026-05-11:

- **Phase 1 is shipped and merged** to `main` at commit `8a857c5`. End-to-end pipeline produces labelled face data (bonafide + print + replay attacks) with full manifest, provenance ledger, determinism CI golden, and 55 passing tests.
- **The eval loop is closed.** Training a tiny CNN on a 96-sample balanced synthetic dataset achieves EER 0.37 — the detector is genuinely learning, but weakly.
- **Phase 1.5 is designed** (spec at `docs/superpowers/specs/2026-05-11-pad-synth-phase1_5-design.md`). It wires up a synthetic cross-domain proxy eval to produce a generalization signal that drives Phase 2 prioritization.
- **Phase 2's "go deep on physics" vs "go wide on attack types" is deferred** to the Phase 1.5 outcome. A decision matrix is committed.

The single biggest unresolved unknown is whether the detector's weakness is physics-limited, capacity-limited, or data-limited. Phase 1.5 is designed to disambiguate.

---

## 1. Project Context

**Goal**: Generate license-clean synthetic data for training PAD detectors across face and voice modalities, with the dataset fully owned by the producer and free of research-license entanglements.

**Primary purpose**: Train PAD/anti-spoof detectors. Not benchmarking, not privacy substitutes, not academic publication of generation methods (although the artifacts could serve all three secondarily).

**Modalities in scope**:
- **Face** (Phase 1 + 1.5 + 2): print, replay (screen), mask, digital injection / deepfake
- **Voice** (Phase 3): voice conversion, replay, cloning/TTS

**Architectural philosophy**: Physics-based simulation primary; third-party generative models used as black-box data sources where physics-from-scratch isn't possible (deepfakes, voice cloning).

---

## 2. Major Architectural Decisions and Trade-offs

### 2.1 Two-package monorepo: `pad-synth-core` + `pad-synth-face`

**Decision**: One repo with two Python packages — a small shared core (manifest, RNG, ontology, QC, orchestrator) plus a modality-specific face package. Voice gets its own `pad-synth-voice` package in Phase 3.

**Why**: Face and voice attack research evolve independently (different literatures, different hardware, different release cadence). Coupling them in a monolith creates merge friction. A generic plugin engine would over-abstract because image tensors and audio waveforms differ enough that a single plugin interface either leaks or becomes too thin.

**Trade-off**: vs. monolith (simpler initial codebase) or generic plugin engine (more flexible long-term). Settled on the middle: clean package boundary, shared schema, independent evolution.

### 2.2 Physics-based simulation primary

**Decision**: For attacks that can be physically simulated (print, replay, mask), simulate the physical chain. For attacks that are inherently generative (deepfake, voice cloning), use third-party generators as black-box data sources and apply post-capture physics on top.

**Why**: Physics-based simulation gives exact labels, full parameter control, and zero risk of detectors learning a generator's fingerprint (the silent failure mode of pure-generative PAD training data).

**Trade-off**: Slower to author than running a generative model end-to-end. Limited realism — Phase 1's MVP physics (e.g., paper-texture multiply + perspective warp instead of true halftoning) are deliberately simplified.

### 2.3 Bonafide sources fully license-clean (Phase 1 and Phase 1.5)

**Decision**:
- **Face Phase 1**: Procedural fixture (no licensing surface). Was scoped to use DigiFace-1M (MIT) but the actual data was deferred.
- **Face Phase 1.5 Set B**: Extended procedural fixture (still no licensing surface).
- **Voice Phase 3**: LibriSpeech (CC BY 4.0) — real human audiobook readings, attribution-only.

**Why**: License hygiene by construction. No consent forms, no GDPR exposure, no derivative-dataset entanglements with research-only PAD corpora (CASIA-FASD, OULU-NPU, etc.).

**Trade-off**: Synthetic-only face bonafide is procedurally generated, not photo-realistic. Real distribution shift between training and deployment is not tested until real-data integration arrives.

**Asymmetry worth understanding**: voice bonafide MUST be real human speech, not synthetic TTS. Synthetic TTS bonafide would contaminate the negative class with the same artifacts that mark the positive (cloning) class. Face bonafide can be synthetic because rendering artifacts in synthetic bonafide don't overlap with the physics-based attack artifacts (print, replay, mask).

### 2.4 Ontology populated from literature only, never from license-restricted datasets

**Decision**: Every attack-parameter range in the ontology YAML has a mandatory `provenance` field citing a published paper, vendor spec, or equipment manual. The loader rejects any axis missing this field.

**Why**: Facts, taxonomies, and parameter distributions are not copyrightable. Reading published literature *about* a dataset is clean; running statistical extraction over the dataset's images is risky. This is the design's main legal defense.

**Trade-off**: Slower to author than scraping a real PAD dataset's parameter distributions automatically. Requires literature reading and citation tracking. The constraint forces honest provenance.

### 2.5 Determinism end-to-end via SHA-256 seed derivation

**Decision**: Master seed → per-sample seed via `sha256(master_seed || modality || attack_type || sample_index)[:4]` (length-prefix encoded to eliminate delimiter collisions). All RNG flows from this derivation. Determinism is enforced by a nightly CI golden test that regenerates 16 fixed samples and compares SHA-256 of outputs.

**Why**: Byte-exact reproducibility is the foundation of every downstream guarantee — being able to say "this exact dataset was used to train detector X" requires that the dataset can be regenerated bit-perfectly from a committed config and seed.

**Trade-off**: ~10–30% throughput penalty in strict deterministic mode (GPU kernels). The pipeline supports a `deterministic` flag in config; fast mode for bulk generation, strict for CI golden.

### 2.6 Generator-as-data-source for deepfakes (Phase 2)

**Decision (planned)**: Third-party deepfake/cloning models will be wrapped as plugins with `license`, `commercial_ok`, `version`, `model_hash` metadata. Mandatory mix of ≥2 generators per attack-with-generator, with rotation enforced and the choice recorded in the manifest.

**Why**: Single-generator pipelines have a fatal failure mode — the PAD detector learns the generator's fingerprint instead of meaningful spoof signatures. Rotation forces the detector to look beyond fingerprint statistics.

**Trade-off**: Maintenance burden of multiple generator integrations. Some generators (XTTS v2, F5-TTS, SimSwap) are research-only — the `commercial_ok` flag is what makes this filterable at runtime.

---

## 3. Phase 1 Empirical Findings

### 3.1 Closing the eval loop — the EER journey

| State | Bonafide / Attacks | EER | val_acc | Reading |
|---|---|---|---|---|
| Initial wiring | 8 / 48 (6:1) | 0.70 | 0.79 | Worse than chance; class prior dominates |
| Balanced (Fix applied) | 48 / 48 (1:1) | **0.37** | 0.625 | Detector genuinely learning |

The first run produced EER 0.70 — *worse than chance*. The root cause was class imbalance: with only 8 bonafide samples and 48 attack samples, the model defaulted to "always predict attack" and got 79% accuracy by class prior alone. The score distribution was so skewed toward "attack" that the optimal-threshold EER ended up backwards.

**The fix**: emit `samples_per_bonafide` bonafide samples per identity instead of one. The pipeline now produces a balanced 48/48 dataset, and the EER becomes meaningful: 0.37, with val_accuracy 0.625 (well above the 50% balanced baseline).

### 3.2 What EER 0.37 doesn't tell us

- **Could be physics-limited**: Phase 1's print/replay physics are deliberately simple. No halftoning, no ICC color management, fixed-position cutouts, single-stripe subpixel model. The detector may have nothing rich to learn.
- **Could be capacity-limited**: TinyCNN is 8→16 channels, AdaptiveAvgPool→Linear. Tiny. Could be underfitting.
- **Could be data-limited**: 96 total samples (72 train / 24 val) is microscopic.

These three hypotheses can't be disambiguated from a single in-domain EER number. Hence Phase 1.5.

### 3.3 Why class balance was the critical fix (lesson for future scaffold eval setup)

The original pipeline emitted *one* bonafide sample per identity, treating bonafide as a per-identity property rather than a per-sample event. That assumption broke the eval scaffold silently. The fix — emit `samples_per_bonafide` bonafide samples per identity, each with distinct RNG-driven sensor params — is the same volume as the attack pass and produces balanced classes.

**Generalizable lesson**: For any in-development PAD scaffold, balanced classes are non-negotiable. The first sanity check should always be `val_accuracy > class_prior`; if it's not, the detector hasn't learned anything yet and the EER is uninformative.

---

## 4. Phase 1.5 Design Trade-offs (Just-Approved Spec)

### 4.1 Synthetic proxy vs real PAD data

**Decision**: Use a synthetic cross-domain proxy (Set A → Set B with different bonafide source and sensor preset).

**Why**: Phase 1.5's purpose is fast feedback. Real-data integration (CelebA-Spoof, MSU-MFSD, Idiap Replay-Attack) requires academic license click-through, multi-GB downloads, and acceptance of license terms — fundamentally slow.

**Trade-off**: The number this produces isn't the spec's headline metric (synthetic-trained → real-evaluated EER). It's a *floor* indicator: if cross-domain synthetic generalization fails, real-world performance will also fail. If it succeeds, real-world is undetermined.

### 4.2 Extended procedural fixture vs real DigiFace-1M for Set B's bonafide

**Decision**: Extended procedural fixture (16 identities, skin-tone color statistics, oval face-shaped silhouette).

**Why**: Real DigiFace-1M (1.2M faces, MIT) gives stronger signal but requires ~6 GB download. Phase 1.5's whole purpose is fast feedback.

**Trade-off**: Weaker domain-shift signal than real face photos would provide. Documented in `LIMITATIONS.md` as a synthetic-to-synthetic generalization test, not synthetic-to-real.

### 4.3 Hold attack ontologies stable between Set A and Set B

**Decision**: Change only bonafide source and sensor preset; same `print.yaml` and `replay.yaml` ontologies for both sets.

**Why**: Controlled experiment. Changing three axes at once (bonafide + sensor + physics) confounds attribution. Holding physics stable isolates the question we need answered: does the detector generalize across bonafide-source and sensor distribution shift?

**Trade-off**: Smaller total distribution shift between sets than would be possible. The trade-off accepts that for cleaner attribution.

### 4.4 Phase 2 decision matrix (committed in spec)

| Cross-domain EER | Interpretation | Phase 2 priority |
|---|---|---|
| **< 0.30** | Detector generalized; physics signal is strong | **Go wide** — add mask + deepfake |
| **0.30–0.45** | Detector working but no headroom | **Hybrid** — improve one attack's physics + add one attack type |
| **> 0.45** | Detector overfit to Set A | **Go deep** — physics fidelity is the bottleneck |

The matrix is explicit in the Phase 1.5 spec §7 so future decision-making is traceable to evidence.

### 4.5 Phase 1.5 outcome (measured 2026-05-12)

| Metric | Value |
|---|---|
| `eer_in_domain` (seed 0, epochs 10) | **0.29** |
| `eer_cross_domain` (seed 0, epochs 10) | **0.36** |
| `eer_cross_domain` range across seeds 0–3 | [0.31, 0.45], mean ≈ 0.39 |
| `n_train` | 72 |
| `n_val_cross_domain` | 128 (64 bonafide + 64 attacks in Set B) |

**Decision-matrix outcome: Hybrid.** Cross-domain EER (0.36 seed 0; 0.39 mean) lands solidly in the 0.30–0.45 band. The detector is generalizing partially across distribution shift but degrading meaningfully (from 0.29 to 0.36 EER, a +0.07 increase).

**Phase 2 recommendation: implement print physics improvements (halftoning + ICC color management) *and* add the mask attack in parallel.** The two are independent enough to ship together; the physics work shows whether weak physics was the in-domain bottleneck, and the mask work expands attack coverage. Defer the deepfake module to Phase 2.5 once the hybrid result clarifies which lever moves the EER more.

**Uncertainty caveat:** with 128 cross-domain eval samples, the EER has roughly ±0.07 noise across seeds. The hybrid call is robust to this — every seed in [0, 3] landed in the hybrid band — but seed 1's 0.45 reading touched the "go deep" threshold. If a future re-run with larger eval sets pushes the mean above 0.45, revisit the recommendation.

---

## 5. Recommendations for Future Progress

### 5.1 Phase 1.5 status (complete as of 2026-05-12)

Phase 1.5 shipped on branch `feat/pad-synth-phase1_5`. Headline cross-domain EER 0.36 → hybrid Phase 2 direction. See §4.5 for full numbers.

### 5.2 Phase 2 candidate scopes (depending on Phase 1.5 outcome)

**If "go deep" wins** (cross-domain EER > 0.45):
- Real halftoning (per-DPI dot patterns), printer ICC profile simulation
- Per-device subpixel models (PenTile, OLED stripe variants)
- Anisotropic specular for glossy paper
- Refresh-rate banding for video bonafide
- Vary source image per attack sample (Phase 1 deferral #6)
- Resolve `_FIXED_IMAGE_SHAPE` parameterization

**If "go wide" wins** (cross-domain EER < 0.30):
- Mask attack: FLAME or 3DMM reconstruction + material rendering (paper/silicone/latex/resin)
- Deepfake attack: generator zoo (face-swap, reenactment, full synthesis) with mandatory ≥2 commercial-clean per attack-with-generator
- Generator-fingerprint probe in distribution QC (linear probe to detect "which generator made this sample")
- Post-capture physics on deepfake outputs

**If "hybrid" wins** (0.30–0.45):
- Improve print physics (halftoning + ICC) — cheaper of the two go-deep options
- Add mask attack only (not deepfake) — cheaper of the two go-wide options

**As of Phase 1.5 (2026-05-12):** the cross-domain EER measurement (0.36) selected this path. The actual Phase 2 plan should scope: (a) print physics improvements — halftoning + ICC profile simulation, and (b) mask attack — FLAME or 3DMM reconstruction + material rendering. Defer deepfake module to Phase 2.5.

### 5.3 Real-data integration (Phase 2 or 2.5 — independent of deep/wide choice)

Regardless of Phase 2's main direction, the project needs a real-data eval baseline before any production claim:

1. **Phase 2.5 (or parallel Phase 2 sub-task)**: integrate MSU-MFSD as a starter (smallest, lightest license click-through). Build the loader to handle the MSU directory layout. Run synthetic→real EER. Log it as the headline metric.
2. **Phase 3 (or later)**: add CelebA-Spoof eval slice for the "official" headline number. Requires more storage and license acceptance.

### 5.4 Phase 3 — Voice modality

Per the original spec, the voice pipeline mirrors face's architecture:

- `pad-synth-voice` package built on the same `pad-synth-core`
- LibriSpeech bonafide (CC BY 4.0, real human audiobook readings)
- Three attack modules: voice conversion (FreeVC/RVC/OpenVoice), replay (impulse response convolution with pyroomacoustics), cloning (XTTS/StyleTTS2 etc.)
- Channel model: mic IR, room IR, codec roundtrip, packet loss, AGC
- ASVspoof 2019 LA eval as evaluation reference (CC BY 4.0)

The architectural lift is moderate — most of `pad-synth-core` (manifest, ontology, RNG, orchestrator, QC) is modality-agnostic and reusable.

### 5.5 Phase 4 — Scale and infrastructure

Once Phase 2 and Phase 3 are stable and the headline EER is meaningful:

- **Multi-process / multi-machine orchestration**: current pipeline is single-process. Workers + work-queue infrastructure already designed (in the original spec §7.3).
- **WebDataset / tar-shard output** for fast distributed training loaders.
- **Croissant metadata emission** (MLCommons standard) for publishable datasets.
- **Optional consent infrastructure** if real-people bonafide is ever added later (e.g., family photos). Currently explicitly out of scope.

### 5.6 Long-term concerns (worth flagging but not scheduled)

- **Scaling to 1M+ samples per attack type**: storage, dedup, distributed generation, network architecture upgrades.
- **Real-time generation**: explicitly out of scope today; would require a different latency posture.
- **Cross-modal attacks**: audio-driven face reenactment producing synchronized fake video+audio. Phase 5+ if ever.
- **Detector-aware adversarial optimization**: generating attacks targeted at a specific detector. Different research problem.

---

## 6. Known Limitations and Stewardship Advice

### 6.1 Current state limitations

- **Synthetic-on-synthetic eval only**. EER 0.37 is *not* a real-world performance claim. Don't quote it as one.
- **96 samples in the smoke set**. Statistical confidence is low.
- **Only 2 of 4 face attack types** implemented (print, replay). Mask and deepfake are Phase 2.
- **No voice modality**.
- **Procedural bonafide is not photographically realistic**. Real DigiFace-1M ingestion or real-photo bonafide is downstream work.
- **Ruff is not pinned in deps**. CI lint could flake on a ruff release.
- **`_ATTACK_REGISTRY` is not type-annotated** against the `FaceAttackModule` protocol — no static safety enforcement.
- **`_FIXED_IMAGE_SHAPE = (64, 64, 3)`** is a magic constant; varying image sizes will produce silent QC failures until parameterized.

### 6.2 Stewardship advice for the next maintainer

1. **Never claim real-world EER without real eval data.** The `LIMITATIONS.md` artifact ships with the dataset for this reason.
2. **Never weaken the ontology citation rule.** Every parameter range needs `provenance.paper` (with optional `doi`/`url`). The lint enforces this — don't disable it. Adding "TODO: cite" entries is the start of the slope toward statistical extraction from license-restricted images.
3. **Every new generator gets a `license` and `commercial_ok` flag in the provenance ledger.** When `commercial_only: true` is set on a run, the registry must filter on this.
4. **Re-run the determinism golden** whenever any code in the simulator path changes. If it fails unexpectedly, investigate before regenerating — a regression in determinism is a real bug.
5. **Resist the temptation to mock the test for `verify_identity_disjoint`** without actually running it on a real pipeline manifest. This QC nearly shipped as a silent no-op in Phase 1; the integration test that exercises it on pipeline output is what caught the gap.

### 6.3 Five Phase 2 deferral items from Phase 1 final review (not blockers, but tracked)

1. Pin `ruff` in a dev-deps group; lock the version
2. Unify the duplicated `QCResult` dataclass between `qc/per_sample.py` and `qc/distribution.py`
3. Vary source image per attack sample (currently always uses `bonafide_samples[0]`)
4. Add a `pad-synth-face init-fixture` CLI subcommand so `phase1_smoke.yaml` runs on a fresh clone
5. Type-annotate `_ATTACK_REGISTRY: dict[str, type[FaceAttackModule]]` for static safety
6. **(New, Phase 1.5):** The CLI's `eval` subcommand writes to stdout only; spec §4.5 said it should also append to `qc/cross_domain_eval/history.jsonl` for trend tracking. Currently done via shell redirect in the final-verification step. Move into the CLI for a clean history audit trail.
7. **(New, Phase 1.5):** Multi-seed reporting for the `eval` subcommand. Cross-domain EER at this dataset size has ~±0.07 noise across seeds; a `--seeds 0,1,2` option that reports min/mean/max would prevent over-reading any single number.

---

## 7. References

| Artifact | Path | Commit |
|---|---|---|
| Original full design spec | `docs/superpowers/specs/2026-05-11-pad-synthetic-dataset-design.md` | `0585c01` |
| Phase 1 implementation plan | `docs/superpowers/plans/2026-05-11-pad-synth-phase1.md` | `b969613` |
| Phase 1.5 design spec | `docs/superpowers/specs/2026-05-11-pad-synth-phase1_5-design.md` | `9a6677a` |
| Phase 1.5 implementation plan | `docs/superpowers/plans/2026-05-11-pad-synth-phase1_5.md` | `e243675` |
| Phase 1 merge to main | — | `0a04114` |
| Eval-loop closure | — | `d382a6c` |
| Phase 1.5 branch HEAD | `feat/pad-synth-phase1_5` | `0644dee` |
| Test count after Phase 1.5 | 70 passing | — |
| Phase 1 headline EER (in-domain, balanced) | 0.37 | smoke run with seed `20260511` |
| Phase 1.5 cross-domain EER (seed 0, epochs 10) | **0.36** | hybrid band → Phase 2 = print physics + mask attack |

---

## 8. Decision Heuristics for Future Sessions

Tactical guidance for whoever picks this up:

- **"Should I add this attack type?"** — Only after the existing attack types produce a detector that generalizes across the cross-domain proxy. Adding breadth on top of weak physics has zero leverage.
- **"Should I add this generator?"** — Yes, if it adds a *different* fingerprint family (faceswap vs reenactment vs synthesis) AND has a permissive license. No, if it's a v2 of an existing family.
- **"Should I scrape this real PAD dataset for parameters?"** — No. Read papers about it.
- **"Should I disable a CI check?"** — Almost certainly no. The determinism golden and ontology lint are load-bearing.
- **"Should I add a new modality?"** — Not until both face and voice have a real-data EER baseline.
- **"Should I commit this to main?"** — Only if tests pass on the merged result. The Phase 1 closure used a feature branch + verified-merge + delete-branch flow; preserve that hygiene.

This is a living document. Update it when Phase 1.5 runs, Phase 2 ships, or any strategic decision changes.


---

## 2026-05-22 update — Spark scaling sweep

The Phase 1.5 open question (capacity- / data- / physics-limited) has been disambiguated by a 3×3×3 sweep on the DGX Spark. **Diagnosis: data-limited.** More data drops cross-domain EER by 0.12–0.17 at every capacity tier; bigger models help only at intermediate data scales (D2) and stop helping at D3. Phase 2 should promote generation-scale to a first-class deliverable while keeping the physics improvements; deprioritize model architecture upgrades. See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) for the heatmaps and the updated Phase 2 recommendation.


---

## 2026-05-22 update — D4 (16k+/32k+) sweep extension

The data-limited diagnosis was extended with a D4 tier (Set A = 16k, Set B = 32k samples). **D4 verdict: axis plateaus.** L1 and L2 cross-domain EER is statistically unchanged from D3; L3 actually rises slightly (overfitting signature: in-domain collapses to 0.04 while cross-domain stays at 0.26). The Phase 2 recommendation reweights: promote print-physics + mask-attack work; bound generation scale around D3; consider real-data integration. See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"D4 result" for the updated heatmap column.


---

## 2026-05-22 update — v2 print physics sweep (artifact found)

Print attack upgraded to v2 (halftoning + ICC). 27-cell v1-vs-v2 sweep at D1–D3 numerically fires across all cells (Δ +0.156 to +0.250 cross-domain), BUT 6/9 cells hit exactly 0.000 EER both in-domain and cross-domain — diagnostic signature of a **generator-fingerprint artifact**: the deterministic halftone screen creates an identical watermark in Set A and Set B that the detector trivially learns. v2-as-implemented should NOT ship as production training data. Next iteration v2.1 needs halftone jitter (random sub-pixel offsets, jittered angles, varied dot shapes). Mask attack module still planned independently. Real-data integration rises in priority. See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"v2 print physics result" for details.


---

## 2026-05-22 update — v2.1 jittered-halftone sweep

v2.1 added per-sample geometric jitter to halftoning to break the v2 deterministic-screen watermark. 27-cell D1–D3 sweep. **Watermark verdict: SURVIVED — 6/9 cells still hit 0.000 cross-domain EER.** The byte-level pattern was broken (confirmed via same-DPI sample sha256 comparison), but the detector latches onto the higher-level **binary-threshold color palette artifact** — each pixel takes one of ~16 quantized colors regardless of dot placement, and this signature is identical across Set A and Set B. v2.1 retains a real ~0.15–0.25 cross-domain improvement at small scales (D1, L1·D2) where the artifact doesn't dominate, confirming physics contributes when not overwhelmed. Phase 2 prioritization update: **real-data integration moves up significantly**; v2.2 (gray-level halftoning) is a possible synthetic-physics next iteration but pure-synthetic halftoning likely produces a learnable palette signature at production scales. See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"v2.1 result" for the three-way table and details.


---

## 2026-05-22 update — real-bonafide (DigiFace-1M) v2.1 sweep

The originally-deferred Phase-1 real-data lever finally landed. DigiFace-1M 118k aligned subset (P1 partition: 33,333 identities, MIT-licensed, no auth required) ingested via the existing DigiFaceLoader (extended with `restrict_to` for identity pinning), resized to 64×64 for apples-to-apples comparison, 8 Set A + 16 Set B identity-disjoint pinned lists committed. v2.1 print physics held constant, 27-cell sweep at D1–D3 on the GB10. **Artifact verdict: BROKEN.** All 9 cross-domain cell means now > 0.001 (range 0.003–0.365). Headline finding: **L1·D3 = 0.178 ± 0.050** — TinyCNN on 4k real-bonafide samples achieves the first plausible cross-domain PAD number this project has produced (v1 at the same cell was 0.228; v2/v2.1 were 0.000 artifact-contaminated). The v2.1 physics genuinely contributes signal once the synthetic-bonafide confound is removed. Phase 2 prioritization: ship v2.1 + DigiFace as the new production baseline; mask-attack sub-project proceeds on this combined base; real-attack capture promoted to top Phase 2.5 priority; L3 ResNet18 stays demoted (still over-memorizes at scale). Also: the pipeline.py `ontology_version` hardcode (deferred since v2) was fixed as a ride-along. See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"real-bonafide v2.1 result" for the full synth-vs-real table and details.
