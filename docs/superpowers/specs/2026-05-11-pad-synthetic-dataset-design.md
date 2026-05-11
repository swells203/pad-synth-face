# Synthetic Presentation Attack Dataset — Design Spec

**Date:** 2026-05-11
**Status:** Approved (pending user review of written spec)
**Purpose:** Generate a synthetic, license-clean dataset for training presentation attack detection (PAD) models across face and voice modalities.

---

## 1. Goals and Constraints

### 1.1 Goals

- Produce a dataset suitable for training PAD/anti-spoof detectors covering **face** and **voice** modalities.
- Cover the major presentation attack types per modality:
  - **Face:** print, replay (screen), mask (2D/3D), digital injection / deepfake
  - **Voice:** voice conversion (VC), replay, cloning (TTS-based)
- The resulting dataset is fully owned by the producer and free of license entanglements from research-only PAD corpora.
- Realistic enough that a detector trained on synthetic data has measurable transfer to real attacks (tracked as the headline quality metric).

### 1.2 Non-goals

- Real-people bonafide data (Phases 1–3); a future phase may add a consent layer.
- Extraction or analysis of license-restricted PAD datasets (e.g., CASIA-FASD, Replay-Attack, OULU-NPU). Eval-only access for a small held-out set is the *only* permitted use, and that data never enters the training pipeline.
- Adversarial / digital-pixel attacks (FGSM, PGD).
- Building a production PAD detector (we ship a baseline for QC only).
- Real-time / streaming generation.
- GUI / dashboard.
- Cross-modal attacks (audio-driven face reenactment).
- Detector-aware adversarial optimization.

### 1.3 Hard constraints

- **License hygiene by construction.** Every input artifact (bonafide samples, generator weights, impulse responses, noise corpora) has its license recorded in the provenance ledger; downstream consumers can filter on it.
- **No automated feature extraction from license-restricted research datasets.** Attack parameter distributions are populated from *published literature only* (papers, surveys, vendor specs), each entry citing its source.
- **Reproducibility:** any sample is byte-exact regenerable from `(config, master_seed, sample_index)` under a determinism flag.

---

## 2. Architectural Approach

**Selected approach: two modality pipelines (`pad-synth-face`, `pad-synth-voice`) sharing a small core library (`pad-synth-core`).**

Rationale: face and voice attack research evolve independently; coupling them in a monolith creates merge friction. A generic plugin engine would over-abstract because the underlying data types (image tensor vs. waveform) and label granularities differ enough that a single plugin interface either leaks or becomes too thin to be useful.

```
              pad-synth-core
              ├── attack ontology (literature-cited YAML)
              ├── sensor/channel preset registry
              ├── manifest + provenance schema
              ├── reproducibility harness
              └── QC checks

              ▲                                    ▲
   ┌──────────┴───────────┐         ┌──────────────┴───────────┐
   │   pad-synth-face     │         │    pad-synth-voice       │
   │                      │         │                          │
   │ Bonafide:            │         │ Bonafide:                │
   │  DigiFace-1M (MIT)   │         │  LibriSpeech (CC BY 4.0) │
   │  SFHQ (secondary)    │         │                          │
   │                      │         │                          │
   │ Attacks: print,      │         │ Attacks: VC, replay,     │
   │ replay, mask,        │         │ cloning                  │
   │ digital injection    │         │                          │
   │                      │         │ Channel: mic, room, codec│
   │ Sensor: cam, optics, │         │                          │
   │ lighting             │         │                          │
   └──────────────────────┘         └──────────────────────────┘
```

### 2.1 Bonafide sources (least-risk configuration)

| Modality | Source | License | Notes |
|---|---|---|---|
| Face | DigiFace-1M (Microsoft) | MIT | 1.2M fully synthetic faces, commercial-use-clean |
| Face | SFHQ (secondary) | Permissive | Synthetic high-quality faces |
| Voice | LibriSpeech | CC BY 4.0 | Public-domain audiobook readings; real human speech, attribution-only |

**Critical asymmetry:** voice bonafide must be real human speech (LibriSpeech), not synthetic TTS, because synthetic TTS bonafide would contaminate the negative class with the same artifacts that mark the positive (cloning) class. Face bonafide can be synthetic (DigiFace) because rendering artifacts in synthetic bonafide do not overlap with the artifacts produced by physics-based attack simulation (print, replay, mask) or by post-capture processing of deepfake outputs.

### 2.2 Generation philosophy

**Physics-based simulation** is the primary approach: take a bonafide sample and simulate the physical attack chain that produces a presentation attack. Highly controllable, exact labels, no risk of detectors overfitting to a generator's fingerprint.

For attack types that are inherently generative (face deepfakes, voice cloning, VC), we use **third-party generative models as black-box data sources**. Physics is then applied *on top* of the generator's output (e.g., a deepfake video is post-processed through the replay/screen pipeline before being labelled as a presentation attack). To prevent detectors from overfitting to a single generator's fingerprint, we enforce **generator rotation**: each attack-type-with-generator must mix outputs from ≥2 different generators, and the manifest records which generator produced each sample.

### 2.3 Attack ontology — provenance rule

Every parameter range in the ontology has a `provenance` field citing the published source (paper DOI, vendor spec URL, equipment manual reference). Hand-written ranges without provenance are rejected by the ontology lint. This is the design's main legal defense: the ontology encodes *facts* about attacks (paper types, screen materials, mic frequency responses) sourced from public knowledge, never from license-restricted PAD images.

---

## 3. Face Attack Modules

All face attack modules implement:

```python
class FaceAttackModule(Protocol):
    name: str
    def sample_params(self, rng) -> dict
    def simulate(self, bonafide_img, params, rng) -> attack_img
```

The orchestrator picks the attack type, the module draws parameters from the ontology, then runs the simulation. Sensor-model effects (camera, optics, lighting) are applied *after* the attack module — shared across attack types.

### 3.1 Print attack

Simulated chain: digital image → print to paper → photograph the paper.

Stages:
1. Printer ICC profile + halftoning (color gamut compression, dot pattern artifacts)
2. Paper texture overlay (paper grain, fiber visibility)
3. Geometry: perspective warp simulating tilt
4. Specular reflection (anisotropic specular term, conditional on paper type)
5. Cutout artifacts (optional eye/mouth holes for wearable prints)

Ontology axes: `paper_type` (matte/glossy/photo), `print_dpi` (150/300/600/1200), `print_size` (4×6/A4/A3/face-actual), `tilt_degrees` ([-30,30]), `holder_present` (bool), `cutout` (none/eyes/eyes+mouth).

### 3.2 Replay (screen) attack

Simulated chain: digital image → display on screen → photograph the screen.

Stages:
1. Subpixel structure (RGB stripe / PenTile / OLED layouts)
2. Moiré pattern (2D sinusoidal interference with rotation/scale jitter)
3. Display gamma + color gamut roundtrip
4. Bezel masking (configurable insets, bezel materials)
5. Screen reflection (low-frequency luminance overlay)
6. Refresh-rate banding (for video bonafide; rolling-shutter × refresh-rate)

Ontology axes: `device_class` (phone-OLED/phone-LCD/tablet/laptop/desktop-monitor), `screen_size_in` (conditional on device_class), `bezel_pct` (0–20%), `viewing_angle` ([-45,45]), `ambient_reflection` (0–1), `refresh_hz` (60/90/120/144).

### 3.3 Mask attack (2D/3D)

Simulated chain: physical mask of victim's face → worn or held by attacker → photographed.

Stages:
1. 3D face reconstruction (FLAME or 3DMM — permissive licenses)
2. Mask material rendering (sub-surface scattering for silicone, matte BRDF for paper, glossy for resin)
3. Eye-hole cutouts (composite attacker's real eyes from a different bonafide identity)
4. Edge artifacts at neck/hairline (Gaussian feathering with discontinuity)
5. Static expression (frozen facial dynamics if video bonafide)

Ontology axes: `mask_type` (paper-2D/paper-3D/silicone/latex/resin-3D-printed), `attacker_eyes_visible` (bool), `mask_fit_quality` (tight/loose), `material_age` (fresh/worn).

Note: most compute-heavy face module due to the 3D reconstruction + rendering step. 3DMM fits are cached.

### 3.4 Digital injection / deepfake

Black-box-generator-as-data-source. No physics-from-scratch; third-party generators are wrapped and post-capture physics applied afterward.

Stages:
1. Generator zoo (pluggable): face-swap (InsightFace inswapper, SimSwap), face-reenactment (LivePortrait), full-face synthesis (StyleGAN3 inversion + edit). Each plugin has `license`, `commercial_ok`, `version`, `model_hash` metadata.
2. Generator rotation enforced via `generators_mix` proportion config.
3. Optional re-capture path (feed deepfake output through print or replay pipeline).
4. Compression artifacts (H.264/JPEG roundtrip at typical web/upload QFs).

Ontology axes: `generator_family` (faceswap/reenactment/synthesis), `generator_id` (versioned), `compression_qf` (50/70/85/95/lossless), `re_capture` (none/print/replay).

**License caveat:** several face-swap/reenactment models are research-only. The generator-zoo plugin metadata makes this filterable; orchestration modes include "commercial-clean only" and "research-permitted".

### 3.5 Shared sensor model (applied to all face attacks)

After attack-specific simulation, every sample passes through:

1. **Lens** — radial distortion, vignetting, chromatic aberration
2. **Optics** — depth of field, motion blur
3. **Sensor** — Bayer pattern → demosaic, ISO noise (shot + read), white balance
4. **ISP** — tone mapping, sharpening, JPEG/HEIC encode
5. **Lighting environment** — preset (indoor-tungsten / indoor-fluorescent / daylight / mixed / low-light)

Sensor presets: `mobile-front-2024`, `webcam-1080p`, `webcam-720p`, `kiosk-fixed-rgb`. Parameter ranges sourced from device teardown specs cited in the ontology.

---

## 4. Voice Attack Modules

All voice attack modules implement:

```python
class VoiceAttackModule(Protocol):
    name: str
    def sample_params(self, rng) -> dict
    def simulate(self, bonafide_audio, params, rng) -> attack_audio
```

### 4.1 Voice conversion (VC)

Simulated chain: attacker's source utterance → VC model converts timbre to target speaker → playback or injection.

Stages:
1. Source utterance picker (different LibriSpeech speaker as attacker source)
2. VC model zoo: FreeVC (MIT), kNN-VC (MIT-style), RVC (MIT), OpenVoice v2 (MIT)
3. Target speaker embedding (ECAPA-TDNN or x-vector, MIT weights)
4. Generator rotation enforced

Ontology axes: `vc_family` (any-to-any/one-to-one/few-shot), `vc_model_id`, `source_speaker_gender_match` (matched/mismatched), `reference_audio_seconds` (3/10/30).

### 4.2 Replay attack

Simulated chain: pre-recorded genuine audio → played through a speaker → captured by a microphone in a room. Most physics-heavy voice attack.

Stages:
1. Loudspeaker impulse response convolution (MIT Acoustical Reverberation Survey, OpenAIR — CC-licensed)
2. Loudspeaker nonlinearity (soft clipping, harmonic distortion; parametric `thd_pct`)
3. Room impulse response via pyroomacoustics (MIT) — small/medium/large rooms with adjustable RT60
4. Microphone impulse response convolution
5. Microphone self-noise at realistic SNR
6. Ambient noise from DEMAND/MUSAN (CC-licensed) at configurable SNR
7. Distance / level attenuation (1/r law + reverberant balance)

Ontology axes: `speaker_class` (phone-builtin/laptop-builtin/studio-monitor/cheap-bluetooth/smart-speaker), `mic_class` (phone-builtin/laptop-builtin/headset/lapel/array-far-field), `room_type` (small-quiet/small-noisy/medium-office/large-hall/outdoor), `rt60_seconds` (0.2–1.5), `distance_m` (0.1–3.0), `snr_db` (0–40), `noise_profile` (clean/babble/traffic/fan/music).

### 4.3 Cloning / TTS

Simulated chain: target speaker's reference audio + arbitrary text → neural TTS synthesizes target voice saying arbitrary content.

Stages:
1. Reference extractor (3–30s of bonafide audio)
2. TTS zoo (license-annotated):
   - XTTS v2 (Coqui Public License — non-commercial; flagged)
   - OpenVoice v2 (MIT — commercial-clean)
   - StyleTTS2 (MIT — commercial-clean)
   - MetaVoice-1B (Apache 2.0 — commercial-clean)
   - F5-TTS (CC BY-NC 4.0 — research only; flagged)
3. Text source: curated attack-content wordlist (login phrases, banking commands, voice-verification prompts) — no third-party text-corpus license issues
4. Generator rotation + license filtering enforced

Ontology axes: `tts_model_id`, `reference_length_s` (3/10/30), `text_category` (short-phrase/banking-prompt/sentence/passage), `prosody_match` (neutral/matched-to-reference).

### 4.4 Shared channel model (applied to all voice attacks)

After attack-specific simulation, every sample passes through:

1. AGC / level normalization
2. High-pass filter (telephony/VoIP DC blocker)
3. Bandwidth limiting (8/16/24/48 kHz)
4. Codec roundtrip (Opus / AMR-NB / AMR-WB / G.711 / MP3 at configurable bitrates via `ffmpeg-python`)
5. Packet loss simulation (for VoIP presets; random drops with PLC artifacts)
6. Final mic IR (replay path skips this since 4.2 already applied it)

Channel presets: `mobile-narrowband`, `mobile-wideband`, `voip-opus`, `landline-g711`, `webrtc`, `studio-clean`.

---

## 5. Manifests, Provenance, and Reproducibility

### 5.1 Sample manifest (JSONL, one row per sample)

```json
{
  "sample_id": "face-replay-000123abc",
  "modality": "face",
  "label": "attack",
  "attack_type": "replay",
  "bonafide_source": {
    "dataset": "DigiFace-1M",
    "id": "00342718",
    "license": "MIT",
    "url": null
  },
  "attack_params": {
    "device_class": "phone-OLED",
    "screen_size_in": 6.1,
    "bezel_pct": 4.2,
    "viewing_angle": -12.5,
    "refresh_hz": 120,
    "moire_seed": 8417263
  },
  "sensor_preset": "mobile-front-2024",
  "sensor_params": { "iso": 400, "wb_k": 5200, "jpeg_qf": 88 },
  "generators_used": [],
  "pipeline_version": "pad-synth-face@0.4.2",
  "core_version": "pad-synth-core@0.3.1",
  "ontology_version": "ontology@2026-05-08",
  "seed": 1739264817,
  "output_path": "face/replay/face-replay-000123abc.jpg",
  "output_sha256": "9f3a...c81",
  "created_at": "2026-05-11T14:33:12Z"
}
```

### 5.2 Provenance ledger (`provenance.jsonl`)

Dataset-level audit trail with one entry per ingested artifact:

```json
{ "type": "bonafide_dataset_ingested", "name": "DigiFace-1M",
  "license": "MIT", "source_url": "...", "sha256_of_index": "...",
  "ingested_at": "2026-05-11T09:00:00Z" }

{ "type": "generator_registered", "name": "OpenVoice-v2",
  "license": "MIT", "commercial_ok": true,
  "model_hash": "...", "registered_at": "..." }

{ "type": "ontology_citation", "attack_type": "replay",
  "axis": "moire_spatial_freq",
  "paper": "Galbally et al. 2014, IEEE TIP",
  "doi": "10.1109/TIP.2013.2292332" }
```

### 5.3 Reproducibility harness

**Seed plumbing:** master seed → per-sample derived seed `sha256(master_seed || modality || attack_type || sample_index)[:8]`. Sample seed is the only RNG source for that sample's attack module — no `time.time()`, no `os.urandom`, no unseeded library calls.

**Determinism contract:** nightly CI regenerates ~100 fixed samples from pinned seeds and compares SHA-256 to a checked-in golden manifest. Failure blocks the PR.

**Pinned nondeterminism sources:**
- PyTorch CUDA kernels (`torch.use_deterministic_algorithms(True)` where possible; CPU fallback where not)
- `ffmpeg` codec encoders (`-threads 1` for golden samples)
- JPEG encoder choice (standardize on one)
- `requirements.lock` pins all Python deps

**Trade-off:** strict determinism on GPU paths costs ~10–30% throughput. Determinism is a per-run flag (`deterministic: true|false` in config); manifest records which mode was used.

### 5.4 Output layout

```
dataset_root/
├── manifest.jsonl
├── provenance.jsonl
├── ontology.yaml
├── requirements.lock
├── face/
│   ├── bonafide/
│   ├── print/
│   ├── replay/
│   ├── mask/
│   └── deepfake/
└── voice/
    ├── bonafide/
    ├── vc/
    ├── replay/
    └── cloning/
```

Splits are emitted as separate manifest files (`splits/train.jsonl`, `splits/dev.jsonl`, `splits/test.jsonl`), not baked into directory layout — allows re-splitting without moving files. Default split strategy: **identity-disjoint** (no bonafide identity appears in both train and test).

---

## 6. Quality Control and Validation

Three QC layers run at different cadences.

### 6.1 Per-sample sanity (in-pipeline)

**Face:** one face detected in output; landmarks roughly match bonafide; pixel histogram non-degenerate; dimensions match preset; no NaN/Inf.

**Voice:** duration within ±5% of input; RMS in plausible range; spectrogram has energy across expected band; no NaN/Inf; VAD finds speech in non-silent segments.

Failed samples are logged with reason and regenerated up to 3× with perturbed seeds; persistent failures go to `failures.jsonl` and are skipped. Failure rate >1% triggers CI alert (likely upstream regression).

### 6.2 Distribution-level checks (post-batch)

- **Class balance** — attack-type counts within target ratios; sensor-preset coverage hits all configured presets; ontology axis coverage report.
- **Identity leakage** — no bonafide identity in both train and test splits (hard-fail).
- **Near-duplicate detection** — PHash (face) / fingerprint hash (voice) across whole dataset, flag pairs > similarity threshold. Distinguishes "same content, different attack" (OK) from "same params, near-identical output" (bug).
- **Generator-fingerprint risk score** — train a quick linear probe to classify *which generator made this sample* from simple features. If accuracy >70%, dataset has a strong generator signature; the detector trained on it will likely overfit. Mitigation: more generator rotation, heavier post-capture physics.
- **Bonafide-vs-attack triviality check** — train a tiny CNN for 1 epoch on a 5K subset. Target val accuracy 60–85%. >95% means dataset is degenerate (accidental shortcut). <55% means attacks aren't recognizable (upstream bug).

### 6.3 Cross-domain evaluation (headline metric)

Train a small PAD baseline (ResNet18 face / RawNet2 voice) on the synthetic dataset only; evaluate on a small held-out *real* PAD set used **for evaluation only, never training**.

**Reference eval sets:**
- Face: held-out slice of CelebA-Spoof (research license permits eval-only) and/or WMCA if accessible
- Voice: ASVspoof 2019 LA eval split (CC BY 4.0 — fully permissive but kept eval-only as honest yardstick)

Synthetic-trained → real-evaluated EER and HTER is the **only** number that tells us the dataset is getting better. Per-sample realism is irrelevant if cross-domain generalization is bad, and vice versa.

### 6.4 QC outputs

```
dataset_root/qc/
├── sample_failures.jsonl
├── coverage_report.json
├── duplicate_pairs.jsonl
├── generator_fingerprint.json
├── triviality_check.json
└── cross_domain_eval/
    ├── face_eer_history.jsonl
    └── voice_eer_history.jsonl
```

QC outputs ship with the dataset — consumers can read known limitations by construction.

---

## 7. Configuration, Orchestration, and CLI

### 7.1 Configuration

Single YAML per run. Hydra-style composition for command-line overrides.

```yaml
run:
  name: face_v1_2026_05_11
  output: ./datasets/face_v1
  seed: 1739264817
  deterministic: false
  workers: 8

modality: face

bonafide:
  source: digiface_1m
  count: 50000
  identity_disjoint_splits: true
  splits: { train: 0.7, dev: 0.15, test: 0.15 }

attacks:
  print:    { weight: 1.0, ontology_overrides: {} }
  replay:   { weight: 1.0, ontology_overrides: {} }
  mask:     { weight: 0.6, ontology_overrides: {} }
  deepfake:
    weight: 1.0
    commercial_only: true
    generators_mix:
      inswapper: 0.5
      liveportrait: 0.5

sensor:
  preset_mix:
    mobile-front-2024: 0.5
    webcam-1080p: 0.3
    kiosk-fixed-rgb: 0.2

qc:
  per_sample: true
  distribution_checks: true
  cross_domain_eval: { enabled: true, dataset: celeba_spoof_eval_slice }
```

Three config principles:
- **`weight`, not `count`** — specify proportions; orchestrator computes counts.
- **`ontology_overrides`** — empty default uses full literature distribution; per-axis overrides for ablation runs.
- **`generators_mix`** — explicit proportions, no hidden defaults; forces acknowledgment of fingerprint risk profile.

### 7.2 CLI surface

```
pad-synth face generate    --config configs/runs/face_v1.yaml
pad-synth voice generate   --config configs/runs/voice_v1.yaml
pad-synth face resume      ./datasets/face_v1
pad-synth voice resume     ./datasets/voice_v1
pad-synth qc run           ./datasets/face_v1
pad-synth qc report        ./datasets/face_v1 --format html
pad-synth manifest verify  ./datasets/face_v1
pad-synth manifest split   ./datasets/face_v1 --strategy identity-disjoint
pad-synth ontology show    face replay
pad-synth ontology lint
pad-synth face plan        --config configs/runs/face_v1.yaml
```

`plan` is a dry-run that emits the work-item enumeration and estimated cost/time/storage without generating samples.

### 7.3 Orchestration

Work items `(bonafide_id, attack_type, seed)` are pre-enumerated deterministically before any sample is generated. Workers consume from a disk-backed queue; the manifest writer is a single dedicated process (no contention; `fsync` every N samples).

**Resumability:** each sample writes a marker on completion. `resume` skips items with existing markers. Crash mid-sample re-queues that one item (worst case: 1 sample of duplicate compute).

Single-machine uses local multiprocessing; multi-machine uses a thin coordinator (Ray actor or filesystem lock on the work queue). Cloud-platform-agnostic by default.

### 7.4 Failure handling

| Failure | Behavior |
|---|---|
| Single sample generation crashes | Logged, regenerated up to 3× with perturbed seed |
| Worker process dies | Supervisor restarts; in-flight item re-queued |
| Whole run crashes | `resume` picks up exactly where left off |
| Bonafide dataset missing | Hard fail with pointer to provenance ledger |
| External generator unavailable | Sample marked `skipped:generator_unavailable`; run continues with remaining mix (manifest records rebalance) |
| Disk full | Writer pauses workers, exits cleanly |
| Determinism violation in golden CI | Hard fail with diff |

### 7.5 Packaging

**Monorepo with three Python packages** (`pad-synth-core`, `pad-synth-face`, `pad-synth-voice`). Simpler ops than three repos; cross-cutting changes (e.g., manifest schema bumps) ship atomically.

---

## 8. Phasing

### Phase 1 — Vertical slice (face + print + replay)

Goal: prove architecture end-to-end before scaling out attack types. Print and replay chosen because they're physics-pure (no generator zoo, no licensing complexity).

Deliverables:
- `pad-synth-core`: manifest schema, provenance ledger, seeded RNG, per-sample QC, CLI scaffold
- `pad-synth-face`: DigiFace-1M loader, print module, replay module, mobile-front sensor preset
- Ontology v0 (print + replay axes, fully cited)
- 5K-sample generation run
- CI: determinism golden test, triviality check, identity-disjoint split test
- Baseline ResNet18 PAD trained on 5K synthetic, evaluated on CelebA-Spoof eval slice

Success: end-to-end run completes; cross-domain eval number logged (whatever value); determinism CI green.

### Phase 2 — Face attack coverage

Mask module; deepfake module with generator zoo + license filtering; remaining sensor presets (webcam, kiosk); distribution-level QC; generator-fingerprint probe.

Success: all four face attack types working; ≥2 commercial-clean generators per attack-with-generator; coverage report shows all ontology axes exercised.

### Phase 3 — Voice modality

`pad-synth-voice` package; LibriSpeech loader; VC + replay + cloning modules; channel-model presets; ASVspoof eval integration.

Success: voice pipeline at parity with face Phase 2; cross-domain eval tracked for both modalities.

### Phase 4 — Scale, optional consent, optional formats

Resumable distributed generation; WebDataset/tar-shard output; Croissant metadata emission; optional consent.yaml infrastructure (for any future addition of real-people bonafide).

---

## 9. Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Cross-domain transfer is weak (synthetic-trained detector flunks real eval) | Medium-high | Track cross-domain EER as headline metric; iterate on physics not on more samples |
| Generator-fingerprint contamination in deepfake/cloning | High without active mitigation | Mandatory generator rotation; fingerprint probe in QC; post-capture physics on generator outputs |
| "MIT-licensed" generator turns out to have restrictive weights license | Medium | Per-generator license review in provenance ledger; `commercial_ok` flag enforced at runtime |
| Determinism breaks from library upgrade | Medium | CI golden test on every PR |
| Storage cost balloons at scale | Medium at Phase 4 | WebDataset tar-shards; lossless preprocessing audit |
| Ontology drifts from literature (numbers tweaked without citation) | Low-medium | Ontology lint requires `provenance` field on every range; CI check |
| 3D mask rendering produces obvious tells | Medium | QC triviality check per attack type will detect |

---

## 10. Limitations (shipped as `LIMITATIONS.md` with the dataset)

- Simulator-grounded, not real-world-validated beyond the small held-out eval set. Real-world deployment requires further validation against actual capture devices in the deployment environment.
- Reflects the literature's snapshot of attacks as of the ontology timestamp; new attack types require ontology updates.
- Detector quality is bounded by simulator fidelity — gains come from physics fidelity, not sample count.
