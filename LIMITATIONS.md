# Limitations

## Phase 1.5 cross-domain eval is a synthetic-to-synthetic proxy

The cross-domain EER reported by `pad-synth-face eval --train-root A --eval-root B`
is a synthetic-to-synthetic generalization test, not a synthetic-to-real-world test.

### What it does measure

- Whether the PAD detector overfits to a specific synthetic distribution
- Robustness to bonafide-source and sensor-preset variation, holding attack
  ontologies stable
- A floor indicator: if cross-domain synthetic generalization fails, real-world
  performance will also fail. If it succeeds, real-world performance is undetermined.

### What it does NOT measure

- Real-world deployment performance against actual print/replay attacks
- Generalization to attack types not present in either set (mask, deepfake)
- Generalization to capture devices not modeled by `mobile-front-2024` or
  `webcam-1080p` sensor presets
- Any production claim about detector quality

### Phase 2 follow-on

Real PAD dataset integration (CelebA-Spoof, MSU-MFSD, Idiap Replay-Attack) is
tracked as a Phase 2 sub-task. Until that lands, treat the cross-domain EER as
a directional signal, not a benchmark number.

### Other current limitations

- Only 2 of 4 face attack types implemented (print, replay; mask and deepfake
  are Phase 2 work)
- No voice modality (Phase 3)
- Bonafide source is procedural; real face data integration is Phase 2
- Datasets are small (under 200 samples in any single config)
- Phase 1 simulator uses simplified physics — full halftoning, ICC profiling,
  per-device subpixel models, and refresh-rate banding are Phase 2 work
