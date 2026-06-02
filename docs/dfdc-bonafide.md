# DFDC-grounded bonafide: ingesting Meta's DFDC for the existing pipeline

Replaces (or augments) the DigiFace bonafide source with real face frames
extracted from Meta's Deepfake Detection Challenge dataset. The existing
synthetic attack physics (print/replay/mask) rides on the new bonafide
distribution unchanged.

## 1. Obtain DFDC

DFDC is licence-gated (research EULA). Download from Meta / the original
Kaggle competition page; Preview (~5 GB) is the easiest first step, full
release is ~470 GB. Accept the licence terms before running anything below.

Unzip chunks anywhere on the laptop — each chunk extracts to a directory
containing videos plus a `metadata.json` mapping `filename -> {label:
REAL|FAKE, original: <REAL filename if FAKE>}`.

## 2. Install the optional `dfdc` extra (MediaPipe)

```bash
pip install -e 'pad-synth-face/[dfdc]'
```

Tests use a stub detector and don't require this; production ingest does.

## 3. Ingest → DigiFace-shaped bonafide root

```bash
.venv/bin/python scripts/prepare_dfdc.py \
  --src /path/to/dfdc/chunks \
  --out datasets/_real/dfdc_224 \
  --license "DFDC research licence (Meta AI)" \
  --source-url "<the URL you downloaded from>" \
  --res 224 \
  --frames-per-video 6
```

Writes `datasets/_real/dfdc_224/<video_stem>/NNN.png` (one directory per
REAL video, frames inside), plus `manifest.jsonl` and `provenance.jsonl`
recording the licence. Optional flags: `--max-videos N` for a quick smoke,
`--crop-margin 1.3` to widen the face crop.

**Resolution note (2026-06-01):** A1 (224 resolution) and A2 (capture-realism
sensor) have both shipped, and the production PAD config (B2 pretrained
ResNet18) runs at 224. The committed `dfdc_set*_d*` sweep configs (§5) point
at `datasets/_real/dfdc_224`, so ingest at `--res 224` — this gives an
apples-to-apples comparison against the DigiFace `real_set*` baselines, which
also use `datasets/_real/digiface_224`. (The earlier `--res 64` drop-in is now
obsolete; ingest at 64 only if you specifically want to reproduce the pre-A1
baseline.)

**Real frames are never committed.** `datasets/` is gitignored; keep
ingested DFDC roots under `datasets/_real/dfdc_<res>/`. Only the script,
fixture, tests, doc, and provenance/manifest schemas are committed.

## 4. Pin Set A / Set B identities

After the first ingest, pin disjoint identity lists from the ingested
videos with the committed helper, then commit the two lists:

```bash
.venv/bin/python scripts/pin_dfdc_identities.py   # reads datasets/_real/dfdc_224
git add configs/dfdc_identities_set*.txt
git commit -m "feat(pad-dfdc): pin DFDC Set A/B identities"
```

The helper writes `configs/dfdc_identities_seta.txt` (8 identities) and
`configs/dfdc_identities_setb.txt` (16 identities) via a deterministic seeded
shuffle (idempotent for a given ingested set; Set A and Set B are
identity-disjoint). Override `--root`, `--seta-count`, `--setb-count`, or
`--seed` if needed. The script errors out cleanly if fewer than 24 identities
were ingested.

## 5. `dfdc_set*_d*` sweep configs (already committed)

The six sweep configs already exist in `configs/runs/`:

```
dfdc_seta_d1.yaml  dfdc_seta_d2.yaml  dfdc_seta_d3.yaml
dfdc_setb_d1.yaml  dfdc_setb_d2.yaml  dfdc_setb_d3.yaml
```

Each is an exact mirror of its `real_<set>_d<n>.yaml` counterpart with only
the bonafide source swapped — `bonafide.root → ./datasets/_real/dfdc_224` and
`bonafide.identities_file → ./configs/dfdc_identities_set<a|b>.txt`. Everything
else is preserved: seeds (Set A 20260522, Set B 20260523), `samples_per_bonafide`
(Set A d1/d2/d3 = 6/32/256; Set B = 4/32/256), the print+replay attack mix, and
the per-set sensor preset (Set A mobile-front-2024, Set B webcam-1080p). This
makes the DFDC sweep an apples-to-apples swap of the DigiFace baseline — the
only variable that changes is the bonafide distribution.

No editing needed. They reference `configs/dfdc_identities_set*.txt`, which §4
produces; until those exist (i.e. until DFDC is ingested) the configs simply
can't run, which is the intended gate.

## 6. Run the sweep

Once §3–§4 are done (data ingested at 224, identities pinned), the sweep is a
one-command-per-step flow — identical to the 2026-05-31 A2 L4 sweep, just
swapping the dataset prefix to `dfdc_`:

```bash
# 1. Generate the 6 synthetic datasets locally (uses the A2 sensor pipeline)
for cfg in configs/runs/dfdc_set{a,b}_d{1,2,3}.yaml; do
  ds=$(basename "$cfg" .yaml); rm -rf "datasets/$ds"
  .venv/bin/python -m pad_synth_face.cli generate --config "$cfg"
done

# 2. rsync code + the 6 dfdc_* datasets to the Spark, then run spark_sweep.py
#    over the L4 cells (mirror Task 9 of the A2 plan, dfdc_ prefixes).
#    Output -> docs/.../runs_dfdc_224_L4/ ; pull back and aggregate.
```

The headline question: does **real DFDC bonafide + A2 synthetic attacks** at
224 beat the **DigiFace-bonafide L4 production baseline** (the `real_set*` /
`mix_set*` numbers — mix·D3 ≈ 0.055–0.059 cross-domain EER on pretrained
ResNet18)? Because the configs differ from `mix_set*` only in the bonafide
source, any EER delta is attributable to the bonafide distribution, not the
attack physics or capture chain. Append the result table to
`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.

This is the first sub-project that swaps in **real** bonafide faces, so it is
also the first real test of whether the A2 capture chain narrows the synth→real
gap (A2's in-synth sweep was flat by design — its payoff is expected to show
here).
