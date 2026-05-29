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
  --out datasets/_real/dfdc_64 \
  --license "DFDC research licence (Meta AI)" \
  --source-url "<the URL you downloaded from>" \
  --res 64 \
  --frames-per-video 6
```

Writes `datasets/_real/dfdc_64/<video_stem>/NNN.png` (one directory per
REAL video, frames inside), plus `manifest.jsonl` and `provenance.jsonl`
recording the licence. Optional flags: `--max-videos N` for a quick smoke,
`--crop-margin 1.3` to widen the face crop. The default `--res 64` is a
drop-in replacement for DigiFace; bump it later when A1+A2 ships.

**Real frames are never committed.** `datasets/` is gitignored; keep
ingested DFDC roots under `datasets/_real/dfdc_<res>/`. Only the script,
fixture, tests, doc, and provenance/manifest schemas are committed.

## 4. Pin Set A / Set B identities

After the first ingest, pick disjoint identity lists from the ingested
videos and commit them:

```bash
.venv/bin/python - <<'PY'
import pathlib, random
ids = sorted(p.name for p in pathlib.Path("datasets/_real/dfdc_64").iterdir() if p.is_dir())
random.Random(20260528).shuffle(ids)
seta, setb = ids[:8], ids[8:24]
pathlib.Path("configs/dfdc_identities_seta.txt").write_text("\n".join(seta) + "\n")
pathlib.Path("configs/dfdc_identities_setb.txt").write_text("\n".join(setb) + "\n")
print("pinned:", len(seta), "Set A,", len(setb), "Set B")
PY
git add configs/dfdc_identities_set*.txt
git commit -m "feat(pad-dfdc): pin DFDC Set A/B identities"
```

## 5. Create `dfdc_set*_d*` sweep configs (paste once after ingest)

For each `(set, d)` ∈ {(a, 1), (a, 2), (a, 3), (b, 1), (b, 2), (b, 3)},
write `configs/runs/dfdc_<set>_d<n>.yaml` as a clone of
`real_<set>_d<n>.yaml` with two changes — point `bonafide.root` at the
DFDC dir and `bonafide.identities_file` at the DFDC list, e.g.
`configs/runs/dfdc_seta_d3.yaml`:

```yaml
run:
  name: dfdc_seta_d3
  output: ./datasets/dfdc_seta_d3
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/dfdc_64
  samples_per_bonafide: 256
  identities_file: ./configs/dfdc_identities_seta.txt
  splits: {train: 0.0, dev: 0.0, test: 1.0}

attacks:
  mask:
    weight: 1.0
    ontology: ./ontology/face/mask.yaml

sensor_preset: mobile-front-2024
```

(Mirror the corresponding `real_set*` files for non-D3 / Set B; preserve
the seeds and `samples_per_bonafide` numbers.)

## 6. Run the sweep

Generate the synthetic datasets locally, rsync to the Spark, sweep on the
GB10 — same procedure as `docs/real-attack-capture.md` §4, swapping
`mix_seta_d*` for `dfdc_seta_d*`. The headline question is whether real
DFDC bonafide + v2.1 synthetic attacks beats the DigiFace-bonafide baseline
(mask-only L3·D3 ≈ 0.089, integrated L2·D3 ≈ 0.094) at 64×64. Append the
result table to `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.

If yes, the next cycle is the A1+A2 resolution bump which compounds with
this.
