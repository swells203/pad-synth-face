# Commercial-bonafide retrain validation

Swaps the (non-commercial, research-only) DigiFace bonafide for a
**commercially-licensed** bonafide face set, runs the existing synthetic
attacks on top, and produces a PASS/FAIL verdict on whether cross-domain EER
holds. Validates a shippable model BEFORE buying data (use free vendor samples
first). Context + licensing rationale: memory `pad-commercial-licensing` and
spec `docs/superpowers/specs/2026-06-02-pad-commercial-bonafide-validation-design.md`.

**Real images are never committed** — `datasets/` is gitignored. Only the
licence string + source URL travel with the data via `provenance.jsonl`.

## 1. Obtain a commercially-licensed sample

Vendors with free, commercially-licensed samples: AxonData (axonlab.ai),
Nexdata, Unidata, Shaip. Request the free sample of a face anti-spoofing /
liveness set and note the exact licence string + the URL you downloaded from —
both are recorded into provenance at ingest.

## 2. Reshape into the canonical contract

The ingest expects `<src>/<identity>/<sample>.{png,jpg,jpeg}` — one directory
per subject. If the vendor ships a different layout (flat folder, parquet,
video), write a thin shim that reshapes it into that contract first. Example
for a flat folder where the filename prefix is the subject id
(`subjA_001.jpg`, `subjA_002.jpg`, `subjB_001.jpg`, …):

```bash
.venv/bin/python - <<'PY'
import pathlib, shutil
src = pathlib.Path("/path/to/vendor_flat")
dst = pathlib.Path("/path/to/vendor_canonical")
for img in src.glob("*.jpg"):
    subj = img.stem.split("_")[0]
    out = dst / subj
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out / img.name)
print("reshaped into", dst)
PY
```

## 3. Ingest → 224 bonafide root (records the licence)

```bash
.venv/bin/python scripts/prepare_commercial_bonafide.py \
  --src /path/to/vendor_canonical \
  --out datasets/_real/commercial_224 \
  --license "<exact vendor commercial licence string>" \
  --source-url "<the URL you downloaded from>" \
  --vendor axondata
```

Writes `datasets/_real/commercial_224/<identity>/NNN.png` + `provenance.jsonl`
+ `_meta.json`. Optional `--max-per-identity N`.

## 4. Pin Set A / Set B identities

```bash
.venv/bin/python scripts/pin_commercial_identities.py
git add configs/commercial_identities_set*.txt
git commit -m "feat(pad-commercial): pin commercial Set A/B identities"
```

Needs >= 24 ingested identities (8 Set A + 16 Set B). Override
`--seta-count`/`--setb-count`/`--seed` for a smaller sample.

## 5. Generate datasets + sweep on Spark

The six `configs/runs/commercial_set*_d*.yaml` are already committed. Generate
the synthetic datasets locally, rsync to the Spark, and run the L4 sweep —
identical to the DFDC/A2 procedure, `commercial_` prefixes:

```bash
for cfg in configs/runs/commercial_set{a,b}_d{1,2,3}.yaml; do
  ds=$(basename "$cfg" .yaml); rm -rf "datasets/$ds"
  .venv/bin/python -m pad_synth_face.cli run --config "$cfg"
done
# rsync code + the 6 commercial_* datasets to the Spark, then run
# spark_sweep.py over the L4 cells -> runs_commercial_224_L4/ ; pull back.
```

**Matched-scale caveat:** the verdict compares against the DigiFace
`runs_mix_224_L4_A2` baseline, which used `samples_per_bonafide` 6/32/256. If
your sample is too small for D3 (256/bonafide), reduce `samples_per_bonafide`
*symmetrically* in both the commercial configs and a re-run DigiFace sweep, and
point `--baseline-dir` at that matched DigiFace run. The verdict script prints a
scale-mismatch WARNING if the two sweeps don't actually match.

## 6. Verdict

```bash
.venv/bin/python scripts/compare_bonafide_eer.py \
  --commercial-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_commercial_224_L4 \
  --baseline-dir   docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2
```

Prints the per-cell delta table and `PASS`/`FAIL`. PASS (every cell |delta| <= 0.03, no
collapse) means the commercial bonafide preserves EER → a shippable model is
viable and the ~$10k purchase is de-risked. FAIL means investigate before
buying. Append the table to
`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.
