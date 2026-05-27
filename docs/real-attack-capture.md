# Real-attack capture: ingesting a real-attack PAD dataset

Harness for the synth→real generalisation test. Train on the synthetic
production base (v2.1 print + DigiFace bonafide + replay + mask), evaluate on
a real-attack dataset.

## 1. Arrange the source (folder convention)

Extract frames from the dataset (video decoding is your pre-step) and lay them
out as:

```
<src>/
  bonafide/**/*.{jpg,jpeg,png}
  attack/<attack_type>/**/*.{jpg,jpeg,png}   # e.g. attack/print, attack/replay
```

`<attack_type>` is any string; it becomes the attack-class subdir in the output.

## 2. Ingest → canonical 64×64 eval dataset

```bash
.venv/bin/python scripts/prepare_real_attack.py \
  --src /path/to/<src> \
  --out datasets/_real_attack/<dataset> \
  --dataset-name "MSU-MFSD" \
  --license "MSU research EULA" \
  --source-url "https://.../msu-mfsd"
```

Writes `datasets/_real_attack/<dataset>/face/{bonafide,<type>}/*.jpg`, a
`manifest.jsonl`, and a `provenance.jsonl` recording the dataset name + licence.
Optional `--max-per-class N` caps each class for a quick smoke ingest. Images
that fail the basic QC check (wrong shape, degenerate histogram) are skipped and
counted in the summary's `qc_skipped`.

**Real data is never committed.** `datasets/` is gitignored; keep ingested real
datasets under `datasets/_real_attack/`. Only the script, fixture, tests, and
this doc are committed.

## 3. Run the synth→real sweep

Generate the synthetic production base first (Set A), then point the sweep's
eval side at the ingested real dir for every data level:

```bash
.venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mix_seta_d1 --set-b-d1 datasets/_real_attack/<dataset> \
  --set-a-d2 datasets/mix_seta_d2 --set-b-d2 datasets/_real_attack/<dataset> \
  --set-a-d3 datasets/mix_seta_d3 --set-b-d3 datasets/_real_attack/<dataset> \
  --set-a-d4 datasets/mix_seta_d3 --set-b-d4 datasets/_real_attack/<dataset> \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_synth2real \
  --cells "$(python3 -c "print(','.join(f'{L}:{D}:{s}' for L in ('L1','L2','L3') for D in ('D1','D2','D3') for s in (0,1,2)))")" \
  --device cuda
```

This trains on synthetic at increasing data scale and evaluates on the fixed
real set — the headline synth→real EER curve. Append the result table to the
sweep-results report.
