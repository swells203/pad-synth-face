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

## 4. Running on the DGX Spark

The sweep needs the GB10 GPU; generation and ingestion are CPU steps. Datasets
are gitignored, so they are built on a machine that has the source data and
then synced to the Spark — the Spark's repo copy is updated by rsync, not git
(its SSH key is a deploy key for a different repo and can't pull this one).

1. **On the machine with the data (the laptop):** generate the synthetic base
   and ingest the real dataset.
   ```bash
   for d in 1 2 3; do
     python -m pad_synth_face.cli generate --config configs/runs/mix_seta_d$d.yaml
   done
   python scripts/prepare_real_attack.py --src <frames> \
     --out datasets/_real_attack/<dataset> \
     --dataset-name "<NAME>" --license "<EULA>" --source-url "<url>"
   ```

2. **Sync code + datasets to the Spark** (`swells@spark-50d2.local`, project at
   `~/ml/projects/pad-spark`):
   ```bash
   rsync -az --exclude='.venv' --exclude='.git' --exclude='datasets' \
     ./ swells@spark-50d2.local:~/ml/projects/pad-spark/
   rsync -az datasets/mix_seta_d1 datasets/mix_seta_d2 datasets/mix_seta_d3 \
     datasets/_real_attack/<dataset> \
     swells@spark-50d2.local:~/ml/projects/pad-spark/datasets/
   ```

3. **Run the §3 sweep on the Spark** via its venv:
   `ssh swells@spark-50d2.local 'cd ~/ml/projects/pad-spark && .venv/bin/python scripts/spark_sweep.py ...'`.
   The `--set-*-d4` args are required by the parser even though no D4 cells are
   requested — point them at the D3 dirs. A 27-cell sweep is ~3–5 min on the GB10.

4. **Pull results back** (`rsync` the `runs_synth2real/` dir to the laptop) and
   append the cross-domain EER table to the sweep-results report. Real images
   themselves are never committed — only the per-cell JSON/CSV results are.
