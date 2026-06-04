# CelebA-Spoof → B1 finetune curve

Answers the pivotal question from the 2026-06-03 reality check (synth→real EER
≈ 0.40 on n=55): **does real finetune data rescue it?** CelebA-Spoof (625k
images, 10,177 subjects) is the free, image-based real set big enough to run B1
meaningfully. Spec: `docs/superpowers/specs/2026-06-04-pad-celeba-spoof-b1-design.md`.

**Licence:** CelebA-Spoof is **non-commercial research only** (with a
"derived data" clause). This answers the buy decision; a model trained on it
**cannot ship** (see memory `pad-commercial-licensing`). Real images are never
committed (`datasets/` is gitignored).

## 1. Obtain

Accept the researcher release agreement and download from
https://github.com/ZhangYuanhan-AI/CelebA-Spoof (Google Drive links). Unzip to a
local path, e.g. `~/data/CelebA_Spoof`.

## 2. Confirm the format (one-time)

The adapter assumes images under `Data/{train,test}/<subject>/{live,spoof}/…`
and a 43-int annotation in `metas/intra_test/{train,test}_label.json` with the
spoof-type code at index 40 (`SPOOF_TYPE_INDEX` in `pad_synth_face/celeba_spoof.py`).
Inspect one label entry and confirm; if the index or layout differs, update the
named constants (one line each). Quick check:

```bash
python3 -c "import json; d=json.load(open('$HOME/data/CelebA_Spoof/metas/intra_test/train_label.json')); k=next(iter(d)); print(k, len(d[k]), 'code@40=', d[k][40])"
```

## 3. Stage + ingest

```bash
.venv/bin/python scripts/prepare_celeba_spoof.py \
  --src ~/data/CelebA_Spoof \
  --out datasets/_real_attack/celeba_spoof \
  --max-subjects 1500          # plenty for N=0..1000 + a disjoint test
```

Writes `datasets/_real_attack/celeba_spoof/` (canonical 224, manifest with
person ids, provenance recording the licence). `--max-subjects` keeps the
symlink/ingest light; raise it for the full run.

## 4. B1 run (pinned to the 0.40 baseline)

Generate the synth pretrain set if needed, then run B1 on the Spark with the
**same synth pretrain + L4** as the reality check so N=0 reproduces ≈0.40:

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d3 \
  --real-root  datasets/_real_attack/celeba_spoof \
  --n-list 0,50,200,1000 --finetune-mode full --model L4 \
  --test-fraction 0.3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_b1_celeba \
  --device cuda
```

## 5. Read the result

- **N=0** should land ≈ 0.40 (matches the reality check; sanity check).
- **EER drops clearly as N grows** → real finetune data rescues the model →
  the commercial-data purchase is justified (buy a *commercially-licensed*
  equivalent to actually ship).
- **Flat near 0.40** → real data alone doesn't fix it; the gap is deeper
  (architecture / domain-generalisation) → don't buy yet.

Append the curve table to
`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.
