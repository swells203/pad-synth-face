# B1: synth-pretrain → real-finetune curve

Quantifies the hybrid hypothesis — pretrain on synthetic, finetune on N real
samples — by reporting **real-test EER as a function of N**. N=0 is the
synth-only baseline. Context: spec
`docs/superpowers/specs/2026-06-03-pad-b1-finetune-curve-design.md` and memory
`pad-next-sub-projects`.

**Real images are never committed** (`datasets/` is gitignored).

## What it does

`scripts/b1_finetune_curve.py` splits the real set once into a fixed
subject-disjoint `(finetune pool, real test)`, pretrains a model on the
synthetic root once, then for each N finetunes on the first N pool samples and
evaluates on the held-out real test. The test set is identical across all N, so
the curve is fair.

## Run

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d3 \
  --real-root  datasets/_real_attack/axondata \
  --n-list 0,50,200,1000 \
  --finetune-mode full \
  --model L4 \
  --test-fraction 0.3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_b1_curve \
  --device cuda
```

Writes `runs_b1_curve/runs/N<n>_seed<s>.json` (one per N) + `curve_summary.json`,
and prints an `N | real-test EER | ACER` table with a does-finetuning-help
readout. `--finetune-mode head` freezes the backbone and trains only the ResNet
`.fc` head (use with `--model L3`/`L4`). The pretrain step is the heavy one —
run on the Spark (`--device cuda`); the finetunes are cheap.

## Honesty notes (current data)

- The only real set staged is the **n=55 AxonData pilot**
  (`datasets/_real_attack/axondata`), licensed **CC-BY-NC-4.0 (NonCommercial)** —
  research-only, like DigiFace/DFDC (see memory `pad-commercial-licensing`). A
  model finetuned on it is **not commercially shippable**.
- n=55 only reaches **N=0 and ~the pool size** (after holding out the test
  split). Requested N larger than the pool are **skipped and logged** (never
  silently capped), so the table tells the truth about what ran.
- The real N=0/50/200/1000 curve needs purchased/larger real data — the same
  data step that unblocks the commercial-bonafide validation. Until then this is
  validated scaffolding: mechanically proven, awaiting data.
- **The pool/test split is only as subject-disjoint as the real manifest's
  `bonafide_source.id`.** The AxonData ingest sets that id to the source *file
  path* (unique per image), so the split is currently **sample-disjoint, not
  person-disjoint** — a person could appear as a bonafide selfie in the pool and
  a derived attack in the test set (leakage). Before any real curve is trusted as
  leakage-free, the real-attack ingest must emit a genuine per-person id. Inert
  for the n=55 plumbing run (EER meaningless at that scale).
- **ACER is not trustworthy below moderate N.** The ISO threshold is fixed on the
  finetune set; at tiny N (or a single-class N-slice) it can be a degenerate
  sentinel. Read the threshold-free `eer_cross_domain` as the curve's signal;
  treat `acer_cross_domain` as indicative only until N is reasonably large.

## Mechanical dry-run (no purchase needed)

```bash
.venv/bin/python scripts/b1_finetune_curve.py \
  --synth-root datasets/mix_seta_d1 --real-root datasets/_real_attack/axondata \
  --n-list 0,8 --model L1 --test-fraction 0.4 \
  --pretrain-epochs 1 --finetune-epochs 1 \
  --output-dir /tmp/b1_dryrun --device cpu
```

Proves the chain runs end-to-end on the pilot; the EER values are meaningless at
this scale.
