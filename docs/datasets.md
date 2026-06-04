# PAD datasets inventory

Snapshot of the data this project uses (as of 2026-06-04). Two fundamentally
different kinds: **source/input data** (externally obtained, fed *into* the
pipeline) and **generated synthetic PAD datasets** (*produced by* the pipeline).

**Real images are never committed** — `datasets/` is gitignored. This file is the
committed record of what lives there; the bytes live on the laptop (and a mirror
on the Spark, `~/ml/projects/pad-spark/datasets/`).

## 1. Source / input data (the only externally-sourced data)

| Path | Size | Count | What it is | Origin | Licence |
|---|---|---|---|---|---|
| `_real/digiface_118k_raw` | 3.5 GB | — | raw download | **DigiFace-1M** (Microsoft Research) — *synthetic computer-rendered* faces, NOT real people | non-commercial research (R-UDA; "no Data **or Results** in any commercial offering") |
| `_real/digiface_118k_64` | 1.3 GB | **33,333 identities / 166,665 imgs** @64px | bonafide source for the early 64×64 work | DigiFace-1M | research-only |
| `_real/digiface_224` | 6.1 MB | **24 identities / 120 imgs** @224px | bonafide source for **all 224 production work (B2/A2)**; split 8 (Set A) / 16 (Set B) | DigiFace-1M (small A1 subset) | research-only |
| `_real_attack/axondata` | 260 KB | **55 samples** = 24 bonafide + 12 print + 12 replay + 7 mask | the **only real faces + real attacks** in the project | HuggingFace `AxonData/face-anti-spoofing-dataset` (free sample) | **CC-BY-NC-4.0** (non-commercial) |

**DigiFace-1M** is the bonafide source — but note it is *itself synthetic*
(rendered CGI faces), so even "bonafide" here is not a real photograph.
**AxonData** is the only real data. Both are non-commercial.

## 2. Generated synthetic PAD datasets (~5.6 GB, 42 dirs under `datasets/`)

Produced by the pipeline = **DigiFace bonafide + the project's own physics-based
attack generators** (`pad_synth_face/attacks/{print,replay,mask}.py` + `base.py`).
These are derived/synthetic, not external data.

Naming: `<family>_set{a,b}_d{1,2,3,4}`. Set A = train, Set B = eval (subject-disjoint).
D-level = `samples_per_bonafide` (D1≈6, D2≈32, D3=256). Families (each ×set×d):

| Family | Dirs | What it is |
|---|---|---|
| `v2_` / `v21_` | 6 / 6 | early v2 / v2.1 print-attack generations |
| `real_` | 6 | DigiFace-bonafide + print+replay (the "real-bonafide" baseline) |
| `mask_` | 6 | mask-attack only |
| `mix_` | 6 | print + replay + mask (the production mix; `mix_seta_d3` = 4,096 samples) |
| `spark_` | 8 | Spark-sweep input sets (includes a d4) |

Plus a few older/POC dirs: `defid_poc_set{a,b}`, `phase1_smoke`, `phase15_setb`,
and test fixtures under `_fixtures/`.

Example sizes: `mix_seta_d3` 4,096 samples / 57 MB (2,048 bonafide + 2,048
attacks); `mix_setb_d3` 8,192 / 128 MB.

## 3. The honest headlines

1. **The entire project contains exactly 55 real-attack samples** (the AxonData
   pilot). Everything else is synthetic — and even the bonafide (DigiFace) is
   *rendered*, not photographed. That two-layer synthetic-ness is the heart of
   the **0.055 synth→synth vs 0.40 synth→real** gap (see
   `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` §2026-06-03).
2. **All 224 production results rest on just 24 DigiFace identities** (8 train /
   16 eval). Large sample counts, tiny identity base — why the synthetic
   cross-domain numbers are optimistic.
3. **Both source datasets are non-commercial** (DigiFace R-UDA, AxonData
   CC-BY-NC) → nothing trained on current data can ship (see memory
   `pad-commercial-licensing`).
4. This is why **CelebA-Spoof** matters: it would be the first real faces at
   scale (10,177 subjects) the project has seen. Ingest is built and waiting
   (`docs/celeba-spoof-b1.md`); also non-commercial (R&D / buy-decision only).

## 4. Not yet obtained (gated on the user)

- **CelebA-Spoof** (625k imgs / 10,177 subjects, research-only) — ingest built,
  blocked on the researcher-agreement download.
- **DFDC Preview** (~5 GB, EULA) — sweep prep built (`docs/dfdc-bonafide.md`).
- **A commercially-licensed real set** (~$10k tier, AxonData/Nexdata) — the only
  route to a *shippable* model; validation harness built
  (`docs/commercial-bonafide.md`).
