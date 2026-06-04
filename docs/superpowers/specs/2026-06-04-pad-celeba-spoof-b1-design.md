# PAD — CelebA-Spoof ingest + B1 run (design)

**Date:** 2026-06-04
**Status:** approved (brainstorm) → ready for implementation plan
**Branch:** `feat/pad-celeba-spoof-b1`

## 1. Problem & motivation

The 2026-06-03 reality check measured the production model's first real
synth→real number: **EER ≈ 0.40** on the n=55 AxonData pilot (vs 0.055
synth→synth). The pivotal open question is **does real finetune data rescue it**
— the B1 curve, with 0.40 as the N=0 point. The B1 harness is built but needs a
real set with far more than 24 identities to be meaningful.

**CelebA-Spoof** (625k images, **10,177 subjects**, 11 spoof types, free,
image-based, real per-subject folders) is the strongest free option for this.
It is **non-commercial research-only** — fine for the *decision* (internal R&D,
not shipping). This sub-project builds the ingest that turns CelebA-Spoof into
the canonical real-attack eval set and wires the B1 run against the 0.40
baseline. **No CelebA-Spoof data is staged yet** — this ships as ready
scaffolding (fixture-validated), inert until the user downloads it, exactly like
the DFDC / commercial-bonafide / B1 harnesses.

## 2. Goal & non-goals

**Goal:** Given a downloaded CelebA-Spoof tree, produce
`datasets/_real_attack/celeba_spoof` (canonical layout, **person-disjoint**
subject ids) with a documented one-command B1 run pinned to the 0.40 baseline's
setup, so the curve's N=0 ≈ 0.40 and N>0 answers the rescue question.

**Non-goals:**
- No download automation (researcher agreement + ~tens-of-GB Drive download is
  the user's step).
- No CelebA-Spoof protocol files — we derive our own person-disjoint split.
- No image copying — symlink staging only.
- No change to the B1 runner or `train_and_cross_domain_eval`; one small
  *additive* extension to `ingest_real_attack`.
- No commercial use — research-only data; a model trained on it can't ship (this
  answers the buy decision, it isn't the product).

## 3. Decisions (locked in brainstorm)

| Decision | Choice |
|---|---|
| Data status | **Not staged — build ready** (fixture-validated, inert until download) |
| Spoof mapping | **Match synth taxonomy, exclude partials:** 0→bonafide; 1,2,3→print; 7,8,9→replay; 4,10→mask; **5,6 skipped** |
| Architecture | **Approach 1** — staging adapter + tiny `ingest_real_attack` extension + reuse the B1 runner |
| Person ids | Preserve CelebA-Spoof subject → **person-disjoint** B1 split (fixes the AxonData leakage caveat) |

## 4. Architecture

Four units; maximal reuse of `ingest_real_attack` and the B1 runner.

```
CelebA-Spoof tree ──► stage_celeba_spoof() ──► <staging>/bonafide/<subj>/*  (symlinks)
   (label file: spoof code per image)          <staging>/attack/{print,replay,mask}/<subj>/*
                                                              │
                                       ingest_real_attack(subject_id_fn=...) ──► datasets/_real_attack/celeba_spoof
                                                              │   (canonical 224 + manifest w/ person ids + provenance)
                                                              ▼
                                       scripts/b1_finetune_curve.py  (EXISTING, unchanged)
                                          --synth-root mix_seta_d3  --real-root .../celeba_spoof  --model L4
                                                              ▼
                                       EER-vs-N curve (N=0 ≈ 0.40 baseline)
```

### 4.1 `pad_synth_face/celeba_spoof.py` — staging adapter

```
SPOOF_TYPE_TO_CLASS = {0:"bonafide", 1:"print",2:"print",3:"print",
                       7:"replay",8:"replay",9:"replay", 4:"mask",10:"mask"}
# 5 (Upper-Body Mask), 6 (Region Mask) intentionally absent -> skipped.

stage_celeba_spoof(src: Path, staging: Path, max_subjects: int | None = None,
                   splits=("train","test")) -> dict[str,int]
```
- Reads CelebA-Spoof per-image annotations to get each image's spoof-type code.
  The code's position in the 43-int label list is a **named constant**
  `SPOOF_TYPE_INDEX` (documented; confirm against the real label file on first
  download — see §5).
- For each image whose code is in `SPOOF_TYPE_TO_CLASS`, **symlink** it to
  `<staging>/bonafide/<subject>/<name>` (class bonafide) or
  `<staging>/attack/<class>/<subject>/<name>`. Subject = the CelebA-Spoof
  per-subject folder name from the image path. Codes 5/6 (and any unknown) are
  skipped and counted.
- `max_subjects` caps the number of distinct subjects staged (deterministic,
  sorted) — you don't need all 10k. Returns counts per class + n_subjects +
  n_skipped.
- Pure staging (no resize/manifest) — `ingest_real_attack` does that next.

### 4.2 `ingest_real_attack` extension (`pad_synth_face/real_attack.py`)

Add one optional param: `subject_id_fn: Callable[[Path], str] | None = None`.
At the single `BonafideSource(... id=...)` site (real_attack.py:101), change to:
```python
id=(subject_id_fn(fp) if subject_id_fn is not None else str(fp.relative_to(src)))
```
Default `None` preserves current behaviour for every existing caller (verified:
the only callers are `scripts/prepare_real_attack.py` and tests). The
CelebA-Spoof shim passes a closure over the staging root that returns the
subject path-segment (`bonafide/<subj>/…` → `<subj>`; `attack/<type>/<subj>/…`
→ `<subj>`), so the manifest's `bonafide_source.id` becomes the **person id** →
`subject_disjoint_split` in B1 is genuinely person-disjoint.

### 4.3 `scripts/prepare_celeba_spoof.py` — CLI shim

Args: `--src` (CelebA-Spoof root, required), `--out`
(default `datasets/_real_attack/celeba_spoof`), `--license` (default the
CelebA-Spoof non-commercial terms string), `--source-url`, `--max-subjects`,
`--staging` (default a temp dir under `datasets/_real_attack/_staging_celeba`),
`--max-per-class`. Calls `stage_celeba_spoof` then
`ingest_real_attack(..., subject_id_fn=<staging-aware fn>)`. Prints the summary.

### 4.4 B1 run wiring — `docs/celeba-spoof-b1.md`

Runbook: obtain (researcher agreement at the GitHub repo) → `prepare_celeba_spoof`
→ the **B1 run pinned to the 0.40 baseline**:
```
scripts/b1_finetune_curve.py --synth-root datasets/mix_seta_d3 \
  --real-root datasets/_real_attack/celeba_spoof --n-list 0,50,200,1000 \
  --finetune-mode full --model L4 --output-dir .../runs_b1_celeba --device cuda
```
Same synth pretrain (`mix_seta_d3`) + L4 as the reality check, so **N=0 should
reproduce ≈0.40** and the curve's slope is the answer. Interpretation note: a
clear EER drop as N grows = real finetune rescues it (commercial buy justified);
flat near 0.40 = deeper problem (don't buy yet).

## 5. CelebA-Spoof format assumptions + risk

The exact on-disk label-file layout could not be fully verified without the data
(the GitHub README is only partial). Documented assumptions, all isolated behind
named constants for a one-line fix on first real download:
- Images under `Data/{train,test}/<subject>/...` with live/spoof images; subject
  = the `<subject>` path segment.
- Per-image 43-int annotation; spoof-type code at `SPOOF_TYPE_INDEX` (default per
  the paper's documented ordering). Live = code 0.
- The parser reads the label file(s) (JSON or whitespace txt — the adapter
  handles both) mapping image path → label list.

**This is a build-against-docs risk** (like the DFDC `metadata.json` vs
`dataset.json` caveat): the runbook's first step is "inspect one label file and
confirm `SPOOF_TYPE_INDEX` / path layout; adjust the constant if needed." The
fixture encodes the assumed format so the logic is fully tested; only the
real-file binding is unverified until download.

## 6. File manifest

| File | Action |
|---|---|
| `pad-synth-face/src/pad_synth_face/celeba_spoof.py` | **Create** |
| `pad-synth-face/src/pad_synth_face/real_attack.py` | **Modify** (add optional `subject_id_fn` param; 1 line at the id site) |
| `scripts/prepare_celeba_spoof.py` | **Create** |
| `pad-synth-face/tests/test_celeba_spoof_ingest.py` | **Create** |
| `pad-synth-face/tests/test_real_attack.py` | **Modify/Create** (add `subject_id_fn` + back-compat test) |
| `docs/celeba-spoof-b1.md` | **Create** |

No change to the B1 runner, the model zoo, or `train_and_cross_domain_eval`.

## 7. Test strategy

TDD. Fixtures are generated images in a tree mimicking CelebA-Spoof
(`Data/train/<subj>/{live,spoof}/*.jpg` + a small label file with spoof codes
incl. 5/6 to prove exclusion) — **no real or licensed faces** enter the repo
(`datasets/` gitignored regardless). Tests:
- staging maps codes → correct classes; 5/6 excluded; subjects preserved;
  symlinks created; `max_subjects` honoured.
- `ingest_real_attack` with `subject_id_fn` writes person ids; with `None`
  writes the old path id (back-compat) — and existing `ingest_real_attack`
  tests still pass.
- end-to-end on fixture: stage → ingest → manifest has person-id
  `bonafide_source.id` + correct labels; then a tiny B1 `run_curve` on the
  fixture confirms the full chain runs and the split is person-disjoint.

## 8. Success criteria

- `stage_celeba_spoof` produces the correct `bonafide/attack/<type>` symlink
  tree with the locked mapping; tests green.
- `ingest_real_attack` gains the optional `subject_id_fn` with verified
  back-compat; existing tests unchanged/green.
- End-to-end fixture run yields a canonical `_real_attack`-shaped dataset whose
  manifest carries person ids; a fixture B1 run completes.
- Inert/ready: the documented prepare→B1 sequence is one flow once CelebA-Spoof
  is downloaded, and N=0 reproduces ≈0.40 for direct comparison.
