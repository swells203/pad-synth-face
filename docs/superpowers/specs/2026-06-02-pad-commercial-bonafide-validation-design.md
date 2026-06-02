# PAD — Commercial-Bonafide Retrain Validation Harness (design)

**Date:** 2026-06-02
**Status:** approved (brainstorm) → ready for implementation plan
**Branch:** `feat/pad-commercial-bonafide-validation`

## 1. Problem & motivation

The PAD project is intended for **commercial deployment**, but every bonafide
source used so far is non-commercial research-only. DigiFace-1M — the current
bonafide — carries Microsoft's Research Use Data Agreement, whose "you may not
use the Data **or any Results** in any commercial offering" clause encumbers
*the trained models themselves*. So B2/A2 and every model trained to date are
not commercially shippable. (See memory `pad-commercial-licensing`.)

The project's *own* IP is clean: the print/replay/mask attack physics, the A2
sensor/capture pipeline, the eval harness, the configs. The encumbrance is
**only the bonafide face source**. The de-risk plan is therefore: swap the
encumbered DigiFace bonafide for a **commercially-licensed** bonafide set,
re-run the existing synthetic-attack pipeline on top, and confirm cross-domain
EER holds. This validates that a shippable model is achievable *before*
committing to a ~$10k commercial-data purchase — using free vendor samples
first.

This spec defines the harness that makes that validation a repeatable,
one-decision-per-step flow.

## 2. Goal & non-goals

**Goal:** Given a commercially-licensed bonafide face set, produce a clear
PASS/FAIL verdict on whether substituting it for DigiFace preserves
cross-domain EER, at matched scale, across the standard 18-cell L4 sweep.

**Non-goals:**
- Real *attacks* (commercial vendors bundle them, but this harness uses
  **only the commercial bonafide** + our synthetic attacks — "option A").
  Real-attack eval stays on the existing `ingest_real_attack` path, untouched.
- Procuring or pricing data (covered in the licensing memory).
- Any change to the attack physics, sensor pipeline, model factories, or
  `spark_sweep.py`. This harness is ingest + configs + a verdict step around
  the existing machinery.
- Auto-detecting arbitrary vendor layouts. We define one canonical input
  contract; per-vendor reshaping is a thin documented shim.

## 3. Decisions (locked in brainstorm)

| Decision | Choice |
|---|---|
| Harness goal | **Option A** — commercial bonafide + our synthetic attacks |
| Ingest input | **Canonical contract + tiny per-vendor shims** |
| Pass criterion | **Matched-scale A/B delta** (commercial vs DigiFace head-to-head) |
| Compute / scale | **Spark, full 18-cell L4 sweep** |

## 4. Architecture

Five units, each independently testable. Four mirror existing components; one
(the verdict script) is new logic.

```
free vendor sample
   │  (shim if needed → canonical <identity>/<sample>.<img>)
   ▼
prepare_commercial_bonafide.py ──► datasets/_real/commercial_224/<id>/NNN.png
   │                                + provenance.jsonl (LICENCE + source URL)
   ▼
pin_commercial_identities.py ──► configs/commercial_identities_set{a,b}.txt
   │
   ▼
configs/runs/commercial_set{a,b}_d{1,2,3}.yaml  (print+replay @224)
   │  (generate datasets locally → rsync → spark_sweep.py, unchanged)
   ▼
runs_commercial_224_L4/   (+ existing runs_mix_224_L4_A2/ = DigiFace baseline)
   │
   ▼
compare_bonafide_eer.py ──► per-cell Δ table + PASS/FAIL verdict (exit code)
```

### 4.1 `scripts/prepare_commercial_bonafide.py` (clone-ish of prepare_digiface + provenance)

- **Canonical input contract:** `<src>/<identity>/<sample>.{png,jpg,jpeg}`,
  one directory per subject.
- Resizes each image to 224 (LANCZOS, reusing `prepare_digiface`'s logic),
  writes `datasets/_real/commercial_224/<identity>/NNN.png`.
- **Records licence provenance** — required args `--src`, `--license`,
  `--source-url`; optional `--vendor`, `--max-per-identity`. Writes
  `provenance.jsonl` and `_meta.json` (target size, identity count, sample
  counts, licence, source URL, vendor) — mirroring `prepare_dfdc` /
  `prepare_real_attack`. This is mandatory: the entire point is commercial
  compliance, so the licence string must travel with the data.
- Idempotent (skips existing destination files), like `prepare_digiface`.

### 4.2 `scripts/pin_commercial_identities.py` (clone of pin_dfdc_identities.py)

- Reads `datasets/_real/commercial_224`, deterministic seeded shuffle, writes
  `configs/commercial_identities_seta.txt` (8) and `…setb.txt` (16),
  identity-disjoint. Errors cleanly if `<24` identities ingested.
  Override `--root`, `--seta-count`, `--setb-count`, `--seed`.

### 4.3 Six sweep configs (clones of real_set*)

`configs/runs/commercial_set{a,b}_d{1,2,3}.yaml`, each an exact mirror of its
`real_<set>_d<n>.yaml` counterpart — preserve seeds (Set A 20260522, Set B
20260523), `samples_per_bonafide` (Set A 6/32/256, Set B 4/32/256),
print+replay attacks, per-set sensor preset (Set A mobile-front-2024, Set B
webcam-1080p) — changing only `bonafide.root → ./datasets/_real/commercial_224`
and `bonafide.identities_file → ./configs/commercial_identities_set{a,b}.txt`.
Because these match `mix_set*` scale, the commercial sweep is directly
comparable to the existing DigiFace `runs_mix_224_L4_A2` baseline.

### 4.4 Sweep (reuse, no new code)

Generate the 6 datasets locally → rsync to Spark → `spark_sweep.py` over the L4
cells → `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_commercial_224_L4/`.
Identical procedure to the DFDC/A2 plan, `commercial_` prefixes.

### 4.5 `scripts/compare_bonafide_eer.py` (NEW — the verdict)

- Args: `--commercial-dir` (required), `--baseline-dir`
  (default: `…/runs_mix_224_L4_A2`), `--band 0.03`, `--collapse 0.001`.
- Reads per-cell JSON from both dirs; groups by `(capacity, data_level)`;
  computes mean ± std `xdomain_eer` across seeds (same aggregation as the A2
  report).
- Renders a table: `cell | DigiFace EER | Commercial EER | Δ | verdict`.
- **PASS gate (all must hold):** every cell `|Δ| ≤ band` AND no commercial cell
  `xdomain_eer ≤ collapse`. Prints `PASS — commercial bonafide ships` or
  `FAIL — <offending cells>`; **exits non-zero on FAIL** (CI-able).
- **Matched-scale guard:** if the two sweeps' configs used different
  `samples_per_bonafide` for a cell (read from each cell JSON if present, else
  inferred), prints a WARNING that the comparison is not matched-scale rather
  than silently comparing apples to oranges.

### 4.6 Tests + doc

- **Ingest test:** in-test fixture — a tiny `<id>/<sample>` tree (3 identities ×
  2 generated images, no real faces) → assert canonical 224 root produced +
  `provenance.jsonl` contains the licence string + source URL.
- **Verdict test:** two hand-authored mini sweep dirs — one PASS (small Δ), one
  FAIL (a cell with Δ>band and a collapsed cell) → assert the printed verdict
  and the process exit code for both.
- **`docs/commercial-bonafide.md`:** obtain free vendor sample → reshape via
  shim if needed → ingest (with licence) → pin identities → generate + sweep →
  `compare_bonafide_eer.py`. Calls out the licence-provenance step and the
  matched-scale caveat for small samples (reduce `samples_per_bonafide`
  symmetrically in both sweeps).

## 5. Matched-scale reconciliation

The brainstorm picked *matched-scale delta* AND *full 18-cell sweep*. These
compose when the commercial set is large enough for D3 (256 samples/bonafide):
the commercial 18-cell sweep then compares directly against the existing
DigiFace `runs_mix_224_L4_A2` baseline at identical scale. For a small free
sample that can't reach D3, the operator reduces `samples_per_bonafide`
symmetrically in both the commercial configs and a re-run DigiFace sweep; the
verdict script's matched-scale guard enforces that they actually match. Paid
set → full 18 cells vs the committed baseline; free sample → reduced-D matched
pair.

## 6. File manifest

| File | Action |
|---|---|
| `scripts/prepare_commercial_bonafide.py` | **Create** |
| `scripts/pin_commercial_identities.py` | **Create** |
| `scripts/compare_bonafide_eer.py` | **Create** |
| `configs/runs/commercial_set{a,b}_d{1,2,3}.yaml` (×6) | **Create** |
| `pad-synth-face/tests/test_commercial_bonafide_ingest.py` | **Create** |
| `tests/test_compare_bonafide_eer.py` | **Create** |
| `docs/commercial-bonafide.md` | **Create** |

No existing files modified. `prepare_digiface.py` resize logic is *reused*
(imported or factored to a shared helper if clean; otherwise duplicated — a
~15-line LANCZOS loop, duplication acceptable to avoid coupling).

## 7. Test strategy

TDD per unit: write the failing ingest/verdict tests first, implement to green.
The sweep itself is operational (no unit test — exercised by the run). The
fixture uses generated images only — **no real or licensed faces ever enter the
repo** (`datasets/` is gitignored regardless).

## 8. Success criteria

- `prepare_commercial_bonafide.py` ingests the canonical contract and records
  licence provenance; test green.
- The 6 configs validate against the pipeline schema (as the DFDC configs did).
- `compare_bonafide_eer.py` produces a correct per-cell Δ table and PASS/FAIL
  verdict with correct exit codes; test green.
- End-to-end (when data is staged): a single documented sequence takes a vendor
  sample to a verdict.
- The harness is inert until commercial data is staged — nothing here depends on
  obtaining data, so it ships now as pure scaffolding (like the DFDC prep).
