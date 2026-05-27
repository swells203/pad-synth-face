% PAD Real-Attack Capture Harness Design
% A dataset-agnostic ingester + synth→real eval wiring so a real-attack PAD dataset can plug into the v2.1+DigiFace+mask production base — harness first, dataset later
% 2026-05-27

---

## 1. Purpose and audience

Every prior sweep in this project is synthetic-on-synthetic (train Set A, eval Set B, both procedurally generated). The mask sweep (2026-05-27) confirmed the recurring ceiling: the bonafide-side confound was broken by DigiFace, but the **attack side is still synthetic**, and the residual at D3 plus L3's in-domain memorisation point at that synthetic-attack ceiling. The roadmap's top Phase 2.5 lever is **real-attack data**: train on synthetic, evaluate on real attacks — a true generalisation test (`docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` §Phase 2.5).

Public real-attack PAD datasets (MSU-MFSD, OULU-NPU, Replay-Attack, CASIA-FASD, 3DMAD, SMAD, HiFiMask) are almost all **EULA/licence-gated** — none is freely downloadable the way DigiFace-1M (MIT, no-auth) was. So this spec builds the **harness first**: a dataset-agnostic ingester and the synth→real eval wiring, validated end-to-end on a procedural fixture, ready for a real dataset to plug in the moment one is obtained.

Audience: future maintainers and whoever lands the first real dataset.

## 2. Scope boundary and success criteria

**This cycle ships the harness, not a synth→real EER number** — there is no real data to measure yet.

Done means:
- `prepare_real_attack.py` ingests a folder-convention source into the canonical 64×64 eval layout with manifest + licence provenance.
- A procedural `build_fixture_real_attack` fixture exercises the full path with zero real data.
- Tests prove: ingestion produces the correct canonical layout/labels/manifest/provenance, is deterministic and idempotent, and that `train_and_cross_domain_eval(train_root=synth, eval_root=ingested_real)` runs and reads both classes from the real eval.
- A short doc records the folder convention + the exact prepare/sweep commands.

**Out of scope:** physical capture; video→frame extraction (a user pre-step); per-dataset native-layout adapters (the folder convention *is* the contract); the EULA dataset itself and its real EER number (a documented follow-up).

## 3. Input contract (folder convention)

The ingester reads a source tree the user (or a tiny per-dataset prep step) arranges as:

```
<src>/
  bonafide/**/*.{jpg,jpeg,png}          # real genuine-presentation frames
  attack/<attack_type>/**/*.{jpg,jpeg,png}   # real attack frames, grouped by type
```

`<attack_type>` is a free string preserved verbatim (e.g. `print`, `replay`, `mask`, `video_replay`). Recursive glob, so per-subject subdirectories are fine. Input is **extracted image frames** — video decoding is the user's pre-step.

## 4. Canonical output (matches every other dataset)

```
<out>/
  face/bonafide/real-bonafide-00000000.jpg     # 64×64 RGB, JPEG q92
  face/<attack_type>/real-<attack_type>-00000000.jpg
  manifest.jsonl                                # one SampleRecord per image
  provenance.jsonl                              # RealAttackIngested event
```

This is exactly the layout `pad_synth_core.eval.baseline.TinyPADDataset` already consumes (`face/bonafide/` = label 0; every other `face/<x>/` = label 1), so the sweep reads it with no changes. Images are resized to 64×64 LANCZOS (apples-to-apples with the synthetic base), QC'd via the existing `check_image_basic((64,64,3))`.

## 5. Components and files

| Change | File | Responsibility |
|---|---|---|
| New ingester script | `scripts/prepare_real_attack.py` | folder-convention `<src>` → canonical `<out>`; resize, manifest, provenance; deterministic, idempotent |
| New provenance event | `pad-synth-core/src/pad_synth_core/provenance.py` | add `RealAttackIngested` (mirrors `BonafideIngested`) to the event union |
| New fixture | `pad-synth-face/src/pad_synth_face/_fixtures.py` | `build_fixture_real_attack(root)` — procedural real-like source in the folder convention |
| Tests | `pad-synth-face/tests/test_prepare_real_attack.py` | ingestion correctness + synth→real wiring |
| Doc | `docs/real-attack-capture.md` | folder convention + prepare/sweep commands + no-commit policy |

`RealAttackIngested` fields: `name: str`, `license: str`, `source_url: str`, `sha256_of_index: str` (hash of the sorted ingested relative-path list), `attack_types: list[str]`, `ingested_at: datetime` (default-factory now). Added to the `ProvenanceEvent` union.

`prepare_real_attack.py` CLI: `--src`, `--out`, `--dataset-name`, `--license`, `--source-url`, optional `--max-per-class` (cap for quick smoke ingests). It reuses `ManifestWriter`, `SampleRecord`, `ProvenanceLedger`, and `check_image_basic`. Each `SampleRecord` carries `label` (`bonafide`/`attack`), `attack_type` (None for bonafide, else the subdir name), `output_path`, `output_sha256`, and a `dataset`/`license` tag. Idempotent: existing sample IDs in the manifest are skipped on re-run; deterministic ordering (sorted source paths) so IDs are stable.

## 6. Synth→real eval wiring (no new sweep code)

The existing `scripts/spark_sweep.py` already accepts arbitrary train/eval roots per data level. Synth→real is a documented invocation, not new code:

- `--set-a-d{1,2,3}` → the synthetic **production base**, Set A: `datasets/mix_seta_d{1,2,3}` (v2.1-print + DigiFace bonafide + replay + mask — the full integrated attack base).
- `--set-b-d{1,2,3}` → **all three pointed at the single ingested real dir** (`datasets/_real_attack/<dataset>`), since the real eval set has no synthetic data-levels.
- `--set-a-d4`/`--set-b-d4` satisfy the required arg by reusing the D3 paths; request only D1–D3 cells.

Result: "train on synthetic at increasing data scale, evaluate on a fixed real set" — the headline synth→real generalisation curve, directly comparable to the synth→synth tables in the existing report. The number is produced once a real dataset is ingested; the harness and command are delivered now.

## 7. Data handling (licence / PII)

Ingested real datasets are written under `datasets/_real_attack/<dataset>/`, already covered by the existing gitignored `datasets/`. **Real images are never committed.** Only the script, fixture, tests, and doc are committed. The `RealAttackIngested` provenance event records the licence string so every ingested dataset self-documents its terms.

## 8. Testing

`pad-synth-face/tests/test_prepare_real_attack.py`:
- **Ingestion layout:** fixture source → `<out>/face/bonafide/` and `<out>/face/<type>/` exist with the expected per-class counts; images are 64×64×3.
- **Manifest:** one record per image; bonafide records have `label="bonafide"`/`attack_type=None`; attack records have `label="attack"` and the correct `attack_type`.
- **Provenance:** a `RealAttackIngested` event is written with the passed `--dataset-name`/`--license` and the discovered `attack_types`.
- **Idempotent + deterministic:** a second run generates 0 new samples; sample IDs are stable across runs.
- **Synth→real wiring:** build a tiny synthetic train dataset via the existing fixture pipeline, ingest the fixture real source, then `train_and_cross_domain_eval(train_root=synth, eval_root=real)` on CPU returns a finite cross-domain EER and a non-zero `n_val_cross_domain` (proves both real classes are read).

## 9. Open follow-up (documented, not built)

Once a licensed real-attack dataset is obtained: extract frames, arrange under the folder convention, run `prepare_real_attack.py`, then the §6 sweep. Append the synth→real curve to the sweep-results report as the new headline metric.
