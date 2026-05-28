% PAD DFDC-Grounded Bonafide Ingest Design
% Use real DFDC face frames as the bonafide source — replace DigiFace's procedural-but-photorealistic faces with genuine in-the-wild captures, so the existing synth attack physics rides on a real-face distribution. Harness now, dataset later.
% 2026-05-28

---

## 1. Purpose and audience

The 2026-05-27 synth→real pilot established that the synthetic-trained detector collapses to chance on real attacks (cross-domain EER 0.55–0.68 — at or worse than chance). The agreed lever queue (`pad-next-sub-projects` memory) puts capture-domain realism first (A1+A2), but the user proposed an orthogonal high-value move: feed the existing synthetic attack physics with **real high-resolution face frames** as the bonafide source, instead of DigiFace's synthetic-but-photoreal procedural faces. Meta's **DFDC (Deepfake Detection Challenge) dataset** is the natural source — ~100k clips from 3,426 paid actors, well-licensed for research.

This spec ships the ingest **harness** that turns DFDC video chunks into a DigiFace-shaped bonafide root the existing pipeline can swap in unchanged. It is the harness-first companion to the real-attack-capture sub-project (`docs/superpowers/specs/2026-05-27-pad-real-attack-capture-design.md`) — same pattern: build now against a fixture, the actual DFDC ingest waits on the user obtaining the dataset (EULA-gated).

Audience: future maintainers and whoever lands the first DFDC ingest. The deferred deepfake-attack-class follow-up will build on this same infrastructure.

## 2. Scope boundary and success criteria

**This cycle ships the harness, not a sweep number** — DFDC is EULA-gated and there is no dataset to measure on yet.

Done means:
- `extract_dfdc_bonafide(...)` reads a DFDC source tree (videos + per-chunk `metadata.json`), extracts face crops from REAL videos via MediaPipe Face Detection, and writes a canonical identity-per-directory layout that `DigiFaceLoader` consumes unchanged.
- A procedural `build_fixture_dfdc` fixture (small synthesized mp4s + a metadata.json) exercises the full pipeline with no real video data and no PII.
- Tests prove: layout/counts correct at the requested resolution, provenance event recorded with the license string, idempotent re-run, integration with `DigiFaceLoader` and `run_pipeline` succeeds end-to-end.
- A short doc records the source-folder convention, the prepare command, the sweep-swap recipe, and the no-commit policy.

**Out of scope:** the deepfake attack class (separate follow-up spec built on this infrastructure); the A1+A2 resolution bump (parameterized here via `--res`, default 64×64; consumed downstream by A1's synth-side retune); the DFDC download itself (user blocker); any sweep number (waits on dataset).

## 3. Input contract — DFDC chunk layout

Each DFDC chunk (~10 GB zipped) extracts to a flat folder of videos plus a per-chunk metadata.json:

```
<src>/
  <chunk_dir>/
    metadata.json          # {video_filename: {label: REAL|FAKE, original: <REAL filename if FAKE>}, ...}
    <video_filename>.mp4   # one per entry
    ...
  ...                      # more chunk_dirs alongside
```

The ingester recursively discovers any `metadata.json` under `<src>/`, processes the videos in its sibling directory, and filters to entries with `label == "REAL"` for this cycle. The `original` field on FAKE entries is recorded in provenance only — the deferred deepfake-attack-class follow-up will consume it for actor-grouped subject-disjoint splits.

## 4. Canonical output — DigiFace-shaped bonafide root

```
<out>/                                # default datasets/_real/dfdc_<res>/
  <video_stem>/                       # one identity per REAL video filename
    000.png                           # frames are 0-padded indices
    001.png
    ...
  manifest.jsonl                      # per-frame audit (SampleRecord-shaped)
  provenance.jsonl                    # DFDCBonafideIngested event
```

This is exactly the layout `pad_synth_face.bonafide.DigiFaceLoader` already consumes — identities-as-directories, image files inside. No loader changes. Existing `run_pipeline` reads it unchanged when `bonafide.root` points here.

## 5. Components and files

| Change | File | Responsibility |
|---|---|---|
| New ingester | `pad-synth-face/src/pad_synth_face/dfdc.py` | `extract_dfdc_bonafide(src, out, res, frames_per_video, max_videos, detector)` — pure function with injectable detector |
| New provenance event | `pad-synth-core/src/pad_synth_core/provenance.py` | `DFDCBonafideIngested` (license, n_chunks, n_videos, n_frames_written, detection_rate, real_filenames_sha256) |
| Fixture | `pad-synth-face/src/pad_synth_face/_fixtures.py` | `build_fixture_dfdc(root)` — synthesized small mp4s + chunk metadata.json, no PII |
| CLI | `scripts/prepare_dfdc.py` | Thin shim over `extract_dfdc_bonafide` |
| Tests | `pad-synth-face/tests/test_dfdc_ingest.py` | layout, counts, provenance, idempotency, detection-failure path, integration |
| Identity lists | `configs/dfdc_identities_set{a,b}.txt` | Reserved filenames; populated after first real ingest (committed empty placeholders in this cycle? — see §10) |
| Run configs | `configs/runs/dfdc_set{a,b}_d{1,2,3}.yaml` | Mirror `real_set*` but with DFDC bonafide root and DFDC identity-list files |
| Doc | `docs/dfdc-bonafide.md` | Source convention, prepare command, sweep-swap recipe, no-commit policy, EULA pointer |
| New optional dep | `pyproject.toml` | `mediapipe` under a new `dfdc` extra (keeps `eval` extra slim) |

## 6. Ingest pipeline (`extract_dfdc_bonafide`)

Pure function; injectable `detector` (default constructed inside if `None`) so tests use a stub.

For each `metadata.json` discovered under `<src>` (sorted), for each `(video_filename, label)` entry with `label == "REAL"` (capped at `max_videos` total):

1. **Extract `frames_per_video` frames** from `<chunk_dir>/<video_filename>` via `ffmpeg -ss <t_i> -vframes 1 ...` at evenly spaced timestamps (default `frames_per_video=6`).
2. **Detect the highest-confidence face** in each frame via the injected `detector(frame_rgb) -> bbox | None`. Default detector wraps MediaPipe Face Detection.
3. **Square-crop with margin** (default `crop_margin=1.3`): take the bbox, expand to a square around its center scaled by `crop_margin`, clip to frame bounds.
4. **Resize to `res × res` LANCZOS**; QC via the existing `check_image_basic((res, res, 3))`; skip on QC fail.
5. **Write** `<out>/<video_stem>/<NNN>.png` (3-digit zero-padded frame index). Append a `SampleRecord` to `manifest.jsonl` (bonafide_source = DFDC, id = `<video_stem>`).
6. **Track** total frames attempted vs written → `detection_rate` recorded in provenance.

Idempotent: existing identity dirs are skipped (a dir with at least one .png is considered done). Deterministic: sorted video order, fixed timestamp grid, no rng.

## 7. Detector injection (testability + future-proofing)

```python
DetectorFn = Callable[[np.ndarray], tuple[int, int, int, int] | None]  # (x, y, w, h) | None
```

`extract_dfdc_bonafide(..., detector: DetectorFn | None = None)`:
- If `detector is None`, construct a default MediaPipe wrapper lazily (imports `mediapipe` only on first use; not imported at module top, so the package imports cleanly even when the `dfdc` extra isn't installed).
- Tests pass a stub `lambda frame: (cx-32, cy-32, 64, 64)` that returns a fixed bbox in the frame center — byte-deterministic outputs, no PII.

This makes the harness testable without ever invoking MediaPipe in the test suite, AND lets a future user swap detectors (YuNet, MTCNN, …) without forking the ingester.

## 8. Data handling — DFDC is licence-gated

- DFDC frames are written under `datasets/_real/dfdc_<res>/`, covered by the existing gitignored `datasets/`. **DFDC frames are never committed.**
- `DFDCBonafideIngested` provenance event records the licence string the caller passes (e.g. `"DFDC research licence (Meta AI)"`), the source chunk paths, video counts, and a SHA-256 of the sorted REAL-filename list — a stable fingerprint of what was ingested.
- The doc explicitly states the EULA must be obtained before running the ingester.

## 9. Subject identity (this cycle: per-video; follow-up: per-actor)

This cycle treats each REAL video filename as one identity (one directory per video). This matches the DigiFace identity-per-dir contract exactly.

DFDC's `original` field links each FAKE video to its source REAL — i.e. actor-grouping is implicit in the FAKE→REAL graph but absent from REAL→REAL records. The deferred deepfake-attack-class follow-up will use this graph to derive actor identities (REAL videos sharing the same actor get the same subject id) by re-reading the source `metadata.json` files; nothing extra is materialised this cycle. The provenance event's `real_filenames_sha256` field is a stable fingerprint of which REALs were ingested.

## 10. Configs (deferred to ingest time)

No `dfdc_set*` configs or identity-list files are committed this cycle — they would reference identities that don't exist until the user has ingested DFDC, and committing stubs that fail-fast on missing files adds noise without value. Instead, §12 below provides the exact YAML template (sibling-of-`real_set*` with two field changes) plus the identity-pinning recipe; the user creates `configs/runs/dfdc_set{a,b}_d{1,2,3}.yaml` and `configs/dfdc_identities_set{a,b}.txt` once after the first ingest and commits them then.

## 11. Testing

`pad-synth-face/tests/test_dfdc_ingest.py`:
- **Layout**: fixture source → `<out>/<video_stem>/*.png` per REAL video, count matches `frames_per_video × n_real_videos`, every image is `res × res × 3`.
- **Default-res drop-in**: ingested fixture's identity dirs are consumable by `DigiFaceLoader` without changes.
- **Manifest**: one `SampleRecord` per written frame; `bonafide_source.dataset == "DFDC"`, `id == <video_stem>`, license string preserved.
- **Provenance**: `DFDCBonafideIngested` event recorded with the right counts and license.
- **Idempotent**: re-run writes 0 new frames; existing identity dirs untouched.
- **Detection-failure path**: stub detector returning `None` → frames skipped, `detection_rate` reflects the skip; identity dir empty (and that identity excluded from the loader-readable count).
- **Integration**: `run_pipeline` with `bonafide.root` pointed at the ingested fixture root produces a synthetic dataset successfully (smoke run with `samples_per_bonafide=1` and the existing print ontology).

Tests use the injected-detector stub — MediaPipe is never imported during the test suite. The `mediapipe` dep can be missing and the suite still passes (the only consumer is the default detector factory, which is constructed lazily and only when no stub is passed).

## 12. The user's blocker (documented)

Once you obtain DFDC (Preview ~5 GB easiest first step, full ~470 GB later):

1. Unzip chunks anywhere on the laptop.
2. `pip install -e '.[dfdc]'` to add MediaPipe.
3. `python scripts/prepare_dfdc.py --src /path/to/dfdc/chunks --out datasets/_real/dfdc_64 --license 'DFDC research licence' --source-url '<the URL>'`.
4. Pick identities for Set A/B (script utility) → commit the two `dfdc_identities_set*.txt` files.
5. Run the existing sweep using `dfdc_set{a,b}_d*.yaml` configs.
6. Append the cross-domain table to the report — the headline question is whether "real high-res bonafide + synth attacks" beats the v2.1+DigiFace baseline (mask-only L3·D3 ≈ 0.089, integrated L2·D3 ≈ 0.094) at 64×64. If yes, the next cycle is the resolution bump (A1+A2) which compounds with this.
