% PAD Real-Bonafide Integration Design (DigiFace-1M)
% Replace the procedural-blob bonafide source with photographic-realism faces; the v2.1 result's recommended top-priority lever
% 2026-05-22

---

## 1. Purpose and audience

The v2.1 sweep result showed that even per-sample geometric jitter does not break the binary-halftone palette artifact — 6/9 cross-domain cells still hit 0.000 EER, identical to v2. The most likely remaining mechanism: the detector latches onto the ~16-color quantized palette of halftoned output, which is uniformly identical across Set A and Set B because both use the same procedural-blob bonafide source. Real photographic-realism face textures would cover that palette under per-pixel color diversity, breaking the shortcut.

This spec wires **DigiFace-1M** (Microsoft Research synthetic-but-photo-realistic face dataset, MIT-licensed) as a bonafide source, regenerates the existing v2.1 datasets pointing at it, and re-runs the 27-cell sweep. The deliverable answers: **does shifting the bonafide source from procedural blobs to photographic-realism faces break the artifact?**

Audience: future maintainers, the Phase 2.5 author. This is the originally-deferred Phase 1 scope item — finally landing it because v2.1's "two-strikes" finding promoted it to dominant Phase 2 lever.

## 2. The question this answers

| v2.1-on-real-bonafide cross-domain EER | Diagnosis | Phase 2 implication |
|---|---|---|
| **No cell hits 0.000** (artifact broken by real-face textures) | The binary-palette artifact was the dominant signal, and real-face textures cover it. v2.1 physics IS the lever once the synthetic-bonafide confound is removed. | Real-bonafide is now the production baseline; ship v2.1 physics + DigiFace bonafide as the next iteration. Mask attack sub-project proceeds on this combined base. |
| **Some cells still hit 0.000** (artifact survives) | The binary-threshold output itself produces a palette-independent signature (e.g., sharp 0/1 channel transitions or CMYK-conversion artifacts) that survives both synthetic and real bonafide. | Synthetic halftoning of any form is unsuitable as production training data. Either move to v2.2 with gray-level dot coverage, OR pivot to real attack capture, OR accept the production model is best trained on real-data alone. |
| **All cells dramatically improve, no 0.000s** AND cross-domain EER lands in a plausible non-trivial band (e.g., 0.10–0.30 across cells) | Both the artifact was real and the physics is genuinely useful. | Strongest possible outcome — proceed to mask-attack with real bonafide. |

Decision rule: same as v2.1 (`no cell ≤ 0.001 EER` = artifact broken).

## 3. Dataset

**DigiFace-1M, 118k aligned subset.** Microsoft Research-published, MIT license, ~118k images across ~10k identities at 112×112 RGB. ~few hundred MB compressed; not the ~6GB full release.

Source: Microsoft Research's published DigiFace-1M release. The implementation plan resolves the exact URL/release-tag at download time; the spec only requires "the 118k aligned subset, MIT-licensed."

**On-disk layout:** `<root>/<identity_id>/<sample_index>.png` — same shape as the existing procedural fixtures, so the existing `DigiFaceLoader` works unchanged. (DigiFace's spec literally inspired the fixture layout; this round-trips.)

**License compliance:** The provenance ledger's existing `BonafideIngested` event captures `name="DigiFace-1M"`, `license="MIT"`, `source_url=<url>`, `sha256_of_index=<deterministic-hash-of-identity-list>`. No new provenance code needed — it's already wired in `pad-synth-face/src/pad_synth_face/pipeline.py`.

## 4. Resolution preprocessing

DigiFace's native 112×112 differs from the existing fixture/sweep resolution of 64×64. To keep apples-to-apples comparison with the synthetic v2.1 sweep, **downsample DigiFace to 64×64** in a one-time preprocessing step.

A new helper script `scripts/prepare_digiface_64.py` reads `datasets/_real/digiface_118k_raw/` and writes `datasets/_real/digiface_118k_64/` with each image PIL-resampled to 64×64 (`Image.LANCZOS` for quality). Output layout preserves `<root>/<identity>/<sample>.png`. The script is deterministic (no RNG), idempotent (skips identities already written), and writes a `_meta.json` recording the source SHA, target resolution, and number of identities/samples.

The configs in §5 point at the *resized* directory (`datasets/_real/digiface_118k_64/`), not the raw one.

## 5. Identity selection

Use the existing `DigiFaceLoader.identity_disjoint_split(seed, ratios)` method to deterministically pick identities. For Set A (8 IDs) and Set B (16 IDs), call:

- Set A's 8 identities: first 8 from `identity_disjoint_split(seed=20260522, ratios=(0.5, 0.0, 0.5))[0][:8]` (or equivalent — exact selection logic pinned in the plan).
- Set B's 16 identities: 16 IDs *disjoint from Set A* — drawn from the test partition or a separate seeded split.

**Reproducibility safeguard:** the operational task writes `configs/digiface_identities_seta.txt` and `configs/digiface_identities_setb.txt` listing the selected identity names. These files ARE committed (small text, no PII) and the generation configs reference them. This makes the selection robust to download-mirror identity ordering differences and reproducible without re-running the disjoint-split logic.

## 6. New configs (six)

Under `configs/runs/`:

| File | Set | bonafide.root | samples_per_bonafide | Seed |
|---|---|---|---|---|
| `real_seta_d1.yaml` | A | `./datasets/_real/digiface_118k_64` | 6 | 20260522 |
| `real_seta_d2.yaml` | A | `./datasets/_real/digiface_118k_64` | 32 | 20260522 |
| `real_seta_d3.yaml` | A | `./datasets/_real/digiface_118k_64` | 256 | 20260522 |
| `real_setb_d1.yaml` | B | `./datasets/_real/digiface_118k_64` | 4 | 20260523 |
| `real_setb_d2.yaml` | B | `./datasets/_real/digiface_118k_64` | 32 | 20260523 |
| `real_setb_d3.yaml` | B | `./datasets/_real/digiface_118k_64` | 256 | 20260523 |

Each `real_*` config follows the same shape as the existing `v21_*` configs (run/modality/bonafide/attacks/sensor_preset blocks), with two changes vs. v2.1:

- `bonafide.root` points at `./datasets/_real/digiface_118k_64` instead of `./datasets/_fixtures/extended_fixture` / `./datasets/_fixtures/digiface`.
- A new optional key `bonafide.identities_file` names the pinned identity-list file (`./configs/digiface_identities_seta.txt` for Set A configs; `./configs/digiface_identities_setb.txt` for Set B). The pipeline reads the file's lines as `restrict_to` for the loader. The exact YAML form is pinned in the plan.

Sample counts (same as v2.1): Set A = 96 / 512 / 4096; Set B = 128 / 1024 / 8192.

## 7. Loader & pipeline integration

The existing `DigiFaceLoader` (in `pad-synth-face/src/pad_synth_face/bonafide.py`) is already written for the `<root>/<identity>/<sample>.png` layout. Two small extensions are needed:

1. **Glob both `.png` and `.jpg`**: currently `samples_for_identity` globs `*.png` only. DigiFace's 118k release ships PNG so this technically works, but extending to also pick up `*.jpg` is a one-line robustness fix that costs nothing and helps if a future mirror ships JPEG.
2. **Restrict-to-identity-list parameter** on `__init__` or `list_identities`: when configs pin a specific identity list, the loader must restrict iteration to it (vs. enumerating the whole `<root>`). A new optional `restrict_to: list[str] | None = None` parameter on `DigiFaceLoader.__init__` accomplishes this — filters `list_identities()` to the intersection with `restrict_to`.

The existing pipeline (`pad-synth-face/src/pad_synth_face/pipeline.py`) instantiates `DigiFaceLoader(Path(cfg["bonafide"]["root"]))`. After this change, it'll read an optional `cfg["bonafide"]["identities_file"]` and pass that file's lines as `restrict_to`. One small modification to `pipeline.py` — the originally-untouched-since-Phase-1 file.

## 8. Measurement plan

After datasets generate:

1. **rsync** the 6 real-bonafide datasets to the Spark.
2. **Run a 27-cell sweep** on the Spark: 3 capacities × 3 D-levels × 3 seeds. **v2.1 physics only** (latest, current production print attack). Same `spark_sweep.py`, same hyperparameters as the parent v2.1 sweep.
3. **rsync results back**, build `summary_real.csv`, append a **"real-bonafide v2.1 result"** section to the existing results report comparing **synthetic-bonafide v2.1 → real-bonafide v2.1** per cell with the artifact verdict per §2.
4. One-line decisions/roadmap update.

The synthetic-bonafide v2.1 numbers from the just-merged work are the comparison column.

## 9. Architecture / component boundaries

- **Modified in place:** `pad-synth-face/src/pad_synth_face/bonafide.py` (loader gains `restrict_to` param + .jpg glob).
- **Modified in place:** `pad-synth-face/src/pad_synth_face/pipeline.py` (pass `identities_file` through to loader; the originally-deferred Phase-1 integration finally lands here — and incidentally also fixes the `ontology_version` hardcode that's been flagged in v2 and v2.1 reports, since we're now editing this file anyway and it's a one-line dynamic-version fix). The hardcode-fix scope is explicitly included here as a small ride-along since pipeline.py is being modified for the real-bonafide work.
- **Modified in place:** `tests/golden/golden_hashes.json` — regenerated *only if* the pipeline.py change shifts the procedural-fixture-based golden test. The hardcode fix means `ontology_version` in manifest rows changes from `"2026-05-11"` to the actual ontology version (`"2026-05-23"` post-v2.1). That changes the manifest BYTES, which is what the golden hashes the *output JPEG files* on — JPEG files don't include `ontology_version` so they should be unchanged. Verify: if golden passes after the pipeline.py fix, great; if it fails, regenerate.
- **New:** `scripts/fetch_digiface_subset.py` (or `.sh`) — downloads + verifies the DigiFace-1M 118k subset to `datasets/_real/digiface_118k_raw/`.
- **New:** `scripts/prepare_digiface_64.py` — resizes raw → `datasets/_real/digiface_118k_64/`.
- **New:** `configs/digiface_identities_seta.txt`, `configs/digiface_identities_setb.txt` — pinned identity selections (committed).
- **New (×6):** `configs/runs/real_seta_d{1,2,3}.yaml`, `configs/runs/real_setb_d{1,2,3}.yaml`.
- **New tests:** at minimum `pad-synth-face/tests/test_bonafide_restrict.py` (loader respects `restrict_to`), `pad-synth-face/tests/test_real_configs.py` (configs well-formed). Existing `test_bonafide.py` tests must still pass unchanged.
- **Append-only:** updates to the results report + decisions/roadmap.

ICC, halftone (v2.1), texture, perspective, cutout, replay, sensor — all untouched. v2.1 physics is the active print attack for this experiment; no physics changes.

## 10. Explicit non-goals

- **No real attack capture.** Real prints / real screen replays are a separate sub-project; this iteration is real BONAFIDE only.
- **No v1 or v2 physics re-runs on real bonafide.** Start with the latest (v2.1) for the most informative single experiment; expand only if results justify it.
- **No full DigiFace-1M (~6GB).** The 118k subset is sufficient for 27 cells × 8-16 identities.
- **No new ontology axes, no ontology version bump.** This is a data-source change, not a physics change.
- **No new sensor presets, no model architecture changes.**
- **No v2.2 (gray-level halftoning).** Stays a possible follow-up if real-bonafide doesn't break the artifact.
- **No FFHQ / CelebA / VGGFace2.** Those have non-MIT licenses (CC BY-NC-SA / research-only / proprietary) and would compromise the project's license-clean posture.
- **No multi-resolution experiment** (testing 64×64 vs 112×112). Downsample to 64×64 for direct apples-to-apples; resolution-effect study is a separate iteration if warranted.

## 11. Success criteria

- DigiFace-1M 118k subset downloads successfully and integrates with `DigiFaceLoader` without breaking changes.
- The preprocessing script produces `datasets/_real/digiface_118k_64/` with 64×64 PNGs preserving the `<identity>/<sample>.png` layout.
- The 8/16 identity-disjoint selections for Set A/B are written to `configs/digiface_identities_set{a,b}.txt` (committed).
- All six `real_*` configs generate datasets cleanly at the same sample counts as the v2.1 sweep (96/512/4096 for Set A; 128/1024/8192 for Set B).
- The pipeline.py modification preserves the existing `test_print_attack` and `test_pipeline_e2e` tests; the golden either passes unchanged OR is regenerated with the ontology_version-fix-only diff (manifest field changes only; output JPEG bytes unchanged).
- 27-cell v2.1-on-real-bonafide sweep completes on the Spark with all JSONs collected.
- Report has a populated synthetic-vs-real v2.1 comparison table and a written verdict per §2.
- `defid-pkg`, `defid-demo-pkg`, `pad-synth-core/src`, `pad-synth-face/src/pad_synth_face/{sensor,cli}.py`, all attack code (`attacks/print.py`, `attacks/replay.py`, `attacks/base.py`) — provably unmodified.

## 12. Operational risk: DigiFace download

DigiFace-1M's distribution may require Azure account login, agreement to terms of use, or specific download tooling. The exact URL is not pinned in this spec. The implementation plan's download task surfaces the actual access mechanism; if it requires interactive auth, the task reports BLOCKED and we either (a) the user fetches it manually and the plan resumes from a pre-staged directory, OR (b) the spec is revised to use a permissively-licensed alternative (e.g., a public-domain face subset). This is the one genuine "could fail at execution" risk; the rest of the work proceeds deterministically.
