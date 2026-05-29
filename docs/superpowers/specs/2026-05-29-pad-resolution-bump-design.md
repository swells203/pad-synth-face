% PAD A1 — Resolution Bump 64×64 → 224×224 Design
% Preserve high-frequency PAD cues (halftone screen geometry, moiré bands, skin micro-texture) the existing 64×64 baseline destroys at the resize. Single canonical `IMAGE_SIZE` constant; physics retuned as image-fraction so the new resolution preserves real-world geometry.
% 2026-05-29

---

## 1. Purpose and audience

The 2026-05-27 synth→real pilot showed cross-domain EER ≈ chance on real attacks despite synth-cross-domain reaching ~0.09–0.25. The diagnosis pointed at two compounding shortfalls: capture-domain realism (the synth pipeline's `sensor.py` is minimal) and **resolution** — at 64×64 the high-frequency cues that distinguish PAD attacks from genuine captures (halftone screen geometry, moiré bands, skin micro-texture, paper grain) are largely destroyed before the model sees them.

This spec handles the resolution lever (A1). The capture-realism expansion (A2) is its own follow-up spec on top of this new resolution baseline.

The bump is **64×64 → 224×224**, chosen for SOTA-standard input size and maximum upside on capture-cue preservation. The work is mostly mechanical (single canonical constant + ~14 hardcoded-shape replacements) plus genuine physics retune for the two resolution-dependent attack modules (print, replay). The mask module is already fraction-based and needs no retune.

Audience: future maintainers; the next sub-projects on the queue (A2 sensor realism, DFDC ingest at the new resolution, B1 synth-pretrain+real-finetune curve, Tier-B benchmark).

## 2. The question this answers

| Behaviour on the next sweep at 224×224 | What it tells us |
|---|---|
| Synth→synth cross-domain EER similar to 64×64 baseline (~0.09 mask-only, ~0.094 integrated) | Resolution alone doesn't change the within-synthetic story. Expected. |
| **Synth→real EER closes meaningfully** (vs the ≈chance 0.55–0.68 pilot) | Resolution was a major piece of the gap. Confirms A1 was the right next lever. |
| Synth→synth cross-domain EER **collapses to 0.000** in some cells | A new generator fingerprint emerged at higher resolution (e.g. the retuned halftone became too uniform). Same artifact-discipline playbook as v2/v2.1. |
| Synth→real unchanged | Resolution wasn't the bottleneck; the gap is dominantly capture-realism. A2 then becomes the higher-priority lever and we measure incrementally. |

Decision rule (mirrors prior sweeps): `no cross-domain cell mean ≤ 0.001` ⇒ artifact-free.

## 3. Architecture and files

Single canonical resolution constant in `pad-synth-core`. All resolution-touching call sites import from there. Physics modules that use absolute-pixel-count formulas get rewritten as image-fraction-based so they preserve real-world geometry across resolutions.

| Change | File | Note |
|---|---|---|
| **New canonical constants** | `pad-synth-core/src/pad_synth_core/__init__.py` | Add `IMAGE_SIZE: int = 224` and `IMAGE_SHAPE: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3)` |
| Pipeline shape gate | `pad-synth-face/src/pad_synth_face/pipeline.py` | Replace `_FIXED_IMAGE_SHAPE = (64, 64, 3)` with `from pad_synth_core import IMAGE_SHAPE` |
| Real-attack QC + resize | `pad-synth-face/src/pad_synth_face/real_attack.py` | `_TARGET` and the `.resize((64, 64), ...)` use `IMAGE_SIZE`/`IMAGE_SHAPE` |
| DFDC ingester default | `pad-synth-face/src/pad_synth_face/dfdc.py` | `extract_dfdc_bonafide(..., res=IMAGE_SIZE)` |
| Procedural fixtures | `pad-synth-face/src/pad_synth_face/_fixtures.py` | `build_fixture_bonafide` and friends emit `IMAGE_SIZE × IMAGE_SIZE` images |
| Print physics retune | `pad-synth-face/src/pad_synth_face/attacks/print.py` | `_apply_halftone` rewritten: cell_px is a fraction of `IMAGE_SIZE` keyed by `print_dpi` |
| Replay physics retune | `pad-synth-face/src/pad_synth_face/attacks/replay.py` | `_subpixel_grid` tile factor and `_moire` freq rewritten as image-fraction |
| DigiFace prep script | rename `scripts/prepare_digiface_64.py` → `scripts/prepare_digiface.py` | `--size` arg (default 224); preserves `--size 64` back-compat |
| Determinism golden | `tests/test_determinism_golden.py` + `tests/golden/golden_hashes.json` | Regenerated at 224 (all hashes change — expected |
| Existing run configs | `configs/runs/{mask,mix,real}_set*_d*.yaml` | One-line edit per file: `bonafide.root` flips to `./datasets/_real/digiface_224` |

## 4. Physics retune (image-fraction-based)

### Print halftone (`pad-synth-face/src/pad_synth_face/attacks/print.py`)

Today's formula (`_apply_halftone`, line ~158):
```python
base_cell = max(2.0, round(8.0 * 150.0 / float(print_dpi)))
```
This is in absolute pixels and ignores the image dimension — at 224 the same 8-pixel cell at 150 dpi occupies 1/28 of the image instead of 1/8. New formula, keyed off `IMAGE_SIZE`:
```python
base_cell = max(2.0, IMAGE_SIZE * 0.125 * (150.0 / float(print_dpi)))
```
The constant `0.125` is calibrated so that at `IMAGE_SIZE=64` and `print_dpi=150` the formula reproduces the prior `8` exactly (back-compat anchor). At `IMAGE_SIZE=224` and the same dpi, `base_cell ≈ 28` — preserving the same physical print-cell-to-image-area ratio.

### Replay subpixel grid (`attacks/replay.py`)

Today's `_subpixel_grid` tiles `[0.92, 0.96, 0.90]` per 3 columns absolute (line ~30). At 224 that's three 1-pixel columns occupying 3/224 of width, vs 3/64 = ~5% at the old baseline. Rewrite the tile so the stripe pattern occupies the same image-width fraction:
```python
PITCH = max(1, round(IMAGE_SIZE / 64 * 3))  # 3 at 64x64, 11 at 224x224
pattern = np.tile(np.array([0.92, 0.96, 0.90], dtype=np.float32)[None, :, None],
                  (h, w // PITCH + 1, 3))[:, :w]
```
At `IMAGE_SIZE=64` this reproduces `PITCH=3` (back-compat). At 224 the stripes are 11px wide — the visible spatial frequency stays constant.

### Replay moiré (`attacks/replay.py`)

Today's `_moire` (line ~38):
```python
freq = 0.18 + (refresh_hz - 60) * 0.0015
```
Absolute cycles per pixel — at 224 there are 3.5× more bands across the image. Rewrite so the *bands-per-image-width* count stays the same as at 64:
```python
freq = (0.18 + (refresh_hz - 60) * 0.0015) * (64.0 / IMAGE_SIZE)
```
At `IMAGE_SIZE=64` the multiplier is 1.0 (back-compat). At 224 it's ~0.286 — same band count visible on the captured image.

### Mask physics — no retune needed

`_dome_normals`, `_seam`, `_aperture_mismatch`, `_drape_warp`, and `_specular` all already use image-fraction coordinates (verified during the mask-attack-module review). They produce the correct visual at any resolution. ✓

## 5. Single canonical `IMAGE_SIZE` constant

In `pad-synth-core/src/pad_synth_core/__init__.py`:
```python
__version__ = "0.1.0"
IMAGE_SIZE: int = 224
IMAGE_SHAPE: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3)
```
Every resolution-touching site imports from here:
```python
from pad_synth_core import IMAGE_SIZE, IMAGE_SHAPE
```
No per-call parameter; one bump for the whole project. Future resolution changes are a one-line edit.

(The DFDC ingester's `res` parameter and the digiface prep script's `--size` flag are kept parameterized — they accept arbitrary sizes for re-prep at different resolutions, defaulting to `IMAGE_SIZE`. This is the only intentional parameterization beyond the constant.)

## 6. Datasets and configs

**In-place bump.** Existing committed run configs (`mask_set{a,b}_d{1,2,3}.yaml`, `mix_*`, `real_*`) keep their structure; only `bonafide.root` flips to `./datasets/_real/digiface_224`. Pipeline regenerates at 224 from the new DigiFace prep.

Old 64×64 datasets remain on disk (gitignored under `datasets/`); the user may delete them or keep them for parallel A/B comparison. Old report numbers (mask-only L3·D3 ≈ 0.089, integrated L2·D3 ≈ 0.094) remain immutable in the committed sweep-results report as the 64×64 baseline.

DFDC configs (when the user lands DFDC) point at `./datasets/_real/dfdc_224` — `prepare_dfdc.py --res 224` produces it.

## 7. Testing

- **Physics back-compat anchors:** the print halftone and replay subpixel/moiré formula tests gain a "behaviour at `IMAGE_SIZE=64`" parametrisation that asserts the rewritten formulas produce the **byte-identical** pre-bump values (cell_px=8, PITCH=3, freq multiplier=1.0). Catches accidental drift in the relative-to-image rewrites.
- **All existing pipeline e2e tests** continue to pass: they assert generator output shape, which now becomes `IMAGE_SHAPE` instead of the old hardcoded `(64,64,3)`. The fixtures auto-bump because they use `IMAGE_SIZE`.
- **Determinism golden:** regenerated (`PAD_SYNTH_UPDATE_GOLDEN=1 pytest tests/test_determinism_golden.py`); the new golden hashes are committed.
- **Mask anti-palette test** at 224: continues to assert > 1000 distinct colours (now from a 224² = 50,176-pixel sample, so the threshold remains well within the trivially-achievable range).
- **ISO metrics (just shipped)** measure the resolution effect on synth→real once a sweep runs — no test changes needed for the eval layer.

## 8. Compute and rollout

- **Dataset regen:** ~4–5× per-dataset CPU vs 64 (more pixels per resize + larger JPEGs). All 12 mask+mix datasets at 224 ≈ 10–15 min local.
- **Spark sweep:** training-time scales with pixel count for the conv front-end; D3 cells ~14s today → ~170s at 224 → 27 cells ≈ 1 hour on GB10. Acceptable.
- **Disk:** D3 mask/mix datasets ~50 KB/JPEG × 4096 samples × 12 datasets ≈ 2–3 GB total. Gitignored.

The sweep result becomes the new "2026-05-XX update — A1 (resolution 64→224)" section in `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`, alongside the existing 64×64 sections for direct comparison.

## 9. Out of scope

- **A2 capture-domain randomisation** (sensor.py expansion: ISP shot/read noise, JPEG recompression, motion blur, lens distortion, replay recapture chain). The next sub-project, built on this 224 baseline.
- **DFDC ingest at 224** — the DFDC harness already accepts `--res`, so this is documented in `docs/dfdc-bonafide.md`, not work here.
- **Architecture changes** (e.g. swapping TinyCNN for a pretrained backbone). The conv front-end uses `AdaptiveAvgPool2d` so it's already resolution-agnostic; this cycle only bumps the resolution, not the model.
- **Multi-resolution experiments** (e.g. parallel 64 + 224 sweeps). Old 64 datasets stay on disk if the user wants A/B; configs are bumped in-place.
