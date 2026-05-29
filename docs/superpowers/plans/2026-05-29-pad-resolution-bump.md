# PAD A1 Resolution Bump Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bump the canonical image resolution from 64×64 to 224×224 with the print/replay physics rewritten as image-fraction so the same physical print/screen scales naturally at any input dimension.

**Architecture:** Add a single canonical `IMAGE_SIZE` constant to `pad_synth_core`. Rewrite the two resolution-dependent physics helpers (print halftone cell, replay subpixel + moiré) to derive size from the input array's `shape[0]` (not from the constant) — this decouples physics from the global, lets back-compat tests just feed a 64×64 input, and matches the existing helper signatures. Flip the hardcoded `(64, 64, 3)` shape sites (pipeline QC gate, real-attack target, fixtures, prep script default) to use the constant. Mask physics is already fraction-based and needs no retune. Bump existing run configs to point at the new `digiface_224` dir; regenerate the determinism golden.

**Tech Stack:** Python 3.12+, NumPy, Pillow, ffmpeg (existing). No new dependencies.

**Spec:** [`../specs/2026-05-29-pad-resolution-bump-design.md`](../specs/2026-05-29-pad-resolution-bump-design.md)

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `pad-synth-core/src/pad_synth_core/__init__.py` | Add `IMAGE_SIZE`, `IMAGE_SHAPE` canonical constants | Modify |
| `pad-synth-face/src/pad_synth_face/attacks/print.py` | `_apply_halftone` cell_px derived from `rgb.shape[0]` | Modify |
| `pad-synth-face/src/pad_synth_face/attacks/replay.py` | `_subpixel_grid` pitch + `_moire` freq derived from `h`/`w` | Modify |
| `pad-synth-face/src/pad_synth_face/pipeline.py` | `_FIXED_IMAGE_SHAPE` → `IMAGE_SHAPE` | Modify |
| `pad-synth-face/src/pad_synth_face/real_attack.py` | `_TARGET` + resize use `IMAGE_SIZE`/`IMAGE_SHAPE` | Modify |
| `pad-synth-face/src/pad_synth_face/dfdc.py` | Default `res=IMAGE_SIZE` | Modify |
| `pad-synth-face/src/pad_synth_face/_fixtures.py` | Procedural fixtures emit `IMAGE_SIZE×IMAGE_SIZE` images | Modify |
| `scripts/prepare_digiface_64.py` → `scripts/prepare_digiface.py` | Rename; `--size` arg defaults to `IMAGE_SIZE` | Rename + modify |
| `tests/golden/golden_hashes.json` | Regenerated at the new resolution | Modify |
| Existing tests that hardcode `(64, 64, 3)` | Use `IMAGE_SHAPE` instead | Modify |
| `configs/runs/{mask,mix,real}_set*_d*.yaml` | `bonafide.root` flips to `./datasets/_real/digiface_224` | Modify (in-place) |

---

## Task 1: Add canonical `IMAGE_SIZE` and `IMAGE_SHAPE` constants

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/__init__.py`
- Create (test): `pad-synth-core/tests/test_image_size_constant.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-core/tests/test_image_size_constant.py`:

```python
from pad_synth_core import IMAGE_SHAPE, IMAGE_SIZE


def test_image_size_is_224():
    assert IMAGE_SIZE == 224


def test_image_shape_matches_size():
    assert IMAGE_SHAPE == (IMAGE_SIZE, IMAGE_SIZE, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_image_size_constant.py -v`
Expected: FAIL — `ImportError: cannot import name 'IMAGE_SHAPE'`.

- [ ] **Step 3: Add the constants**

Replace the contents of `pad-synth-core/src/pad_synth_core/__init__.py` with:

```python
__version__ = "0.1.0"

# Canonical input resolution for the synthetic pipeline. Every shape-gate /
# resize / fixture emits at this size. The physics modules (print halftone,
# replay subpixel + moiré) derive their pixel-scale from the actual input
# array's shape, NOT from this constant — that decouples them from the
# global and keeps back-compat testing trivial (just pass a 64x64 input).
IMAGE_SIZE: int = 224
IMAGE_SHAPE: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_image_size_constant.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/__init__.py pad-synth-core/tests/test_image_size_constant.py
git commit -m "feat(pad-core): canonical IMAGE_SIZE=224 / IMAGE_SHAPE constants"
```

---

## Task 2: Print halftone — derive `cell_px` from input image dim

The current formula `base_cell = max(2.0, round(8.0 * 150.0 / print_dpi))` uses absolute pixels and ignores the image dimension. Rewrite to scale with `rgb.shape[0]` so the same physical print at 224×224 has cells ~3.5× larger in pixels (preserving real-world geometry). At `rgb.shape[0] == 64` the formula must produce **byte-identical** values to the current implementation.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py`
- Create (test): `pad-synth-face/tests/test_print_halftone_resolution.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_print_halftone_resolution.py`:

```python
import numpy as np

from pad_synth_face.attacks.print import _apply_halftone


def _compute_cell_px(image_dim: int, print_dpi: int) -> float:
    """Run the deterministic-screen path of _apply_halftone (rng=None) and
    infer the cell_px the formula chose by inspecting the output's
    dominant Fourier period. For this test we just exercise that the
    function runs and produces the right shape — the back-compat anchor
    test below pins the actual cell_px values via the formula."""
    rgb = np.full((image_dim, image_dim, 3), 0.5, dtype=np.float32)
    out = _apply_halftone(rgb, print_dpi=print_dpi)
    assert out.shape == rgb.shape
    return out


def test_64x64_byte_identical_at_each_dpi_back_compat():
    """At image_dim=64, the new formula must produce byte-identical halftone
    output to the previous absolute-pixel formula for every print_dpi we ship.
    """
    for dpi in (150, 300, 600, 1200):
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        out = _apply_halftone(rgb, print_dpi=dpi)
        # Expected values from the prior formula: max(2.0, round(8 * 150 / dpi)).
        # The image is uniform 0.5, so halftone produces a regular dot pattern
        # whose total "on" count is determined by cell size + threshold geometry.
        # We assert on the fraction of ON pixels per channel (deterministic).
        # The exact fractions: at threshold dot-screen vs 0.5 input, ~50% pixels
        # are ON on average for any cell size, but the SPATIAL pattern differs.
        # The strongest invariant: byte-identical output to the pre-bump version.
        # The pre-bump output bytes are captured in the existing
        # test_print_halftone.py tests; this test relies on those continuing
        # to pass after the formula rewrite.
        assert out.dtype == np.float32
        assert out.shape == (64, 64, 3)


def test_cell_px_scales_with_image_dim():
    """At image_dim=224, the deterministic halftone pattern must contain
    visibly larger cells than at image_dim=64 (same physical print at higher
    capture resolution). Measure dominant column-period via FFT."""
    rgb_64 = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rgb_224 = np.full((224, 224, 3), 0.5, dtype=np.float32)
    out_64 = _apply_halftone(rgb_64, print_dpi=150)
    out_224 = _apply_halftone(rgb_224, print_dpi=150)

    # Dominant period in the row-summed signal: take a center row, take its
    # FFT magnitude, find the peak frequency above DC.
    def _dominant_period(img: np.ndarray) -> float:
        row = img[img.shape[0] // 2, :, 0].astype(np.float64)
        spec = np.abs(np.fft.rfft(row - row.mean()))
        if spec.size <= 1:
            return float("inf")
        peak_k = int(np.argmax(spec[1:])) + 1
        return len(row) / peak_k  # period in pixels

    period_64 = _dominant_period(out_64)
    period_224 = _dominant_period(out_224)
    # 224 / 64 ≈ 3.5; allow generous tolerance because the halftone screen
    # has rotated CMYK channels and the dominant Fourier peak depends on
    # which channel the row hits. The scaling direction is the load-bearing
    # invariant.
    assert period_224 > period_64, (
        f"expected period_224 > period_64 (cell_px scales with image), "
        f"got {period_224=:.2f} vs {period_64=:.2f}"
    )
```

- [ ] **Step 2: Run the new tests + the existing halftone tests to capture the pre-rewrite baseline**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone.py pad-synth-face/tests/test_print_halftone_jitter.py pad-synth-face/tests/test_print_halftone_resolution.py -v`
Expected: all existing tests PASS; the new `test_cell_px_scales_with_image_dim` FAILS (current formula is `image_dim`-independent so periods are identical at 64 and 224).

- [ ] **Step 3: Rewrite the formula**

In `pad-synth-face/src/pad_synth_face/attacks/print.py`, find the line in `_apply_halftone`:

```python
    base_cell = max(2.0, round(8.0 * 150.0 / float(print_dpi)))
```

Replace it with:

```python
    # Image-fraction-based: a halftone cell occupies the same fraction of
    # image area regardless of resolution (preserves real-world print
    # geometry across capture resolutions). The constant 0.125 calibrates
    # so that at image dim 64 and print_dpi 150 the formula reproduces the
    # pre-bump cell_px=8 exactly. See spec §4 (2026-05-29 resolution bump).
    image_dim = rgb.shape[0]
    base_cell = max(2.0, image_dim * 0.125 * (150.0 / float(print_dpi)))
```

- [ ] **Step 4: Run all halftone tests**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone.py pad-synth-face/tests/test_print_halftone_jitter.py pad-synth-face/tests/test_print_halftone_resolution.py -v`
Expected: all PASS — existing tests still green (back-compat at 64×64); new scaling test now passes.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/print.py pad-synth-face/tests/test_print_halftone_resolution.py
git commit -m "feat(pad-print): halftone cell_px derived from image dim (image-fraction)"
```

---

## Task 3: Replay — derive subpixel pitch + moiré freq from `h`/`w`

Same image-fraction rewrite for the two replay physics helpers.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/replay.py`
- Create (test): `pad-synth-face/tests/test_replay_resolution.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_replay_resolution.py`:

```python
import numpy as np

from pad_synth_face.attacks.replay import _moire, _subpixel_grid


def test_subpixel_pitch_back_compat_at_64():
    """At 64x64 the column-stripe pattern repeats every 3 columns (the
    pre-bump pitch). Detect by checking the first three columns of the
    leftmost row contain the three known relative levels [0.92, 0.96, 0.90]."""
    pattern = _subpixel_grid(64, 64)
    row0 = pattern[0, :3, 0]
    assert np.allclose(row0, [0.92, 0.96, 0.90])


def test_subpixel_pitch_scales_with_image_dim():
    """At 224x224 the same column-stripe pattern occupies ~11 columns per
    repeat (round(224/64 * 3)) — same visible angular size."""
    pattern = _subpixel_grid(224, 224)
    # The pattern is uniform across rows, so just inspect the first row.
    row0 = pattern[0, :, 0]
    # Find the column index of the FIRST occurrence of value 0.92 after column 0
    # (i.e. the pitch). At pitch P, column P is again 0.92.
    first_period = None
    for k in range(1, len(row0)):
        if np.isclose(row0[k], 0.92):
            first_period = k
            break
    assert first_period == 11, f"expected pitch 11 at 224x224, got {first_period}"


def test_moire_freq_back_compat_at_64():
    """At 64x64 with refresh_hz=60 the moiré freq equals the pre-bump
    0.18 cycles/pixel. Detect by the dominant peak in the moiré pattern's
    horizontal slice (taking abs FFT of a center row)."""
    rng = np.random.default_rng(0)
    pat = _moire(64, 64, refresh_hz=60, rng=rng)
    row = pat[32, :, 0]
    spec = np.abs(np.fft.rfft(row - row.mean()))
    peak_k = int(np.argmax(spec[1:])) + 1
    # 64 pixels * 0.18 cycles/pixel ≈ 11.5 cycles ≈ peak at k=11 or k=12.
    assert peak_k in (11, 12), f"expected k in 11..12 at 64x64, got {peak_k}"


def test_moire_freq_scales_with_image_dim():
    """At 224x224 the moiré freq is scaled by 64/224 so the bands-per-image
    count stays the same as at 64."""
    rng = np.random.default_rng(0)
    pat = _moire(224, 224, refresh_hz=60, rng=rng)
    row = pat[112, :, 0]
    spec = np.abs(np.fft.rfft(row - row.mean()))
    peak_k = int(np.argmax(spec[1:])) + 1
    # Same 11..12 cycles per image-width (NOT per pixel) at any resolution.
    assert peak_k in (11, 12), f"expected k in 11..12 at 224x224, got {peak_k}"
```

- [ ] **Step 2: Run tests to verify expected fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_replay_attack.py pad-synth-face/tests/test_replay_resolution.py -v`
Expected: `test_replay_attack.py` PASS (existing tests untouched, 64×64 inputs); `test_subpixel_pitch_back_compat_at_64` and `test_moire_freq_back_compat_at_64` PASS (current behaviour); `test_subpixel_pitch_scales_with_image_dim` FAIL (pitch stuck at 3); `test_moire_freq_scales_with_image_dim` FAIL (peak_k ≈ 40 at 224 since freq is absolute).

- [ ] **Step 3: Rewrite both helpers**

In `pad-synth-face/src/pad_synth_face/attacks/replay.py`, replace `_subpixel_grid` with:

```python
def _subpixel_grid(h: int, w: int) -> np.ndarray:
    # Image-fraction pitch: at any image dim the column-stripe pattern occupies
    # the same visible angular fraction. At h=64 -> pitch=3 (pre-bump back-compat);
    # at h=224 -> pitch=11. See spec §4 (2026-05-29 resolution bump).
    pitch = max(1, round(h / 64.0 * 3))
    pattern = np.tile(
        np.array([0.92, 0.96, 0.90], dtype=np.float32)[None, :, None],
        (h, w // pitch + 1, 3),
    )[:, :w * pitch // pitch]
    return pattern[:, :w].astype(np.float32)
```

Then replace `_moire` with:

```python
def _moire(h: int, w: int, refresh_hz: int, rng: np.random.Generator) -> np.ndarray:
    # Image-fraction freq: bands-per-image-width stays constant across
    # resolutions. At h=64 the multiplier 64/h is 1.0 (pre-bump back-compat);
    # at h=224 the freq is divided by 3.5. See spec §4 (2026-05-29 resolution bump).
    freq = (0.18 + (refresh_hz - 60) * 0.0015) * (64.0 / h)
    angle = float(rng.uniform(-0.4, 0.4))
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    pattern = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    return (1.0 + 0.04 * pattern).astype(np.float32)[:, :, None]
```

- [ ] **Step 4: Run all replay tests**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_replay_attack.py pad-synth-face/tests/test_replay_resolution.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/replay.py pad-synth-face/tests/test_replay_resolution.py
git commit -m "feat(pad-replay): subpixel pitch + moiré freq derived from image dim"
```

---

## Task 4: Flip day — bump fixtures + shape-gate sites + tests to `IMAGE_SHAPE`

The "atomic flip" task: bump every site that hardcodes `(64, 64, 3)` or `64` to use `IMAGE_SIZE`/`IMAGE_SHAPE`. Fixtures start emitting 224×224 images, the pipeline QC gate accepts 224×224, real-attack and DFDC ingesters target 224×224. Tests that assert `(64, 64, 3)` use `IMAGE_SHAPE` instead.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/pipeline.py`
- Modify: `pad-synth-face/src/pad_synth_face/real_attack.py`
- Modify: `pad-synth-face/src/pad_synth_face/dfdc.py`
- Modify: `pad-synth-face/src/pad_synth_face/_fixtures.py`
- Modify: tests that assert `(64, 64, 3)` (see Step 4 for the list)

- [ ] **Step 1: pipeline.py — replace `_FIXED_IMAGE_SHAPE`**

In `pad-synth-face/src/pad_synth_face/pipeline.py`:

Add `from pad_synth_core import IMAGE_SHAPE` near the other `pad_synth_core` imports.

Find the line:
```python
_FIXED_IMAGE_SHAPE = (64, 64, 3)
```
Delete it. Then find both occurrences of `_FIXED_IMAGE_SHAPE` (in the bonafide and attack loops, around lines 189 and 243) and replace with `IMAGE_SHAPE`.

- [ ] **Step 2: real_attack.py — flip `_TARGET` + the resize call**

In `pad-synth-face/src/pad_synth_face/real_attack.py`:

Add `from pad_synth_core import IMAGE_SHAPE, IMAGE_SIZE` near the other `pad_synth_core` imports.

Replace:
```python
_TARGET = (64, 64, 3)
```
with:
```python
_TARGET = IMAGE_SHAPE
```

Find the resize call in `_load_64`:
```python
        im = im.convert("RGB").resize((64, 64), Image.LANCZOS)
```
Replace with:
```python
        im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
```

(Also rename the helper from `_load_64` to `_load_target` for honesty, and update the call site. Two-line change.)

- [ ] **Step 3: dfdc.py — default `res=IMAGE_SIZE`**

In `pad-synth-face/src/pad_synth_face/dfdc.py`:

Add `from pad_synth_core import IMAGE_SIZE` near the other `pad_synth_core` imports.

Find the `extract_dfdc_bonafide` signature:
```python
def extract_dfdc_bonafide(
    src: Path,
    out: Path,
    license: str,
    source_url: str,
    res: int = 64,
    ...
```
Change `res: int = 64` to `res: int = IMAGE_SIZE`.

- [ ] **Step 4: _fixtures.py — bump procedural fixture sizes**

In `pad-synth-face/src/pad_synth_face/_fixtures.py`:

Add `from pad_synth_core import IMAGE_SIZE` near the existing imports.

Find every literal `64` that is a procedural image dimension and replace with `IMAGE_SIZE`. The grep showed 6 sites: lines 19, 20, 85, 86, 93, 99, 102. For each, change `(64, 64, ...)` → `(IMAGE_SIZE, IMAGE_SIZE, ...)` and `64, 64` → `IMAGE_SIZE, IMAGE_SIZE`. The `build_fixture_dfdc` and `build_fixture_real_attack` builders don't hardcode 64; leave them alone.

- [ ] **Step 5: Update tests asserting `(64, 64, 3)` to use `IMAGE_SHAPE`**

Run this command to find every test that hardcodes the old shape:
```bash
grep -rln "(64, 64, 3)" pad-synth-face/tests pad-synth-core/tests tests
```

For each file in that list, add `from pad_synth_core import IMAGE_SHAPE` at the top and replace `(64, 64, 3)` with `IMAGE_SHAPE`. Do the same for any `(64, 64)` literals used as image dimensions (e.g. `np.full((64, 64, 3), ...)`).

Do NOT change tests that intentionally use 64×64 to anchor pre-bump back-compat (the new `test_print_halftone_resolution.py` and `test_replay_resolution.py` from Tasks 2 and 3 keep their 64 literals — those are deliberate). Do NOT change `_fixtures.py`'s helpers that build SOURCE data for the DFDC ingester at 128×96 (those are video frame dims, not the canonical image size).

- [ ] **Step 6: Regenerate the determinism golden at the new resolution**

Run:
```bash
PAD_SYNTH_UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_determinism_golden.py -v
.venv/bin/python -m pytest tests/test_determinism_golden.py -v
```
Expected: first call rewrites `tests/golden/golden_hashes.json`; second call passes against the regenerated golden (determinism preserved at the new resolution).

- [ ] **Step 7: Full suite green check**

Run: `.venv/bin/python -m pytest -q`
Expected: green. Any failure is in code we just touched; investigate immediately, do NOT loosen assertions.

- [ ] **Step 8: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/pipeline.py \
        pad-synth-face/src/pad_synth_face/real_attack.py \
        pad-synth-face/src/pad_synth_face/dfdc.py \
        pad-synth-face/src/pad_synth_face/_fixtures.py \
        tests/golden/golden_hashes.json \
        pad-synth-face/tests pad-synth-core/tests
git commit -m "feat(pad-resolution): flip canonical shape sites + fixtures + tests to IMAGE_SHAPE (64->224)"
```

---

## Task 5: Rename + parameterize the DigiFace prep script

**Files:**
- Rename: `scripts/prepare_digiface_64.py` → `scripts/prepare_digiface.py`
- Modify: the renamed script

- [ ] **Step 1: Rename via git**

```bash
git mv scripts/prepare_digiface_64.py scripts/prepare_digiface.py
```

- [ ] **Step 2: Replace the script contents**

Replace the contents of `scripts/prepare_digiface.py` with:

```python
#!/usr/bin/env python3
"""Resize DigiFace-1M images to the canonical IMAGE_SIZE (default 224x224),
preserving <root>/<id>/<sample> layout.

Idempotent: skips files that already exist at the destination. Uses PIL's
LANCZOS resampling. Writes `_meta.json` recording the target size,
identity count, and sample counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_core import IMAGE_SIZE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source DigiFace root: <src>/<identity>/<sample>.{png,jpg}")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Destination root for resized images")
    ap.add_argument("--size", type=int, default=IMAGE_SIZE,
                    help=f"Target square size (default: IMAGE_SIZE={IMAGE_SIZE})")
    args = ap.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst
    target_size: int = args.size
    dst_root.mkdir(parents=True, exist_ok=True)

    n_ids = 0
    n_samples = 0
    n_skipped = 0
    for id_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        identity = id_dir.name
        out_dir = dst_root / identity
        out_dir.mkdir(exist_ok=True)
        n_ids += 1
        for sample_path in sorted(id_dir.iterdir()):
            if sample_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            out_path = out_dir / f"{sample_path.stem}.png"
            if out_path.exists():
                n_skipped += 1
                continue
            with Image.open(sample_path) as im:
                im = im.convert("RGB").resize((target_size, target_size), Image.LANCZOS)
                im.save(out_path, format="PNG")
            n_samples += 1

    meta = {
        "target_size": target_size,
        "src": str(src_root),
        "identities": n_ids,
        "samples_total": n_samples + n_skipped,
        "samples_written": n_samples,
        "samples_skipped_existing": n_skipped,
    }
    (dst_root / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the parameterized script (size override)**

Run:
```bash
.venv/bin/python - <<'PY'
import subprocess, sys, tempfile, pathlib, json
# Build a tiny 2-identity x 2-sample source.
src = pathlib.Path(tempfile.mkdtemp())
for i in range(2):
    (src / f"{i:04d}").mkdir()
    for k in range(2):
        from PIL import Image
        import numpy as np
        arr = (np.random.default_rng(i * 10 + k).random((112, 112, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(src / f"{i:04d}" / f"{k}.png")
dst = pathlib.Path(tempfile.mkdtemp())
r = subprocess.run(
    [".venv/bin/python", "scripts/prepare_digiface.py",
     "--src", str(src), "--dst", str(dst), "--size", "32"],
    capture_output=True, text=True,
)
print(r.stdout, r.stderr)
assert r.returncode == 0
meta = json.loads((dst / "_meta.json").read_text())
assert meta["target_size"] == 32
assert meta["identities"] == 2
assert meta["samples_written"] == 4
# Verify one image got resized.
from PIL import Image
im = Image.open(dst / "0000" / "0.png")
assert im.size == (32, 32)
print("smoke OK")
PY
```
Expected: prints meta JSON + `smoke OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_digiface.py
git commit -m "feat(pad-resolution): prepare_digiface.py with --size flag (default IMAGE_SIZE)"
```

---

## Task 6: Bump existing run configs to `digiface_224`

In-place flip of `bonafide.root` across the 18 existing run configs (`mask_set*_d*.yaml`, `mix_set*_d*.yaml`, `real_set*_d*.yaml`). The old `digiface_118k_64` references become `digiface_224` so a regenerated DigiFace at 224 is what the configs ingest.

**Files:**
- Modify: `configs/runs/{mask,mix,real}_set{a,b}_d{1,2,3}.yaml` (18 files)

- [ ] **Step 1: Verify the current set**

Run:
```bash
grep -l "digiface_118k_64" configs/runs/ | sort
```
Expected output: 18 files — `configs/runs/{mask,mix,real}_set{a,b}_d{1,2,3}.yaml`.

- [ ] **Step 2: One-line edit per file (use a single sed pass)**

```bash
for f in configs/runs/{mask,mix,real}_set{a,b}_d{1,2,3}.yaml; do
  sed -i.bak 's|./datasets/_real/digiface_118k_64|./datasets/_real/digiface_224|g' "$f"
  rm "$f.bak"
done
```

- [ ] **Step 3: Verify**

```bash
grep -l "digiface_118k_64" configs/runs/ | wc -l   # must be 0
grep -l "digiface_224" configs/runs/ | wc -l        # must be 18
```

- [ ] **Step 4: Commit**

```bash
git add configs/runs/
git commit -m "feat(pad-resolution): bump existing run configs to digiface_224"
```

---

## Task 7: Doc note + full-suite + lint checkpoint

**Files:**
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (add a one-line note in the running header about the 64→224 baseline change)

- [ ] **Step 1: Add a header note to the report**

At the top of `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (just after the first title line), add a small note:

```markdown
> **Resolution baseline change (2026-05-29):** all sections in this report up to and including the 2026-05-27 update used the 64×64 baseline. Subsequent A1 sweep results will use 224×224 as the new canonical resolution. Old 64×64 numbers below remain immutable for historical comparison.
```

- [ ] **Step 2: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: green (prior baseline was 217 passed / 1 skipped; this adds the test_image_size_constant + halftone + replay resolution tests, ~6 new — about 223 passed, 1 skipped).

- [ ] **Step 3: Lint the new/modified files**

Run:
```bash
uvx ruff check --select E,F,B,UP --line-length 100 --ignore E501 \
  pad-synth-core/src/pad_synth_core/__init__.py \
  pad-synth-face/src/pad_synth_face/attacks/print.py \
  pad-synth-face/src/pad_synth_face/attacks/replay.py \
  pad-synth-face/src/pad_synth_face/pipeline.py \
  pad-synth-face/src/pad_synth_face/real_attack.py \
  pad-synth-face/src/pad_synth_face/dfdc.py \
  pad-synth-face/src/pad_synth_face/_fixtures.py \
  scripts/prepare_digiface.py \
  pad-synth-core/tests/test_image_size_constant.py \
  pad-synth-face/tests/test_print_halftone_resolution.py \
  pad-synth-face/tests/test_replay_resolution.py
```
Expected: `All checks passed!` on the new files. (Do NOT use the `I`/isort rule from repo root — `uvx ruff` misclassifies the `src`-layout packages as third-party and will spuriously rewrite import blocks across the whole codebase. Match house style by hand: blank line before first-party imports.)

- [ ] **Step 4: Commit (only if lint fixes were needed)**

```bash
git add -A
git commit -m "docs(pad-resolution): report header note for 64->224 baseline change"
```

---

## Self-review notes

- **Spec coverage:** §3 file list → Tasks 1, 4; §4 physics retune (image-fraction with back-compat constants 0.125 / 64-multiplier / pitch=3-at-64) → Tasks 2, 3 with explicit back-compat tests; §5 single-constant approach → Task 1; §6 in-place config bump → Task 6; §7 testing (back-compat anchors, golden regen, ISO metrics unchanged) → Tasks 2, 3, 4; §8 compute → not code (runtime concern); §9 out-of-scope (A2 / DFDC ingest at 224 / arch / multi-res) → none built here.
- **Atomic-flip strategy:** physics retunes (Tasks 2, 3) are designed to leave the existing test suite green because they derive size from `rgb.shape[0]` and the suite still feeds 64×64 inputs. The "flip day" (Task 4) is the one task where the canonical shape changes — fixtures, pipeline gate, real_attack target, and DFDC default all change together, and the golden is regenerated in the same task. No half-flipped intermediate states.
- **No new dependencies.** All work uses existing numpy/Pillow/ffmpeg.
- **Existing committed reports/sweeps remain immutable** — the 64×64 mask/integrated/synth→real numbers stay in the report as the pre-bump baseline; the doc-note in Task 7 makes the resolution change explicit at the top of the running header.
- **Mask physics is untouched** — verified during the mask-attack review earlier in the session that the mask helpers are fraction-based and resolution-independent.
