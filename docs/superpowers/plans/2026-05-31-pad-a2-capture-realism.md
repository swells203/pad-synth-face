# PAD A2 — Capture-Domain Realism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `pad_synth_face.sensor` into a physically-ordered capture pipeline that adds four new rng-jittered effects (radial lens distortion, motion blur, shot+read noise, multi-pass JPEG) on top of the existing vignette + WB chain, then measure cross-domain EER delta vs the 2026-05-30 L4 baseline.

**Architecture:** Each new effect is a pure helper `f(img, params..., rng) -> ndarray`. `apply_sensor` draws per-sample parameters, applies effects in physical pipeline order (lens → motion → noise → vignette → WB → JPEG-chain), and returns `(uint8 img, params dict)`. The `SensorPreset` dataclass gains three new range fields; `MOBILE_FRONT_2024` and `WEBCAM_1080P` ship with calibrated defaults per spec §4. No call-site changes — `apply_sensor(img, preset, rng) -> (img, params)` signature unchanged.

**Tech Stack:** Python 3.11, NumPy, OpenCV (`cv2.remap`, `cv2.filter2D`), Pillow (existing JPEG roundtrip), pytest, the existing pad-synth-face / pad-synth-core / scripts/spark_sweep.py infra. Spark GB10 (CUDA) for the sweep. No new dependencies — `cv2` is already a transitive dep used by `attacks/print.py` and `attacks/replay.py`.

**Spec:** `docs/superpowers/specs/2026-05-31-pad-a2-capture-realism-design.md`

**Branch:** `feat/pad-a2-capture-realism` (already created from main; spec committed as `97f5f5b`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `pad-synth-face/src/pad_synth_face/sensor.py` | The capture pipeline | **Modify** (extend dataclass, add 3 helpers, replace `_noise` body, replace `_jpeg_roundtrip` use with `_jpeg_chain`, reorder `apply_sensor`, extend `sensor_params` dict) |
| `pad-synth-face/tests/test_sensor_a2.py` | Per-effect + integration tests for the new code | **Create** |
| `pad-synth-face/tests/test_sensor.py` | Existing sensor contract tests | **Leave unchanged unless a test asserts pre-A2-specific behaviour** (Task 1 audits this) |
| `tests/golden/golden_hashes.json` | Determinism reference hashes | **Regenerate** (sensor output bytes change) |
| `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` | Running sweep results | **Append** "2026-05-31 update — A2 capture-realism" section |
| `runs_mask_224_L4_A2/`, `runs_mix_224_L4_A2/` (under reports dir) | Per-cell JSON + summary.csv | **Create** via sweep |

No other files change. `configs/runs/*.yaml`, `attacks/*.py`, `eval/baseline.py`, `eval/metrics.py`, `eval/models_zoo.py`, `scripts/spark_sweep.py` are all untouched.

---

## Task 1: Audit existing sensor tests + extend `SensorPreset` and presets

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py:13-37`
- Test: `pad-synth-face/tests/test_sensor_a2.py` (new)
- Audit: `pad-synth-face/tests/test_sensor.py` (read; only modify if a test will break)

**Context:** Adding three new fields to the `SensorPreset` frozen dataclass and defaulting them in both existing preset instances. Because the dataclass is frozen, existing `SensorPreset(name=..., iso_range=..., ...)` constructor calls will fail at import time if any field is added without a value. We must add the three new fields to both presets in the same edit. After this task, the dataclass + presets are A2-ready but the helpers and pipeline still behave pre-A2.

The existing `test_sensor.py` exercises `apply_sensor`'s outward contract (shape, dtype, determinism, vignette-darkens-corners, preset differences). Those assertions are agnostic to the internal pipeline ordering, so they should keep passing. Skim the file once before editing.

- [ ] **Step 1: Audit existing sensor tests**

```bash
grep -n "iso_range\|jpeg_qf_range\|wb_k_range\|vignette_strength\|SensorPreset(" pad-synth-face/tests/test_sensor.py
```

Expected: matches only on preset-field comparisons (e.g., the `WEBCAM_1080P.iso_range[1] > MOBILE_FRONT_2024.iso_range[1]` style). No test should construct a `SensorPreset` directly. If a test does, note it — it'll need the three new fields added.

- [ ] **Step 2: Write the failing test for the new preset fields**

Create `pad-synth-face/tests/test_sensor_a2.py`:

```python
"""A2 capture-realism tests: extended preset fields + per-effect helpers."""

from __future__ import annotations

import numpy as np

from pad_synth_core.rng import sample_rng
from pad_synth_face.sensor import MOBILE_FRONT_2024, WEBCAM_1080P, apply_sensor


def test_mobile_preset_has_a2_fields():
    p = MOBILE_FRONT_2024
    assert p.lens_k1_range == (-0.10, 0.10)
    assert p.motion_blur_px_range == (1, 7)
    assert p.jpeg_passes_range == (1, 3)


def test_webcam_preset_has_a2_fields():
    p = WEBCAM_1080P
    assert p.lens_k1_range == (-0.05, 0.05)
    assert p.motion_blur_px_range == (1, 4)
    assert p.jpeg_passes_range == (1, 2)


def test_preset_a2_ranges_are_valid_intervals():
    for p in (MOBILE_FRONT_2024, WEBCAM_1080P):
        assert p.lens_k1_range[0] <= 0.0 <= p.lens_k1_range[1]
        assert p.motion_blur_px_range[0] >= 1
        assert p.motion_blur_px_range[0] <= p.motion_blur_px_range[1]
        assert p.jpeg_passes_range[0] >= 1
        assert p.jpeg_passes_range[0] <= p.jpeg_passes_range[1]
```

- [ ] **Step 3: Run the test, verify it fails**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v
```

Expected: FAIL with `AttributeError: 'SensorPreset' object has no attribute 'lens_k1_range'` (or similar).

- [ ] **Step 4: Extend `SensorPreset` + both presets**

Edit `pad-synth-face/src/pad_synth_face/sensor.py`, replacing the dataclass and both preset constants:

```python
@dataclass(frozen=True)
class SensorPreset:
    name: str
    iso_range: tuple[int, int]
    jpeg_qf_range: tuple[int, int]
    wb_k_range: tuple[int, int]
    vignette_strength: float
    # A2 capture-realism fields (2026-05-31)
    lens_k1_range: tuple[float, float]
    motion_blur_px_range: tuple[int, int]
    jpeg_passes_range: tuple[int, int]


MOBILE_FRONT_2024 = SensorPreset(
    name="mobile-front-2024",
    iso_range=(100, 800),
    jpeg_qf_range=(75, 95),
    wb_k_range=(4200, 6500),
    vignette_strength=0.35,
    lens_k1_range=(-0.10, 0.10),
    motion_blur_px_range=(1, 7),
    jpeg_passes_range=(1, 3),
)

WEBCAM_1080P = SensorPreset(
    name="webcam-1080p",
    iso_range=(200, 1600),
    jpeg_qf_range=(70, 92),
    wb_k_range=(3200, 6000),
    vignette_strength=0.20,
    lens_k1_range=(-0.05, 0.05),
    motion_blur_px_range=(1, 4),
    jpeg_passes_range=(1, 2),
)
```

- [ ] **Step 5: Run the new tests + full sensor suite, verify all pass**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py pad-synth-face/tests/test_sensor.py -v
```

Expected: 3 new tests PASS; all existing `test_sensor.py` tests still PASS (the pipeline body hasn't changed yet).

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor_a2.py
git commit -m "feat(pad-a2): extend SensorPreset with lens/motion/jpeg-chain ranges

Adds three new range fields (lens_k1_range, motion_blur_px_range,
jpeg_passes_range) to SensorPreset and defaults them in both
MOBILE_FRONT_2024 and WEBCAM_1080P per spec §4. Pipeline body unchanged
in this commit — helpers land in subsequent commits."
```

---

## Task 2: Implement `_lens_distort` (radial k1-only)

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py` (add helper after `_white_balance`)
- Test: `pad-synth-face/tests/test_sensor_a2.py` (append tests)

**Context:** Single-parameter radial distortion via Brown-Conrady `k1`. For each pixel, compute its normalised radius `r` from the image centre, displace it by `r' = r * (1 + k1*r²)`, and resample via `cv2.remap`. `k1 = 0` is the identity transform (output ≈ input modulo interpolation noise). The helper is pure: `_lens_distort(img: np.ndarray, k1: float) -> np.ndarray`. No rng inside the helper — `apply_sensor` will draw `k1` and pass it in. Use `cv2.INTER_LINEAR` + `cv2.BORDER_REFLECT_101` for boundaries (matches the `attacks/print.py` convention).

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-face/tests/test_sensor_a2.py`:

```python
def test_lens_distort_identity_when_k1_zero():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _lens_distort(img, k1=0.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # k1=0 with linear interpolation is exactly identity at integer sample grid
    assert np.array_equal(out, img)


def test_lens_distort_changes_image_when_k1_nonzero():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(1).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _lens_distort(img, k1=0.10)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Non-zero k1 must visibly change a non-degenerate image
    assert not np.array_equal(out, img)


def test_lens_distort_deterministic():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(2).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out1 = _lens_distort(img, k1=0.08)
    out2 = _lens_distort(img, k1=0.08)
    assert np.array_equal(out1, out2)


def test_lens_distort_barrel_and_pincushion_differ():
    from pad_synth_face.sensor import _lens_distort

    img = (np.random.default_rng(3).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    barrel = _lens_distort(img, k1=-0.10)
    pincushion = _lens_distort(img, k1=0.10)
    assert not np.array_equal(barrel, pincushion)
```

- [ ] **Step 2: Run, verify failure**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k lens
```

Expected: 4 FAIL with `ImportError: cannot import name '_lens_distort'`.

- [ ] **Step 3: Implement `_lens_distort`**

Add `import cv2` near the existing imports in `pad-synth-face/src/pad_synth_face/sensor.py` (if not already imported — it isn't), then add the helper after `_white_balance`:

```python
def _lens_distort(img: np.ndarray, k1: float) -> np.ndarray:
    """Radial (Brown-Conrady k1-only) distortion via cv2.remap.

    k1=0 is identity. k1>0 is pincushion, k1<0 is barrel. Normalised radius
    `r` measured from image centre, displaced by r' = r * (1 + k1*r²).
    """
    h, w = img.shape[:2]
    if k1 == 0.0:
        return img.copy()
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Normalise so r=1 at the image corner (half-diagonal).
    r_norm = float(np.hypot(cy, cx))
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xv - cx) / r_norm
    dy = (yv - cy) / r_norm
    r2 = dx * dx + dy * dy
    factor = 1.0 + k1 * r2
    map_x = (dx * factor) * r_norm + cx
    map_y = (dy * factor) * r_norm + cy
    return cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k lens
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor_a2.py
git commit -m "feat(pad-a2): _lens_distort helper (Brown-Conrady k1-only via cv2.remap)

Pure helper: k1=0 is identity, k1>0 pincushion, k1<0 barrel. Uses
BORDER_REFLECT_101 to match the convention in attacks/print.py.
Wired into apply_sensor in a later task."
```

---

## Task 3: Implement `_motion_blur` (directional line kernel)

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py` (add helper)
- Test: `pad-synth-face/tests/test_sensor_a2.py` (append tests)

**Context:** A directional line kernel of length `L` at angle `θ` (radians). Real handheld motion blur is directional; a Gaussian is the wrong shape and is itself a fingerprint. Helper signature: `_motion_blur(img: np.ndarray, length_px: int, angle_rad: float) -> np.ndarray`. `length_px = 1` is identity (a 1×1 kernel of value 1.0). For `length_px >= 2`, build an `(L, L)` kernel, draw a 1-px line through the centre at angle `θ` (use `cv2.line` for sub-pixel-free rasterisation), normalise to sum 1, then `cv2.filter2D`. Wrap `θ` to `[0, π)` since lines are direction-agnostic.

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-face/tests/test_sensor_a2.py`:

```python
def test_motion_blur_identity_when_length_one():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _motion_blur(img, length_px=1, angle_rad=0.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_motion_blur_smooths_when_length_large():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(1).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _motion_blur(img, length_px=7, angle_rad=0.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Linear smoothing of a high-frequency noise image must reduce variance
    assert out.var() < img.var() * 0.85


def test_motion_blur_direction_matters():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(2).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    horiz = _motion_blur(img, length_px=7, angle_rad=0.0)
    vert = _motion_blur(img, length_px=7, angle_rad=np.pi / 2.0)
    assert not np.array_equal(horiz, vert)


def test_motion_blur_deterministic():
    from pad_synth_face.sensor import _motion_blur

    img = (np.random.default_rng(3).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out1 = _motion_blur(img, length_px=5, angle_rad=0.6)
    out2 = _motion_blur(img, length_px=5, angle_rad=0.6)
    assert np.array_equal(out1, out2)
```

- [ ] **Step 2: Run, verify failure**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k motion
```

Expected: 4 FAIL with `ImportError`.

- [ ] **Step 3: Implement `_motion_blur`**

Add after `_lens_distort`:

```python
def _motion_blur(img: np.ndarray, length_px: int, angle_rad: float) -> np.ndarray:
    """Directional line-kernel blur. length_px=1 is identity.

    Draws a 1-px line through the centre of an (L, L) kernel at the given
    angle, normalises to sum 1, applies via cv2.filter2D.
    """
    L = int(length_px)
    if L <= 1:
        return img.copy()
    kernel = np.zeros((L, L), dtype=np.float32)
    cx = (L - 1) / 2.0
    cy = (L - 1) / 2.0
    half = (L - 1) / 2.0
    dx = np.cos(angle_rad) * half
    dy = np.sin(angle_rad) * half
    x0 = int(round(cx - dx))
    y0 = int(round(cy - dy))
    x1 = int(round(cx + dx))
    y1 = int(round(cy + dy))
    cv2.line(kernel, (x0, y0), (x1, y1), color=1.0, thickness=1)
    s = kernel.sum()
    if s <= 0.0:  # degenerate (shouldn't happen for L>=2, defensive)
        return img.copy()
    kernel /= s
    return cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k motion
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor_a2.py
git commit -m "feat(pad-a2): _motion_blur helper (directional line kernel)

Pure helper: length_px=1 is identity; otherwise rasterises a 1-px line
at angle_rad through an (L, L) kernel via cv2.line, normalises, applies
with cv2.filter2D. Wired into apply_sensor in a later task."
```

---

## Task 4: Replace `_noise` body with shot + read model

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py:54-57` (existing `_noise`)
- Test: `pad-synth-face/tests/test_sensor_a2.py` (append tests)

**Context:** Current `_noise(img, iso, rng)` is a single Gaussian with `sigma = 0.5 + iso/800 * 4`. Replace with shot + read per spec §4.3. Signature unchanged so all existing call-sites keep working. Per spec: `shot_sigma = sqrt(max(signal, 1)) * (iso/800) * 0.5`, `read_sigma = 1.5` (fixed). Both are signal-dependent / fixed Gaussian respectively. Output remains `uint8`, clipped to `[0, 255]`.

The existing `test_apply_sensor_adds_noise` test (in `test_sensor.py`) asserts `out.std() > 1.0` on a flat image — that must still pass under the new noise model. With a flat input of value 128 and ISO ~450 (mid of mobile range), `shot_sigma ≈ sqrt(128) * (450/800) * 0.5 ≈ 3.2`, `read ≈ 1.5` → combined ≈ 3.5. Well above 1.0. Existing assertion stays intact.

- [ ] **Step 1: Write the failing tests for the new noise model**

Append to `pad-synth-face/tests/test_sensor_a2.py`:

```python
def test_noise_scales_with_signal_level():
    """Shot noise must be larger on bright pixels than dark pixels."""
    from pad_synth_face.sensor import _noise

    dark = np.full((128, 128, 3), 10, dtype=np.uint8)
    bright = np.full((128, 128, 3), 200, dtype=np.uint8)
    rng_d = sample_rng(0)
    rng_b = sample_rng(0)  # identical rng so only the signal differs
    noisy_dark = _noise(dark, iso=800, rng=rng_d)
    noisy_bright = _noise(bright, iso=800, rng=rng_b)
    # bright signal -> larger shot sigma -> wider noise std
    assert noisy_bright.astype(np.float32).std() > noisy_dark.astype(np.float32).std()


def test_noise_scales_with_iso():
    """Doubling ISO must measurably increase noise on a fixed signal."""
    from pad_synth_face.sensor import _noise

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    low = _noise(img, iso=100, rng=sample_rng(0))
    high = _noise(img, iso=1600, rng=sample_rng(0))
    assert high.astype(np.float32).std() > low.astype(np.float32).std()


def test_noise_deterministic_given_rng():
    from pad_synth_face.sensor import _noise

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    out1 = _noise(img, iso=400, rng=sample_rng(5))
    out2 = _noise(img, iso=400, rng=sample_rng(5))
    assert np.array_equal(out1, out2)


def test_noise_jitters_with_rng():
    from pad_synth_face.sensor import _noise

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    out1 = _noise(img, iso=400, rng=sample_rng(5))
    out2 = _noise(img, iso=400, rng=sample_rng(6))
    assert not np.array_equal(out1, out2)
```

- [ ] **Step 2: Run, verify the signal-scaling test fails (the others may already pass with old Gaussian)**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k noise
```

Expected: `test_noise_scales_with_signal_level` FAILS (old pure Gaussian is signal-independent; the std on dark vs bright is statistically identical). The other three (iso-scaling, determinism, jitter) likely PASS already — that's fine; they're guardrails the new model also needs to satisfy.

- [ ] **Step 3: Replace `_noise` body**

In `pad-synth-face/src/pad_synth_face/sensor.py`, replace the existing `_noise` function:

```python
def _noise(img: np.ndarray, iso: int, rng: np.random.Generator) -> np.ndarray:
    """Shot (signal-dependent) + read (fixed) noise.

    Shot: Poisson approximated as Gaussian with sigma=sqrt(signal),
    scaled by iso/800 * 0.5. Read: fixed-magnitude electronics floor.
    """
    signal = img.astype(np.float32)
    shot_sigma = np.sqrt(np.maximum(signal, 1.0)) * (iso / 800.0) * 0.5
    shot = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32) * shot_sigma
    read = rng.normal(0.0, 1.5, size=signal.shape).astype(np.float32)
    return np.clip(signal + shot + read, 0, 255).astype(np.uint8)
```

(Per-element `shot_sigma` requires drawing standard-normal then scaling — `rng.normal(0.0, shot_sigma, ...)` works too but `np.random.Generator.normal` accepts an array `scale` argument and is equally valid; the standard-normal-then-multiply form above is unambiguous.)

- [ ] **Step 4: Run all noise tests + the full sensor suite**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py pad-synth-face/tests/test_sensor.py -v
```

Expected: all 4 noise A2 tests PASS; existing `test_sensor.py` (including `test_apply_sensor_adds_noise` asserting `std > 1.0` on flat 128) still PASSES.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor_a2.py
git commit -m "feat(pad-a2): replace _noise body with shot + read model

Shot noise (Poisson-approximated Gaussian, sigma=sqrt(signal)) scales
with ISO; read noise is a fixed-sigma=1.5 Gaussian electronics floor.
Signature unchanged. Existing test_apply_sensor_adds_noise still passes."
```

---

## Task 5: Implement `_jpeg_chain` (multi-pass JPEG recompression)

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py` (add helper; keep `_jpeg_roundtrip` private — `_jpeg_chain` will call it internally)
- Test: `pad-synth-face/tests/test_sensor_a2.py` (append tests)

**Context:** Real social-media images pass through capture → app encode → server re-encode → CDN re-encode. A single roundtrip leaves a distinguishable single-encode signature. The helper signature: `_jpeg_chain(img: np.ndarray, qf_per_pass: list[int]) -> np.ndarray`. For each `qf` in the list, encode → decode (reusing the existing `_jpeg_roundtrip`). The caller (`apply_sensor`, next task) draws `n_passes` and the per-pass QFs from `rng`. `len(qf_per_pass) == 1` is the current behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-face/tests/test_sensor_a2.py`:

```python
def test_jpeg_chain_single_pass_matches_jpeg_roundtrip():
    from pad_synth_face.sensor import _jpeg_chain, _jpeg_roundtrip

    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    chained = _jpeg_chain(img, qf_per_pass=[85])
    single = _jpeg_roundtrip(img, qf=85)
    assert np.array_equal(chained, single)


def test_jpeg_chain_multiple_passes_degrades_more_than_single():
    from pad_synth_face.sensor import _jpeg_chain

    img = (np.random.default_rng(1).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    one_pass = _jpeg_chain(img, qf_per_pass=[75])
    three_pass = _jpeg_chain(img, qf_per_pass=[75, 75, 75])
    # Each re-encode at the same QF accumulates loss; pixel-wise L2 grows.
    delta_one = np.abs(img.astype(np.int16) - one_pass.astype(np.int16)).mean()
    delta_three = np.abs(img.astype(np.int16) - three_pass.astype(np.int16)).mean()
    assert delta_three > delta_one


def test_jpeg_chain_deterministic():
    from pad_synth_face.sensor import _jpeg_chain

    img = (np.random.default_rng(2).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out1 = _jpeg_chain(img, qf_per_pass=[90, 80])
    out2 = _jpeg_chain(img, qf_per_pass=[90, 80])
    assert np.array_equal(out1, out2)


def test_jpeg_chain_preserves_shape_dtype():
    from pad_synth_face.sensor import _jpeg_chain

    img = (np.random.default_rng(3).integers(0, 256, size=(64, 64, 3))).astype(np.uint8)
    out = _jpeg_chain(img, qf_per_pass=[88, 82, 78])
    assert out.shape == img.shape
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run, verify failure**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k jpeg
```

Expected: 4 FAIL with `ImportError: cannot import name '_jpeg_chain'`.

- [ ] **Step 3: Implement `_jpeg_chain`**

Add after `_jpeg_roundtrip`:

```python
def _jpeg_chain(img: np.ndarray, qf_per_pass: list[int]) -> np.ndarray:
    """Apply n encode→decode JPEG passes with the given per-pass quality factors.

    len(qf_per_pass) == 1 is the single-roundtrip baseline. Multi-pass
    simulates the capture → app → server → CDN re-encode chain.
    """
    out = img
    for qf in qf_per_pass:
        out = _jpeg_roundtrip(out, qf)
    return out
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k jpeg
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor_a2.py
git commit -m "feat(pad-a2): _jpeg_chain helper (multi-pass JPEG recompression)

Pure helper iterating _jpeg_roundtrip over per-pass QF list. Single-pass
matches existing roundtrip; multi-pass accumulates loss. apply_sensor
will draw n_passes and per-pass QFs in a later task."
```

---

## Task 6: Reorder `apply_sensor` to the physical pipeline + record new params

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py:70-86` (existing `apply_sensor`)
- Test: `pad-synth-face/tests/test_sensor_a2.py` (append integration tests)

**Context:** This task wires everything together. New pipeline order per spec §5:

```
lens_distort → motion_blur → noise → vignette → white_balance → jpeg_chain
```

`apply_sensor` draws all rng-driven parameters at the top, applies the chain, returns `(uint8 img, params dict)`. The `params` dict gains five new keys: `lens_k1` (float), `motion_blur_L` (int), `motion_blur_theta` (float radians), `jpeg_passes` (int), `jpeg_qf_per_pass` (list[int]). The existing `jpeg_qf` key is *removed* (subsumed by `jpeg_qf_per_pass`); the existing `iso`, `wb_k`, `preset` keys stay.

**Important:** Existing `test_apply_sensor_*` tests in `test_sensor.py` use specific seeds. Output bytes will change. Tests that assert *shape*, *dtype*, *params dict key presence*, or qualitative properties (vignette darkens corners, noise raises variance, presets differ) will still pass. Tests that assert *exact pixel values* would not — there are none currently, but verify in step 1.

**Important:** The pipeline calls `apply_sensor` with a `rng` from `sample_rng`. The pipeline doesn't read individual `sensor_params` keys by name (just stores the dict in the manifest) — verified at `pad-synth-face/src/pad_synth_face/pipeline.py:188,209,241,266`. Removing `jpeg_qf` is therefore safe for code, but downstream report/analysis tooling reading `manifest.jsonl` may expect it. Search for any reader before committing (step 7).

- [ ] **Step 1: Confirm no existing test asserts exact pixel values from `apply_sensor`**

```bash
grep -n "array_equal\|allclose\|assert.*== np.uint8\|== 128" pad-synth-face/tests/test_sensor.py
```

Expected: no `array_equal` / `allclose` calls against literal arrays. (Equality with `img.shape` or scalars is fine; literal pixel-value comparisons would not be.)

- [ ] **Step 2: Write the failing integration tests**

Append to `pad-synth-face/tests/test_sensor_a2.py`:

```python
def test_apply_sensor_records_new_params_keys():
    img = (np.random.default_rng(0).integers(0, 256, size=(128, 128, 3))).astype(np.uint8)
    out, params = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(0))
    for key in ("lens_k1", "motion_blur_L", "motion_blur_theta",
                "jpeg_passes", "jpeg_qf_per_pass"):
        assert key in params, f"missing param key: {key}"
    # Existing keys still present
    for key in ("iso", "wb_k", "preset"):
        assert key in params
    # Types
    assert isinstance(params["lens_k1"], float)
    assert isinstance(params["motion_blur_L"], int)
    assert isinstance(params["motion_blur_theta"], float)
    assert isinstance(params["jpeg_passes"], int)
    assert isinstance(params["jpeg_qf_per_pass"], list)
    assert len(params["jpeg_qf_per_pass"]) == params["jpeg_passes"]


def test_apply_sensor_params_are_within_preset_ranges():
    img = (np.random.default_rng(0).integers(0, 256, size=(128, 128, 3))).astype(np.uint8)
    for seed in range(10):
        _, params = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(seed))
        p = MOBILE_FRONT_2024
        assert p.lens_k1_range[0] <= params["lens_k1"] <= p.lens_k1_range[1]
        assert p.motion_blur_px_range[0] <= params["motion_blur_L"] <= p.motion_blur_px_range[1]
        assert p.jpeg_passes_range[0] <= params["jpeg_passes"] <= p.jpeg_passes_range[1]
        assert 0.0 <= params["motion_blur_theta"] < np.pi
        for qf in params["jpeg_qf_per_pass"]:
            assert p.jpeg_qf_range[0] <= qf <= p.jpeg_qf_range[1]


def test_apply_sensor_anti_watermark_byte_level_jitter():
    """Two different seeds on the same input must produce byte-different outputs."""
    img = (np.random.default_rng(0).integers(0, 256, size=(128, 128, 3))).astype(np.uint8)
    out1, _ = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(0))
    out2, _ = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(1))
    assert not np.array_equal(out1, out2)


def test_apply_sensor_still_deterministic_with_same_seed():
    img = (np.random.default_rng(0).integers(0, 256, size=(128, 128, 3))).astype(np.uint8)
    out1, p1 = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(42))
    out2, p2 = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(42))
    assert np.array_equal(out1, out2)
    assert p1 == p2
```

- [ ] **Step 3: Run, verify failure**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor_a2.py -v -k apply_sensor
```

Expected: `test_apply_sensor_records_new_params_keys` and `test_apply_sensor_params_are_within_preset_ranges` FAIL (`KeyError`). The other two may PASS already.

- [ ] **Step 4: Rewrite `apply_sensor`**

Replace the existing function in `pad-synth-face/src/pad_synth_face/sensor.py`:

```python
def apply_sensor(
    img: np.ndarray, preset: SensorPreset, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    # Draw all per-sample parameters up-front from rng so the order of consumption
    # is stable and the params dict is fully formed before any pixel work.
    iso = int(rng.integers(preset.iso_range[0], preset.iso_range[1] + 1))
    kelvin = int(rng.integers(preset.wb_k_range[0], preset.wb_k_range[1] + 1))
    lens_k1 = float(rng.uniform(preset.lens_k1_range[0], preset.lens_k1_range[1]))
    motion_L = int(rng.integers(preset.motion_blur_px_range[0],
                                preset.motion_blur_px_range[1] + 1))
    motion_theta = float(rng.uniform(0.0, np.pi))
    n_passes = int(rng.integers(preset.jpeg_passes_range[0],
                                preset.jpeg_passes_range[1] + 1))
    qf_per_pass = [
        int(rng.integers(preset.jpeg_qf_range[0], preset.jpeg_qf_range[1] + 1))
        for _ in range(n_passes)
    ]

    # Physical pipeline order: optics -> motion -> sensor -> ISP -> compression.
    out = _lens_distort(img, lens_k1)
    out = _motion_blur(out, motion_L, motion_theta)
    out = _noise(out, iso, rng)
    out = _vignette(out, preset.vignette_strength)
    out = _white_balance(out, kelvin)
    out = _jpeg_chain(out, qf_per_pass)

    params = {
        "iso": iso,
        "wb_k": kelvin,
        "lens_k1": lens_k1,
        "motion_blur_L": motion_L,
        "motion_blur_theta": motion_theta,
        "jpeg_passes": n_passes,
        "jpeg_qf_per_pass": qf_per_pass,
        "preset": preset.name,
    }
    return out, params
```

- [ ] **Step 5: Run the full sensor suite**

```bash
.venv/bin/pytest pad-synth-face/tests/test_sensor.py pad-synth-face/tests/test_sensor_a2.py -v
```

Expected: all A2 tests PASS; all existing `test_sensor.py` tests still PASS (shape, dtype, determinism, vignette-darker, preset distinctness).

- [ ] **Step 6: Run the full pad-synth-face suite to catch any pipeline integration regression**

```bash
.venv/bin/pytest pad-synth-face/tests/ -v --ignore=pad-synth-face/tests/test_determinism_golden.py
```

Expected: all pass except possibly tests that compare exact pixel bytes (none expected). If `test_determinism_golden.py` is *not* ignored it will fail — that's the next task.

- [ ] **Step 7: Verify no downstream code reads `params["jpeg_qf"]`**

```bash
grep -rn 'jpeg_qf"\|"jpeg_qf' pad-synth-face/src/ pad-synth-core/src/ scripts/ docs/ 2>/dev/null | grep -v 'jpeg_qf_range\|jpeg_qf_per_pass'
```

Expected: no matches (or only comments). If a reader exists, decide whether to (a) keep a `jpeg_qf` alias in the params dict for back-compat, or (b) update the reader. Default: keep it simple and update the reader; if the only reader is a script that mean-aggregates QFs over the manifest, swap to `mean(jpeg_qf_per_pass)`.

- [ ] **Step 8: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor_a2.py
git commit -m "feat(pad-a2): reorder apply_sensor to physical pipeline + record A2 params

Pipeline: lens -> motion -> noise -> vignette -> WB -> JPEG-chain. New
sensor_params keys: lens_k1, motion_blur_L, motion_blur_theta,
jpeg_passes, jpeg_qf_per_pass (the latter subsumes the old jpeg_qf).
Existing test_sensor.py contract assertions (shape, dtype, determinism,
vignette-darker, preset distinctness) remain green."
```

---

## Task 7: Regenerate determinism golden hashes

**Files:**
- Regenerate: `tests/golden/golden_hashes.json`

**Context:** Every sensor-stage byte change invalidates the golden hashes (this is by design — that's what the golden is for). Regenerate with `PAD_SYNTH_UPDATE_GOLDEN=1`, verify the regenerated file is a clean overwrite, commit. No code changes.

- [ ] **Step 1: Run determinism test once to confirm it fails (proof the change is observable)**

```bash
.venv/bin/pytest tests/test_determinism_golden.py -v
```

Expected: FAIL with `AssertionError: Determinism regression. If intentional, run PAD_SYNTH_UPDATE_GOLDEN=1 pytest tests/test_determinism_golden.py`.

- [ ] **Step 2: Regenerate the golden file**

```bash
PAD_SYNTH_UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_determinism_golden.py -v
```

Expected: PASS (the test self-overwrites the golden file and returns).

- [ ] **Step 3: Re-run without the env var to confirm determinism**

```bash
.venv/bin/pytest tests/test_determinism_golden.py -v
```

Expected: PASS (hashes match the freshly-written file).

- [ ] **Step 4: Inspect the diff and confirm only hashes changed (not the sample-id set)**

```bash
git diff tests/golden/golden_hashes.json | head -60
```

Expected: hash strings differ; sample-id keys are unchanged (same number of entries, same `sample_id` strings on the left of each `:`).

- [ ] **Step 5: Commit**

```bash
git add tests/golden/golden_hashes.json
git commit -m "test(pad-a2): regenerate determinism golden after sensor pipeline change

A2 sensor pipeline (lens + motion + shot/read noise + JPEG-chain) emits
different bytes than the pre-A2 chain by design. Sample-id set unchanged;
only output hashes differ."
```

---

## Task 8: Run the full repo test suite, fix any incidental breakage

**Files:**
- Modify (only if a test breaks): whatever tests assert pre-A2-specific sensor behaviour

**Context:** Catch-all integration check across pad-synth-face, pad-synth-core, and top-level tests. No new assertions written here unless a real breakage surfaces. The expected outcome is "all green" — this task exists to make that explicit and to fix anything that did break.

- [ ] **Step 1: Run the full suite**

```bash
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -v
```

Expected: all PASS. If anything fails:
- A test that asserts `params["jpeg_qf"]` exists → update to `params["jpeg_qf_per_pass"][0]` or `mean(params["jpeg_qf_per_pass"])` (Task 6 Step 7 should have caught this, but verify here).
- A test that asserts an exact pixel value → that test was pre-A2-fingerprint-specific; it shouldn't exist, but if it does, mark it as outdated and update it to a property-based assertion.
- A genuine bug in the new code → fix and re-run.

- [ ] **Step 2: Commit only if changes were needed**

```bash
git status
# If nothing changed, skip the commit.
# If a fix was needed:
git add <files>
git commit -m "test(pad-a2): update <test name> for new sensor_params schema"
```

---

## Task 9: Local dataset regeneration + sweep launch on Spark

**Files:**
- Regenerate: 12 datasets under `datasets/mask_set{a,b}_d{1,2,3}/` and `datasets/mix_set{a,b}_d{1,2,3}/` at 224×224
- Output: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/`, `runs_mix_224_L4_A2/`

**Context:** This task is operational, not code-change. It regenerates the 12 evaluation datasets locally with the new sensor, rsyncs them + the updated code to the Spark, runs the existing `spark_sweep.py` for 9 mask + 9 mix L4 cells, and pulls per-cell JSON + summary CSV back. Reuses the exact pattern from the 2026-05-30 B2 sweep — no new infra.

The dataset paths come from the existing `configs/runs/mask_*.yaml` and `configs/runs/mix_*.yaml` configs. Each config drives `run_pipeline` which writes to its own output dir. The B2 sweep used `datasets/mask_set{a,b}_d{1,2,3}` and `datasets/mix_set{a,b}_d{1,2,3}` as input names to `spark_sweep.py`. We overwrite those same directories with the regenerated A2 data.

**Important:** The dataset regen ABSOLUTELY MUST use the A2-extended sensor — that's the whole point. The pipeline imports `apply_sensor` from `pad_synth_face.sensor`, which is the file we just rewrote. So a fresh dataset-regen run automatically applies A2. Sanity-check this in step 2.

- [ ] **Step 1: Locate the existing dataset-regen entrypoint**

```bash
ls scripts/ | grep -Ei 'build|prepare|regen|gen'
grep -n "configs/runs" scripts/*.py | head -20
```

Expected: an existing script (likely `scripts/build_datasets.py`, `scripts/prepare_*.py`, or similar) that iterates over configs and runs the pipeline. If none exists as a single entrypoint, the 12 configs can be driven manually:

```bash
for cfg in configs/runs/mask_set{a,b}_d{1,2,3}.yaml configs/runs/mix_set{a,b}_d{1,2,3}.yaml; do
  .venv/bin/python -m pad_synth_face.cli run --config "$cfg"
done
```

(Or whatever the CLI invocation pattern is — check `pad-synth-face/tests/test_cli.py` for the actual command.)

- [ ] **Step 2: Sanity-check that the dataset regen uses A2**

Pick one config, regen, and inspect a manifest entry:

```bash
rm -rf datasets/mask_seta_d3
.venv/bin/python -m pad_synth_face.cli run --config configs/runs/mask_seta_d3.yaml
head -1 datasets/mask_seta_d3/manifest.jsonl | python3 -m json.tool | grep -E "lens_k1|motion_blur|jpeg_passes"
```

Expected: the manifest entry's `sensor_params` block contains `lens_k1`, `motion_blur_L`, `motion_blur_theta`, `jpeg_passes`, `jpeg_qf_per_pass`. If those keys are absent, the pipeline isn't using the updated sensor — investigate before continuing.

- [ ] **Step 3: Regenerate the remaining 11 datasets**

```bash
for cfg in configs/runs/mask_setb_d3.yaml configs/runs/mask_set{a,b}_d{1,2}.yaml configs/runs/mix_set{a,b}_d{1,2,3}.yaml; do
  ds=$(basename "$cfg" .yaml)
  rm -rf "datasets/$ds"
  .venv/bin/python -m pad_synth_face.cli run --config "$cfg"
done
```

Expected: 12 dataset directories under `datasets/`, each with `manifest.jsonl` + sample images. Spec §9 estimates ~10–15 min total. Sanity-check one mix manifest entry the same way as step 2.

- [ ] **Step 4: Push code + datasets to the Spark**

```bash
# Code
rsync -azP --delete --exclude='.venv' --exclude='datasets/' --exclude='__pycache__' \
  --exclude='.git/' --exclude='*.egg-info' \
  /Users/stuartwells/test/ swells@spark-50d2.local:~/ml/projects/pad-spark/

# Datasets (12 dirs)
rsync -azP datasets/mask_set{a,b}_d{1,2,3}/ datasets/mix_set{a,b}_d{1,2,3}/ \
  swells@spark-50d2.local:~/ml/projects/pad-spark/datasets/ 2>/dev/null || \
  rsync -azP datasets/ swells@spark-50d2.local:~/ml/projects/pad-spark/datasets/
```

(The fallback `rsync -azP datasets/` syncs every dataset under `datasets/` — slower but unambiguous if the brace-glob doesn't expand the way you expect.)

Expected: rsync exits 0. Code SHA on the Spark matches local `git rev-parse HEAD`.

- [ ] **Step 5: Launch the mask sweep on the Spark (background)**

```bash
ssh swells@spark-50d2.local 'bash -lc "
cd ~/ml/projects/pad-spark
CELLS=\$(python3 -c \"print(\\\",\\\".join(f\\\"L4:{D}:{s}\\\" for D in (\\\"D1\\\",\\\"D2\\\",\\\"D3\\\") for s in (0,1,2)))\")
nohup .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mask_seta_d1 --set-b-d1 datasets/mask_setb_d1 \
  --set-a-d2 datasets/mask_seta_d2 --set-b-d2 datasets/mask_setb_d2 \
  --set-a-d3 datasets/mask_seta_d3 --set-b-d3 datasets/mask_setb_d3 \
  --set-a-d4 datasets/mask_seta_d3 --set-b-d4 datasets/mask_setb_d3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2 \
  --cells \"\$CELLS\" --device cuda \
  > /tmp/sweep_mask_a2.log 2>&1 &
echo \$!
"'
```

Expected: prints a PID. Sweep runs ~3 min per spec §9.

- [ ] **Step 6: Poll until the mask sweep finishes**

```bash
ssh swells@spark-50d2.local 'tail -5 /tmp/sweep_mask_a2.log; ls docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/ 2>/dev/null | wc -l'
```

Re-run every ~90 s. Expected end state: 9 per-cell JSON files + `summary.csv` in `runs_mask_224_L4_A2/`; log tail shows the sweep complete.

- [ ] **Step 7: Launch the mix sweep (same pattern, mix_ prefixes)**

```bash
ssh swells@spark-50d2.local 'bash -lc "
cd ~/ml/projects/pad-spark
CELLS=\$(python3 -c \"print(\\\",\\\".join(f\\\"L4:{D}:{s}\\\" for D in (\\\"D1\\\",\\\"D2\\\",\\\"D3\\\") for s in (0,1,2)))\")
nohup .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mix_seta_d1 --set-b-d1 datasets/mix_setb_d1 \
  --set-a-d2 datasets/mix_seta_d2 --set-b-d2 datasets/mix_setb_d2 \
  --set-a-d3 datasets/mix_seta_d3 --set-b-d3 datasets/mix_setb_d3 \
  --set-a-d4 datasets/mix_seta_d3 --set-b-d4 datasets/mix_setb_d3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2 \
  --cells \"\$CELLS\" --device cuda \
  > /tmp/sweep_mix_a2.log 2>&1 &
echo \$!
"'
```

Expected: prints a PID. ~4.5 min wall-time per spec §9.

- [ ] **Step 8: Poll until the mix sweep finishes**

Same polling pattern as Step 6, watching `runs_mix_224_L4_A2/`.

- [ ] **Step 9: Pull the two results dirs back**

```bash
rsync -azP swells@spark-50d2.local:~/ml/projects/pad-spark/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/ \
  /Users/stuartwells/test/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/

rsync -azP swells@spark-50d2.local:~/ml/projects/pad-spark/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2/ \
  /Users/stuartwells/test/docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2/
```

Expected: both directories exist locally with 9 JSON + 1 CSV each.

- [ ] **Step 10: Sanity-check the summary CSVs are well-formed**

```bash
head -3 docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/summary.csv
wc -l docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/summary.csv
wc -l docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2/summary.csv
```

Expected: header + 9 data rows (mask) and header + 9 rows (mix). Each row has a finite `xdomain_eer` value (not `nan`).

- [ ] **Step 11: Commit the sweep outputs**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4_A2/
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2/
git commit -m "report(pad-a2): L4+A2 sweep outputs (9 mask + 9 mix cells at 224)

Generated on Spark GB10 from the A2-extended sensor pipeline (commits
<auto-fill from log>). Report write-up follows in next commit."
```

(Substitute the actual commit hashes of Tasks 1–6 in the commit body if convenient.)

---

## Task 10: Append the A2 report section

**Files:**
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (append)

**Context:** The running sweep-results doc has sections for v1/v2/v2.1/real-bonafide/mask/mix/synth→real/A1@224/B2@224. Append a new "2026-05-31 update — A2 capture-realism" section that mirrors the B2 section's structure: setup → cross-domain EER tables (mask + mix, L4·D1/D2/D3 mean±std across 3 seeds) → ACER@5%APCER tables → headline finding (which of the four §2 decision-matrix branches we landed on) → comparison vs the 2026-05-30 L4 baseline → phase-recommendation update.

The numbers come from the JSON files written in Task 9. The headline finding is a one-paragraph judgement based on the L4·D3 mean cross-domain EER vs the 2026-05-30 baseline (mask 0.060, mix 0.059).

- [ ] **Step 1: Compute aggregates from the per-cell JSONs**

```bash
cd /Users/stuartwells/test
.venv/bin/python - <<'PY'
import json, statistics, pathlib
def agg(p):
    rows = [json.loads(f.read_text()) for f in pathlib.Path(p).glob("*.json")]
    # group (L, D) -> [eer per seed]
    g = {}
    for r in rows:
        key = (r["capacity"], r["data_level"])
        g.setdefault(key, []).append(r)
    for k, vs in sorted(g.items()):
        xeer = [v["xdomain_eer"] for v in vs]
        ideer = [v.get("indomain_eer") for v in vs]
        acer = [v.get("acer_at_5pct_apcer") for v in vs]
        print(f"{k}: xdomain_eer mean={statistics.mean(xeer):.3f} std={statistics.pstdev(xeer):.3f} "
              f"in_eer={statistics.mean(ideer):.3f} acer5={statistics.mean(acer):.3f}")
for tag in ("mask", "mix"):
    print(f"--- {tag} ---")
    agg(f"docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_{tag}_224_L4_A2")
PY
```

Expected: 6 lines (3 D-levels × 2 sweeps). Note any cell where `xdomain_eer mean ≤ 0.001` — that triggers the §2 "Any cell collapses to 0.000" branch (catastrophic fingerprint, requires diagnosis before the report can claim victory).

- [ ] **Step 2: Identify which §2 decision-matrix branch fired**

From the spec §2 matrix:
- L4+A2 mask-D3 mean < 0.060 (and mix < 0.059) → branch 1 ("EER drops"). New production baseline.
- |L4+A2 - L4| within ±0.010 → branch 2 ("flat"). A2 still ships but impact deferred to synth→real.
- L4+A2 substantially worse → branch 3 ("worsens"). Fingerprint diagnosis required before report claims success.
- Any cell ≤ 0.001 → branch 4 ("collapses"). Bisect the four effects.

If branch 3 or 4: stop here, write a diagnostic report instead of the success-case template, and revisit the sensor before declaring done.

- [ ] **Step 3: Write the report section**

Append to `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`:

```markdown
## 2026-05-31 update — A2 capture-realism on L4 pretrained

**Setup:** Mask + mix sweeps at 224×224 with `make_resnet18_pretrained`
(ImageNet weights), Adam lr=1e-3, 8 epochs, batch 32. Sensor pipeline
extended per spec `docs/superpowers/specs/2026-05-31-pad-a2-capture-realism-design.md`:
radial lens distortion, directional motion blur, shot+read sensor noise,
multi-pass JPEG recompression — each per-sample jittered, default-on in
the MOBILE/WEBCAM presets. Code SHA <fill>. Spark wall-time: ~<X> min
mask + ~<Y> min mix on GB10.

### L4+A2 cross-domain EER (mean ± std across 3 seeds)

|  | mask-only | integrated (print+replay+mask) |
|---|---|---|
| L4·D1 | <m1> ± <s1> | <m2> ± <s2> |
| L4·D2 | <m3> ± <s3> | <m4> ± <s4> |
| L4·D3 | <m5> ± <s5> | <m6> ± <s6> |

### L4+A2 in-domain EER (mean across 3 seeds)

|  | mask-only | integrated |
|---|---|---|
| L4·D1 | <…> | <…> |
| L4·D2 | <…> | <…> |
| L4·D3 | <…> | <…> |

### L4+A2 ACER@5% APCER (mean across 3 seeds)

|  | mask-only | integrated |
|---|---|---|
| L4·D1 | <…> | <…> |
| L4·D2 | <…> | <…> |
| L4·D3 | <…> | <…> |

### Comparison vs 2026-05-30 L4 baseline

| Cell | L4 baseline | L4+A2 | Δ |
|---|---|---|---|
| mask·D3 | 0.060 ± 0.012 | <…> | <…> |
| mix·D3  | 0.059 ± 0.015 | <…> | <…> |

### Headline finding

<one paragraph: which §2 decision-matrix branch fired and what it implies>

### Phase recommendation update

<one paragraph: where this leaves the queue — DFDC sweep / B1 / Tier-B>
```

Fill in every `<…>` with the actual aggregates from Step 1.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md
git commit -m "report(pad-a2): append 2026-05-31 A2 capture-realism sweep results

L4+A2 mask·D3 EER <X> ± <Y> (vs L4 baseline 0.060 ± 0.012); mix·D3
<X2> ± <Y2> (vs 0.059 ± 0.015). Decision matrix §2 branch: <which>."
```

---

## Task 11: Update project memory

**Files:**
- Modify: `/Users/stuartwells/.claude/projects/-Users-stuartwells-test/memory/pad-next-sub-projects.md`
- Modify: `/Users/stuartwells/.claude/projects/-Users-stuartwells-test/memory/MEMORY.md` (only if a new memory file is added)

**Context:** After every sub-project the queue memory gets a one-paragraph update describing what shipped and how the queue re-orders. This is the same housekeeping done after B2 (the queue memory already has the post-2026-05-30 capacity-spike entry).

- [ ] **Step 1: Append the post-A2 update**

Append a paragraph to `pad-next-sub-projects.md` summarising:
- The §2 decision-matrix branch that fired
- The new headline L4+A2 cross-domain EER numbers
- Whether A2 ships as the production capture chain (yes if branches 1 or 2 fired)
- Whether the next queue item changes (the default queue after A2 is DFDC / B1 / Tier-B)

Format-consistent with the existing post-B2 paragraph. No new memory file needed unless something genuinely new and non-derivable from code/git emerged (e.g., "shot-noise sigma needed to be halved to avoid a fingerprint" would qualify; "A2 worked" does not).

- [ ] **Step 2: Commit the memory update**

```bash
git -C /Users/stuartwells/.claude/projects/-Users-stuartwells-test/memory \
  add pad-next-sub-projects.md
git -C /Users/stuartwells/.claude/projects/-Users-stuartwells-test/memory \
  commit -m "memory: post-A2 queue update"
```

(The memory directory may or may not be a git repo; if not, just save the file.)

---

## Final Verification

Before declaring the plan complete, run from `/Users/stuartwells/test`:

```bash
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -v
git log --oneline feat/pad-a2-capture-realism ^main
```

Expected: all tests green; commit history shows ~10 commits (one per Task 1–6, the golden regen, possibly the Task 8 fix, the sweep outputs, the report).

Then hand off to `superpowers:finishing-a-development-branch` to merge or PR the branch per the user's established pattern ("merge to local main instead").
