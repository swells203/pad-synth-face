# PAD Print Physics v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the print attack with halftoning + ICC profile simulation (v2 physics), regenerate the determinism golden, then run a 27-cell Spark sweep on regenerated datasets to compare v1-vs-v2 cross-domain EER and determine whether physics is the lever that data scaling could not move.

**Architecture:** Pure-numpy halftoning (per-channel AM screening at standard rosette angles, dot-cell frequency driven by `print_dpi`) and an sRGB-space parameterized ICC transform (gamut compression + white-point shift + tone curve keyed by `paper_type`, strength scaled by a new `icc_profile_strength` ontology axis). Modifies `print.py` in place; bumps `ontology/face/print.yaml` version; regenerates the golden hashes file. The measurement sweep reuses the existing `spark_sweep.py` (already D4-aware) — only new configs and a new output subdir distinguish v2 from v1.

**Tech Stack:** Same as parent: Python 3.11+ (laptop) / 3.12 (Spark), numpy, opencv (existing import in print.py — kept for warp), PyTorch nightly cu128 on the Spark, pytest. No new external deps.

---

## Reference: facts the engineer needs

**Current `print.py` (verified).** Has 4 stages applied in order: paper tint (RGB multiply by `_PAPER_TINTS[paper_type]`), paper texture (per-pixel multiplicative noise), perspective warp (cv2), cutout. `PrintAttack` class: `__init__(self, ontology)`, `sample_params(self, rng) -> dict`, `simulate(self, bonafide, params, rng) -> np.ndarray`. Body converts `uint8 → float32/255`, applies tint+texture in float, then `clip*255 → uint8`, then warp + cutout. Existing helpers `_paper_texture`, `_perspective_warp`, `_apply_cutout`. The class API is stable; only internals change.

**Existing test at `pad-synth-face/tests/test_print_attack.py`** (3 tests): shape-and-dtype, modifies-image, deterministic-under-same-seed. These MUST continue to pass after wiring.

**Golden test (`tests/test_determinism_golden.py`).** Builds the basic 8-identity fixture (`build_fixture_bonafide`), runs the pipeline with seed=20260511, samples_per_bonafide=2, both print+replay attacks, sensor_preset=mobile-front-2024. Writes 32 entries to `tests/golden/golden_hashes.json` (16 bonafide + 8 print + 8 replay). Regenerate via `PAD_SYNTH_UPDATE_GOLDEN=1 pytest tests/test_determinism_golden.py`.

**Ontology loader (`pad-synth-core/src/pad_synth_core/ontology.py:51`)** has `Ontology.sample_params(self, rng)` that samples ONE value per axis in declaration order. Appending a new axis at the END of the YAML keeps all existing axes' sampled values bit-identical for the same seed.

**Spec.** `docs/superpowers/specs/2026-05-22-pad-print-physics-v2-design.md`. Halftoning formula in §3.1; ICC parameters in §4.1; transform pipeline in §4.2 (sRGB space); ontology in §5; measurement plan in §7.

**Halftoning correction over the spec wording:** the spec §3 phrasing leans toward a *line screen* (single-axis cosine grating); the implementation here uses a proper **2D dot screen** via `cos(2π·x'/cell) · cos(2π·y'/cell)` in rotated coords. Both are valid halftoning approaches; the 2D dot screen produces actual dots (matching the spec's "rosette" language).

---

## Task 1: Halftoning helpers + tests

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py` (add module-level helpers, NO wiring into `simulate` yet)
- Create: `pad-synth-face/tests/test_print_halftone.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_print_halftone.py`:
```python
import numpy as np

from pad_synth_face.attacks.print import (
    _apply_halftone,
    _inv_cmyk,
    _to_cmyk,
)


def test_to_cmyk_roundtrip_neutral_gray():
    """Pure gray RGB(128,128,128) -> CMYK and back yields (128,128,128)."""
    rgb = np.full((4, 4, 3), 128, dtype=np.float32) / 255.0
    cmyk = _to_cmyk(rgb)
    back = _inv_cmyk(cmyk)
    assert back.shape == rgb.shape
    np.testing.assert_allclose(back, rgb, atol=1e-3)


def test_to_cmyk_roundtrip_white_and_black():
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    np.testing.assert_allclose(_inv_cmyk(_to_cmyk(rgb)), rgb, atol=1e-3)
    rgb = np.ones((2, 2, 3), dtype=np.float32)
    np.testing.assert_allclose(_inv_cmyk(_to_cmyk(rgb)), rgb, atol=1e-3)


def test_halftone_changes_dot_count_with_dpi():
    """Lower DPI -> larger cells -> fewer transitions/dots in the screen."""
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    low = _apply_halftone(rgb, print_dpi=150)
    high = _apply_halftone(rgb, print_dpi=1200)
    # Count horizontal sign-flip transitions in the green channel of each.
    def transitions(img):
        bin_ = (img[:, :, 1] > 0.5).astype(np.int32)
        return int(np.abs(np.diff(bin_, axis=1)).sum())
    n_low = transitions(low)
    n_high = transitions(high)
    assert n_low < n_high, f"expected low-DPI={n_low} < high-DPI={n_high}"


def test_halftone_deterministic():
    """Same input -> byte-identical output (no RNG in halftoning)."""
    rgb = np.full((32, 32, 3), 0.4, dtype=np.float32)
    a = _apply_halftone(rgb, print_dpi=300)
    b = _apply_halftone(rgb, print_dpi=300)
    assert np.array_equal(a, b)


def test_halftone_preserves_shape_and_dtype():
    rgb = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    out = _apply_halftone(rgb, print_dpi=300)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/stuartwells/test && .venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone.py -q 2>&1 | tail -5`
Expected: collection error / `ImportError: cannot import name '_apply_halftone' from 'pad_synth_face.attacks.print'`.

- [ ] **Step 3: Add helpers to `print.py`**

In `pad-synth-face/src/pad_synth_face/attacks/print.py`, AFTER the `_apply_cutout` function and BEFORE the `class PrintAttack:` line, insert:

```python
# --- v2 physics: halftoning ---------------------------------------------------

# CMYK rosette angles (degrees) per standard 4-color print convention.
_HALFTONE_ANGLES_DEG: tuple[float, float, float, float] = (15.0, 75.0, 0.0, 45.0)


def _to_cmyk(rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] (H,W,3) -> CMYK float [0,1] (H,W,4). No profile math."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.where(k < 1.0, 1.0 - k, 1.0)
    c = np.where(k < 1.0, (1.0 - r - k) / denom, 0.0)
    m = np.where(k < 1.0, (1.0 - g - k) / denom, 0.0)
    y = np.where(k < 1.0, (1.0 - b - k) / denom, 0.0)
    return np.stack([c, m, y, k], axis=-1).astype(np.float32)


def _inv_cmyk(cmyk: np.ndarray) -> np.ndarray:
    """CMYK float [0,1] (H,W,4) -> RGB float [0,1] (H,W,3)."""
    c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _dot_screen(h: int, w: int, cell_px: float, angle_deg: float) -> np.ndarray:
    """2D rotated cosine dot screen, range [0,1]. Deterministic — no RNG."""
    theta = np.deg2rad(angle_deg)
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    xp = xv * np.cos(theta) + yv * np.sin(theta)
    yp = -xv * np.sin(theta) + yv * np.cos(theta)
    grid = np.cos(2.0 * np.pi * xp / cell_px) * np.cos(2.0 * np.pi * yp / cell_px)
    return (0.5 + 0.5 * grid).astype(np.float32)


def _halftone_channel(channel: np.ndarray, cell_px: float, angle_deg: float) -> np.ndarray:
    """Binary halftone: pixel ON where channel value > screen threshold."""
    screen = _dot_screen(channel.shape[0], channel.shape[1], cell_px, angle_deg)
    return (channel > screen).astype(np.float32)


def _apply_halftone(rgb: np.ndarray, print_dpi: int | float) -> np.ndarray:
    """Per-channel AM halftoning at the standard rosette angles.

    rgb: float [0,1] (H,W,3); print_dpi drives the dot-cell frequency.
    Returns float [0,1] (H,W,3). Deterministic; no RNG.
    """
    cell_px = max(2.0, round(8.0 * 150.0 / float(print_dpi)))
    cmyk = _to_cmyk(rgb)
    out = np.empty_like(cmyk)
    for i, angle in enumerate(_HALFTONE_ANGLES_DEG):
        out[..., i] = _halftone_channel(cmyk[..., i], cell_px, angle)
    return _inv_cmyk(out)
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone.py -q 2>&1 | tail -5`
Expected: 5 passed.

- [ ] **Step 5: Full suite green (NO wiring yet — golden test should still pass)**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 143 passed, 1 skipped, 4 warnings (138 prior + 5 new — `PrintAttack.simulate` is unchanged at this point, only new module-level helpers added).

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/print.py pad-synth-face/tests/test_print_halftone.py
git commit -m "feat(pad-face): halftoning helpers (CMYK + rotated dot screen)"
```

---

## Task 2: ICC transform helper + tests

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py` (add `_apply_icc` and the `_ICC_PARAMS` table, NO wiring into `simulate` yet)
- Create: `pad-synth-face/tests/test_print_icc.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_print_icc.py`:
```python
import numpy as np

from pad_synth_face.attacks.print import _ICC_PARAMS, _apply_icc


def test_icc_params_table_complete():
    """All three paper types present with the spec'd parameter tuples."""
    assert set(_ICC_PARAMS.keys()) == {"matte", "glossy", "photo"}
    # Tuple shape: (gamut_compression: float, (Δx, Δy): tuple, tone_gamma: float)
    for v in _ICC_PARAMS.values():
        assert len(v) == 3
        assert isinstance(v[0], float)
        assert isinstance(v[1], tuple) and len(v[1]) == 2
        assert isinstance(v[2], float)


def test_icc_strength_zero_is_near_identity():
    """At strength=0, gamut compression is off; output ~= input modulo
    white-point shift and tone gamma (which still apply at full strength)."""
    rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = _apply_icc(rgb, "glossy", strength=0.0)
    # Glossy has tiny white-point shift and gamma 0.95 -> output close to 0.5
    np.testing.assert_allclose(out, rgb, atol=0.1)


def test_icc_matte_warms_relative_to_glossy():
    """Matte's positive Δx (warm shift) should raise R relative to glossy."""
    rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
    matte = _apply_icc(rgb, "matte", strength=1.0)
    glossy = _apply_icc(rgb, "glossy", strength=1.0)
    assert matte[..., 0].mean() > glossy[..., 0].mean()


def test_icc_compresses_extremes():
    """Gamut compression pushes extremes toward 0.5 (more visible at strength=1)."""
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0] = 1.0  # pure red
    matte = _apply_icc(rgb, "matte", strength=1.0)
    assert matte[..., 0].mean() < 1.0  # was 1.0, now pulled down


def test_icc_output_clipped_to_unit():
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 16, 3)).astype(np.float32)
    for paper in ("matte", "glossy", "photo"):
        out = _apply_icc(rgb, paper, strength=1.0)
        assert out.shape == rgb.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_icc.py -q 2>&1 | tail -5`
Expected: collection error / `ImportError: cannot import name '_ICC_PARAMS' from 'pad_synth_face.attacks.print'`.

- [ ] **Step 3: Add the ICC helper to `print.py`**

In `pad-synth-face/src/pad_synth_face/attacks/print.py`, AFTER the halftoning helpers added in Task 1 and BEFORE the `class PrintAttack:` line, insert:

```python
# --- v2 physics: ICC profile simulation ---------------------------------------

# Per-paper-type tuple: (gamut_compression, (Δx, Δy) white-point shift, tone_gamma).
# Parameters per spec §4.1; references: Lukac & Plataniotis (eds.), 2007;
# Marini & Rizzi 2000 (white-point handling).
_ICC_PARAMS: dict[str, tuple[float, tuple[float, float], float]] = {
    "matte":  (0.12, (+0.012, +0.008), 1.10),
    "glossy": (0.05, (+0.002, +0.001), 0.95),
    "photo":  (0.03, (-0.003, -0.002), 0.92),
}


def _apply_icc(rgb: np.ndarray, paper_type: str, strength: float) -> np.ndarray:
    """sRGB-space parameterized print-profile transform.

    rgb: float [0,1] (H,W,3); paper_type in {matte, glossy, photo};
    strength scales the gamut-compression effect [0,1]. Returns float [0,1].
    """
    gamut, (dx, dy), gamma = _ICC_PARAMS[paper_type]
    out = rgb.astype(np.float32, copy=True)

    # 1. Gamut compression: pull every pixel toward middle gray by c.
    c = float(gamut) * float(strength)
    out = (1.0 - c) * out + c * 0.5

    # 2. White-point shift: chromaticity-to-RGB approximation (clipped later).
    out[..., 0] += float(dx) * 0.5
    out[..., 1] += float(dy) * 0.5
    out[..., 2] -= (float(dx) + float(dy)) * 0.25

    # 3. Tone curve: out := out ** gamma.
    out = np.clip(out, 0.0, 1.0) ** float(gamma)

    return out.astype(np.float32)
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_icc.py -q 2>&1 | tail -5`
Expected: 5 passed.

- [ ] **Step 5: Full suite green (still NO wiring; helpers added but unused)**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 148 passed, 1 skipped, 4 warnings (143 prior + 5 new).

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/print.py pad-synth-face/tests/test_print_icc.py
git commit -m "feat(pad-face): ICC parameterized print-profile helper"
```

---

## Task 3: Ontology v2 (version bump + new axis)

**Files:**
- Modify: `ontology/face/print.yaml` (bump version; append `icc_profile_strength` axis at the end)

- [ ] **Step 1: Confirm pre-existing tests cover ontology loading**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_ontology.py -q 2>&1 | tail -3`
Expected: all pass. (No new test needed for the bump — the existing ontology loader test validates loading + lint; the new axis just exercises the same code path.)

- [ ] **Step 2: Update the ontology file**

In `ontology/face/print.yaml`:

a) Change the `version` line from:
```yaml
version: "2026-05-11"
```
to:
```yaml
version: "2026-05-22"
```

b) Append to the END of the `axes:` block (after the existing `cutout` axis):
```yaml
  icc_profile_strength:
    type: uniform
    low: 0.5
    high: 1.0
    provenance:
      paper: "Lukac & Plataniotis (eds.), Color Image Processing: Methods and Applications, CRC Press 2007"
      doi: null
      url: null
```

c) Also remove the comment line above `print_dpi` that says "informational only — recorded in attack_params but not consumed by Phase 1 simulate()" — once Task 4 wires it, `print_dpi` becomes active. (Keep `holder_present`'s "informational only" comment — that axis stays informational in this scope.)

- [ ] **Step 3: Run ontology tests**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_ontology.py -q 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 4: Sanity-check the new param appears in samples**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
import numpy as np
from pad_synth_core.ontology import load_ontology
ont = load_ontology(Path('ontology/face/print.yaml'))
print('version:', ont.version)
print('params:', list(ont.sample_params(np.random.default_rng(1)).keys()))
"
```
Expected: `version: 2026-05-22`; params list includes `icc_profile_strength`. The other 5 axes are present and unchanged in order.

- [ ] **Step 5: Full suite WILL FAIL on the golden test (expected — fixed in Task 5)**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -5`
Expected: the determinism golden test fails because the new axis adds an RNG draw, shifting subsequent random state for downstream attack sampling. **DO NOT regenerate the golden yet** — Task 4 must wire the new physics first; otherwise the golden would reflect "ontology change but no physics change," which would obscure the v2 baseline. Treat this failure as expected here; proceed to Task 4.

- [ ] **Step 6: Commit**

```bash
git add ontology/face/print.yaml
git commit -m "feat(pad-face): bump print ontology to 2026-05-22 + icc_profile_strength axis"
```

(Note: at this commit the suite has 1 failing test — the determinism golden — which is expected and fixed in Task 5 after physics wiring. The two intermediate commits T3 and T4 represent an atomic ontology+physics change; T5 regenerates the golden as the canonical signal.)

---

## Task 4: Wire halftoning + ICC into `PrintAttack.simulate`

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py` (modify the `simulate` method body)
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py` (update the module docstring to reflect v2)
- Create: `pad-synth-face/tests/test_print_v2_integration.py`

- [ ] **Step 1: Write the failing integration test**

`pad-synth-face/tests/test_print_v2_integration.py`:
```python
from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.print import PrintAttack

REPO = Path(__file__).resolve().parents[2]


def _attack() -> PrintAttack:
    return PrintAttack(load_ontology(REPO / "ontology" / "face" / "print.yaml"))


def test_simulate_returns_correct_shape_and_dtype():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = _attack()
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.shape == bonafide.shape
    assert out.dtype == np.uint8


def test_simulate_uses_icc_profile_strength_param():
    """params dict must contain the new icc_profile_strength axis."""
    attack = _attack()
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    assert "icc_profile_strength" in params
    assert 0.5 <= params["icc_profile_strength"] <= 1.0


def test_simulate_low_dpi_has_more_dot_structure_than_high_dpi():
    """Two attacks differing only in print_dpi yield outputs with different
    high-frequency dot structure (low-DPI -> more coarse transitions)."""
    bonafide = np.full((64, 64, 3), 200, dtype=np.uint8)
    attack = _attack()
    # Identical other params; only print_dpi differs.
    base = {
        "paper_type": "matte",
        "tilt_degrees": 0.0,
        "holder_present": False,
        "cutout": "none",
        "icc_profile_strength": 0.75,
    }
    out_low = attack.simulate(bonafide, {**base, "print_dpi": 150}, sample_rng(7))
    out_high = attack.simulate(bonafide, {**base, "print_dpi": 1200}, sample_rng(7))

    def transitions(img: np.ndarray) -> int:
        g = (img[:, :, 1] > 127).astype(np.int32)
        return int(np.abs(np.diff(g, axis=1)).sum())

    assert transitions(out_low) < transitions(out_high)


def test_simulate_deterministic_under_same_seed():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = _attack()

    rng1 = sample_rng(99)
    p1 = attack.sample_params(rng1)
    o1 = attack.simulate(bonafide, p1, rng1)

    rng2 = sample_rng(99)
    p2 = attack.sample_params(rng2)
    o2 = attack.simulate(bonafide, p2, rng2)

    assert p1 == p2
    assert np.array_equal(o1, o2)


def test_simulate_actually_modifies_the_image():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = _attack()
    rng = sample_rng(2)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert not np.array_equal(out, bonafide)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_v2_integration.py -q 2>&1 | tail -8`
Expected: the `test_simulate_low_dpi_has_more_dot_structure_than_high_dpi` test fails (v1 ignores `print_dpi`, so the output transition count is identical regardless of DPI). Other tests likely pass already since v1 simulate is deterministic and shape-preserving; the new-axis-in-params test passes by virtue of Task 3.

- [ ] **Step 3: Update `PrintAttack.simulate` to apply halftone + ICC**

In `pad-synth-face/src/pad_synth_face/attacks/print.py`, replace the `simulate` method body (currently lines 84–101) with:

```python
    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Linear-RGB float space for the float-domain stages.
        img = bonafide.astype(np.float32) / 255.0

        # Paper-color tint (matte/glossy/photo).
        tint = np.array(_PAPER_TINTS[params["paper_type"]], dtype=np.float32)
        img = img * tint

        # v2: halftone (driven by print_dpi).
        img = _apply_halftone(img, params["print_dpi"])

        # v2: ICC profile (keyed by paper_type, scaled by strength).
        img = _apply_icc(
            img,
            params["paper_type"],
            float(params["icc_profile_strength"]),
        )

        # Paper-texture multiplicative noise (RNG used here).
        texture = _paper_texture(img.shape[0], img.shape[1], rng)
        img = img * texture

        # Back to uint8 for the spatial-domain stages.
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = _perspective_warp(img, params["tilt_degrees"], rng)
        img = _apply_cutout(img, params["cutout"])
        return img
```

Also update the module docstring at the top of the file. Replace lines 1–12 (the existing docstring) with:

```python
"""Phase 2 print-attack simulator (v2 physics).

Pipeline:
  1. Paper-color tint (matte/glossy/photo per ontology)
  2. Halftone — per-channel AM dot screening at standard rosette angles
     (C=15°, M=75°, Y=0°, K=45°); dot-cell frequency driven by print_dpi.
  3. ICC profile transform — gamut compression + white-point shift +
     tone gamma, parameterized per paper_type and scaled by
     icc_profile_strength.
  4. Paper-texture multiplicative noise (uses RNG).
  5. Perspective warp simulating a tilted printed page (uses RNG).
  6. Optional cutout (eyes / eyes+mouth).

Anisotropic specular highlights remain explicitly deferred to a follow-up.
The v1 single-tier-physics version is captured by ontology_version
2026-05-11; this module corresponds to ontology_version 2026-05-22.
"""
```

- [ ] **Step 4: Run the v2 integration tests**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_v2_integration.py -q 2>&1 | tail -5`
Expected: 5 passed.

- [ ] **Step 5: Run the existing print-attack tests — they must still pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_attack.py -q 2>&1 | tail -3`
Expected: 3 passed. (The existing tests don't pin specific pixel values — only shape, "image modified," and determinism — so v2 satisfies all three.)

- [ ] **Step 6: Confirm the golden still fails (next task fixes it)**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 failed (the v2 physics produces different hashes than v1 — this is the deliberate physics change; Task 5 regenerates).

- [ ] **Step 7: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/print.py pad-synth-face/tests/test_print_v2_integration.py
git commit -m "feat(pad-face): wire halftone + ICC into PrintAttack.simulate (v2 physics)"
```

---

## Task 5: Regenerate the determinism golden

**Files:**
- Modify: `tests/golden/golden_hashes.json` (regenerated by the test's update mode)

- [ ] **Step 1: Verify the golden currently fails (the v2 baseline check)**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 failed with "Determinism regression. If intentional, run PAD_SYNTH_UPDATE_GOLDEN=1 pytest tests/test_determinism_golden.py".

- [ ] **Step 2: Regenerate the golden**

Run: `PAD_SYNTH_UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 passed (the test takes the update branch: writes the new hashes to disk and returns).

- [ ] **Step 3: Verify the golden now passes on a fresh run**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 4: Spot-check the regenerated file**

Run: `python3 -c "import json; d=json.load(open('tests/golden/golden_hashes.json')); print('entries:', len(d)); print('first 3:', list(d.items())[:3])"`
Expected: 32 entries; the 16 `face-bonafide-*` hashes match the prior file (bonafide ingestion didn't change); the 8 `face-print-*` and 8 `face-replay-*` hashes are **different** from the prior file (print hashes change because v2 physics, replay hashes change because the new ontology axis shifts the per-sample RNG state downstream).

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 153 passed, 1 skipped, 4 warnings (148 prior + 5 new from Task 4's integration test).

- [ ] **Step 6: Commit**

```bash
git add tests/golden/golden_hashes.json
git commit -m "fix(pad-face): regenerate determinism golden for v2 print physics"
```

---

## Task 6: Six v2 measurement configs

**Files:**
- Create: `configs/runs/v2_seta_d1.yaml`, `v2_seta_d2.yaml`, `v2_seta_d3.yaml`
- Create: `configs/runs/v2_setb_d1.yaml`, `v2_setb_d2.yaml`, `v2_setb_d3.yaml`
- Create: `pad-synth-face/tests/test_v2_configs.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_v2_configs.py`:
```python
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "configs" / "runs"

EXPECTED = {
    "v2_seta_d1.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 6),
    "v2_seta_d2.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 32),
    "v2_seta_d3.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 256),
    "v2_setb_d1.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 4),
    "v2_setb_d2.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 32),
    "v2_setb_d3.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 256),
}


def test_v2_configs_present_and_well_formed():
    for fname, (seed, sensor, fixture, spb) in EXPECTED.items():
        cfg = yaml.safe_load((CFG_DIR / fname).read_text())
        assert cfg["run"]["seed"] == seed, fname
        assert cfg["run"]["deterministic"] is True, fname
        assert cfg["run"]["output"] == f"./datasets/{Path(fname).stem}", fname
        assert cfg["modality"] == "face", fname
        assert cfg["sensor_preset"] == sensor, fname
        assert cfg["bonafide"]["root"] == fixture, fname
        assert cfg["bonafide"]["samples_per_bonafide"] == spb, fname
        assert set(cfg["attacks"].keys()) == {"print", "replay"}, fname
        assert cfg["attacks"]["print"]["weight"] == 1.0, fname
        assert cfg["attacks"]["replay"]["weight"] == 1.0, fname
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_v2_configs.py -q 2>&1 | tail -3`
Expected: 1 failed (file not found).

- [ ] **Step 3: Create all six configs**

`configs/runs/v2_seta_d1.yaml`:
```yaml
run:
  name: v2_seta_d1
  output: ./datasets/v2_seta_d1
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_fixtures/digiface
  samples_per_bonafide: 6
  splits:
    train: 0.0
    dev: 0.0
    test: 1.0

attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml

sensor_preset: mobile-front-2024
```

`configs/runs/v2_seta_d2.yaml`: identical to `v2_seta_d1.yaml` except `name: v2_seta_d2`, `output: ./datasets/v2_seta_d2`, `samples_per_bonafide: 32`.

`configs/runs/v2_seta_d3.yaml`: identical to `v2_seta_d1.yaml` except `name: v2_seta_d3`, `output: ./datasets/v2_seta_d3`, `samples_per_bonafide: 256`.

`configs/runs/v2_setb_d1.yaml`:
```yaml
run:
  name: v2_setb_d1
  output: ./datasets/v2_setb_d1
  seed: 20260523
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_fixtures/extended_fixture
  samples_per_bonafide: 4
  splits:
    train: 0.0
    dev: 0.0
    test: 1.0

attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml

sensor_preset: webcam-1080p
```

`configs/runs/v2_setb_d2.yaml`: identical to `v2_setb_d1.yaml` except `name: v2_setb_d2`, `output: ./datasets/v2_setb_d2`, `samples_per_bonafide: 32`.

`configs/runs/v2_setb_d3.yaml`: identical to `v2_setb_d1.yaml` except `name: v2_setb_d3`, `output: ./datasets/v2_setb_d3`, `samples_per_bonafide: 256`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_v2_configs.py -q 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 154 passed, 1 skipped, 4 warnings (153 prior + 1 new).

- [ ] **Step 6: Commit**

```bash
git add configs/runs/v2_seta_d1.yaml configs/runs/v2_seta_d2.yaml configs/runs/v2_seta_d3.yaml configs/runs/v2_setb_d1.yaml configs/runs/v2_setb_d2.yaml configs/runs/v2_setb_d3.yaml pad-synth-face/tests/test_v2_configs.py
git commit -m "feat(pad-spark): six v2 measurement configs (D1/D2/D3 on v2 physics)"
```

---

## Task 7: Generate v2 datasets locally

**Files:** none (output to gitignored `datasets/`).

- [ ] **Step 1: Generate all six**

```bash
cd /Users/stuartwells/test
for f in v2_seta_d1 v2_seta_d2 v2_seta_d3 v2_setb_d1 v2_setb_d2 v2_setb_d3; do
  echo "=== generating $f ==="
  .venv/bin/python -m pad_synth_face.cli generate --config configs/runs/${f}.yaml | tail -3
done
```

Expected: each prints a JSON summary with `"failed": 0`. Wall-time: ~10–15 minutes for the v2 D3 sets (4096+8192 samples × halftoning overhead).

- [ ] **Step 2: Verify counts**

```bash
for d in datasets/v2_seta_d{1,2,3} datasets/v2_setb_d{1,2,3}; do
  n=$(wc -l < "$d/manifest.jsonl")
  bona=$(grep -c '"label":"bonafide"' "$d/manifest.jsonl")
  attack=$(grep -c '"label":"attack"' "$d/manifest.jsonl")
  printf "%-22s total=%5d  bonafide=%5d  attack=%5d\n" "$d" "$n" "$bona" "$attack"
done
```

Expected (exact):
```
datasets/v2_seta_d1     total=   96  bonafide=   48  attack=   48
datasets/v2_seta_d2     total=  512  bonafide=  256  attack=  256
datasets/v2_seta_d3     total= 4096  bonafide= 2048  attack= 2048
datasets/v2_setb_d1     total=  128  bonafide=   64  attack=   64
datasets/v2_setb_d2     total= 1024  bonafide=  512  attack=  512
datasets/v2_setb_d3     total= 8192  bonafide= 4096  attack= 4096
```

- [ ] **Step 3: Confirm ontology version recorded in manifest**

```bash
python3 -c "
import json
with open('datasets/v2_seta_d1/manifest.jsonl') as fh:
    rec = json.loads(fh.readline())
print('ontology_version:', rec.get('ontology_version'))
"
```
Expected: `ontology_version: 2026-05-22` (the v2 version stamp).

- [ ] **Step 4: No commit (gitignored)**

---

## Task 8: rsync to Spark + run 27-cell v2 sweep

**Files:** none (operations on a remote host).

- [ ] **Step 1: Sync the latest code to the Spark**

```bash
rsync -a --delete \
  --exclude='.venv' --exclude='__pycache__' --exclude='datasets' \
  --exclude='.superpowers' --exclude='.git/objects/pack' \
  /Users/stuartwells/test/ \
  swells@spark-50d2.local:~/ml/projects/pad-spark/
```

- [ ] **Step 2: rsync the six v2 datasets**

```bash
for d in v2_seta_d1 v2_seta_d2 v2_seta_d3 v2_setb_d1 v2_setb_d2 v2_setb_d3; do
  rsync -a --partial \
    "/Users/stuartwells/test/datasets/${d}/" \
    "swells@spark-50d2.local:~/ml/datasets/${d}/"
done
```

- [ ] **Step 3: Run the 27-cell v2 sweep**

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)_v2
echo "$ts" > /tmp/padspark_v2_ts
ssh swells@spark-50d2.local "cd ~/ml/projects/pad-spark && .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 ~/ml/datasets/v2_seta_d1 --set-b-d1 ~/ml/datasets/v2_setb_d1 \
  --set-a-d2 ~/ml/datasets/v2_seta_d2 --set-b-d2 ~/ml/datasets/v2_setb_d2 \
  --set-a-d3 ~/ml/datasets/v2_seta_d3 --set-b-d3 ~/ml/datasets/v2_setb_d3 \
  --set-a-d4 ~/ml/datasets/spark_seta_d4 --set-b-d4 ~/ml/datasets/spark_setb_d4 \
  --output-dir ~/ml/logs/pad-spark/${ts} \
  --device cuda --epochs 10 --batch-size 32 \
  --cells L1:D1:0,L1:D1:1,L1:D1:2,L1:D2:0,L1:D2:1,L1:D2:2,L1:D3:0,L1:D3:1,L1:D3:2,L2:D1:0,L2:D1:1,L2:D1:2,L2:D2:0,L2:D2:1,L2:D2:2,L2:D3:0,L2:D3:1,L2:D3:2,L3:D1:0,L3:D1:1,L3:D1:2,L3:D2:0,L3:D2:1,L3:D2:2,L3:D3:0,L3:D3:1,L3:D3:2" 2>&1 | tail -30
```

(D4 args are required by the script even though `--cells` doesn't include D4 cells — pass the existing v1 D4 datasets as placeholders; they won't be touched.)

Expected: 27 lines `L? D? seed=?  eer_in=0.??  eer_cross=0.??  ??.?s`. Total wall-time ~5 minutes on the GB10.

- [ ] **Step 4: Confirm 27 JSONs**

```bash
ssh swells@spark-50d2.local "ls ~/ml/logs/pad-spark/$(cat /tmp/padspark_v2_ts)/runs/ | wc -l"
```
Expected: `27`.

- [ ] **Step 5: No commit**

---

## Task 9: rsync results back, author v1-vs-v2 report append, commit

**Files:**
- Add: 27 JSONs at `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v2/L{1,2,3}_D{1,2,3}_{0,1,2}.json`
- Add: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_v2.csv`
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (append v2 section)
- Modify: `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` (one-line append)

- [ ] **Step 1: rsync the 27 v2 JSONs into a new subdir**

```bash
ts=$(cat /tmp/padspark_v2_ts)
mkdir -p docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v2
rsync -av "swells@spark-50d2.local:~/ml/logs/pad-spark/${ts}/runs/" \
  docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v2/
ls docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v2/ | wc -l
```
Expected: 27.

- [ ] **Step 2: Build `summary_v2.csv` from the v2 JSONs**

```bash
.venv/bin/python - <<'PY'
import csv, json
from pathlib import Path
runs_dir = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v2")
out = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_v2.csv")
rows = []
for p in sorted(runs_dir.glob("*.json")):
    r = json.loads(p.read_text())
    rows.append([r["capacity"], r["data_level"], r["seed"],
                 r["eer_in_domain"], r["eer_cross_domain"],
                 f"{r['train_seconds']:.2f}"])
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["capacity", "data_level", "seed",
                "eer_in_domain", "eer_cross_domain", "train_seconds"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
PY
```
Expected: `wrote 27 rows`.

- [ ] **Step 3: Compute v1-vs-v2 cell aggregates**

```bash
.venv/bin/python - <<'PY'
import json, statistics as st
from pathlib import Path
report_dir = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results")

def load(subdir, level_filter=None):
    rows = []
    for p in sorted((report_dir / subdir).glob("*.json")):
        r = json.loads(p.read_text())
        if level_filter and r["data_level"] not in level_filter:
            continue
        rows.append(r)
    cells = {}
    for r in rows:
        cells.setdefault((r["capacity"], r["data_level"]), []).append(r)
    return cells

v1 = load("runs", level_filter={"D1", "D2", "D3"})
v2 = load("runs_v2")
print(f"{'cell':<8} {'v1 cross (mean+-std)':<22}  {'v2 cross (mean+-std)':<22}  {'delta':>8}  {'overlap':>7}")
for L in ("L1", "L2", "L3"):
    for D in ("D1", "D2", "D3"):
        v1g = [r["eer_cross_domain"] for r in v1[(L, D)]]
        v2g = [r["eer_cross_domain"] for r in v2[(L, D)]]
        m1, s1 = st.mean(v1g), st.stdev(v1g)
        m2, s2 = st.mean(v2g), st.stdev(v2g)
        delta = m1 - m2
        overlap = not (m1 + s1 < m2 - s2 or m2 + s2 < m1 - s1)
        print(f"{L} {D}    {m1:.3f} +- {s1:.3f}      {m2:.3f} +- {s2:.3f}      "
              f"{delta:+.3f}     {'yes' if overlap else 'no'}")
PY
```

Record the 9 rows. Each row is one cell's v1-vs-v2 comparison; the verdict per spec §2 is: **fires** if Δ ≥ 0.05 AND bands non-overlapping; **rises** if Δ ≤ -0.05 AND bands non-overlapping; **flat** otherwise.

- [ ] **Step 4: Append the v2 section to the existing report**

Open `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` and append at the end:

```markdown

---

## 2026-05-22 update — v2 print physics result

The print attack was upgraded to v2 physics: per-channel AM halftoning (rosette angles 15°/75°/0°/45°, dot-cell frequency driven by `print_dpi`) and a parameterized sRGB-space ICC transform keyed by `paper_type` (gamut compression + white-point shift + tone gamma) scaled by a new `icc_profile_strength` axis. Bumped ontology to `2026-05-22`. Regenerated 6 D1–D3 datasets (`datasets/v2_set{a,b}_d{1,2,3}/`) and ran a 27-cell sweep on the same GB10. Code SHA at sweep time: <fill_v2_sha>. Torch: `2.12.0.dev20260407+cu128`.

**v2 cross-domain EER (mean ± std):**

| | D1 (96/128) | D2 (512/1024) | D3 (4096/8192) |
|---|---|---|---|
| **L1 (TinyCNN)** | <fill_v2_L1_D1> | <fill_v2_L1_D2> | <fill_v2_L1_D3> |
| **L2 (SmallCNN)** | <fill_v2_L2_D1> | <fill_v2_L2_D2> | <fill_v2_L2_D3> |
| **L3 (ResNet18)** | <fill_v2_L3_D1> | <fill_v2_L3_D2> | <fill_v2_L3_D3> |

**v2 in-domain EER (mean ± std):**

| | D1 | D2 | D3 |
|---|---|---|---|
| **L1** | <fill_in_L1_D1> | <fill_in_L1_D2> | <fill_in_L1_D3> |
| **L2** | <fill_in_L2_D1> | <fill_in_L2_D2> | <fill_in_L2_D3> |
| **L3** | <fill_in_L3_D1> | <fill_in_L3_D2> | <fill_in_L3_D3> |

**v1 → v2 effect, per cell (cross-domain mean):**

| Cell | v1 mean ± std | v2 mean ± std | Δ (v1 − v2) | Bands overlap? | Verdict |
|---|---|---|---|---|---|
| L1·D1 | 0.396 ± 0.033 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L1·D2 | 0.441 ± 0.029 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L1·D3 | 0.228 ± 0.022 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L2·D1 | 0.354 ± 0.070 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L2·D2 | 0.214 ± 0.005 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L2·D3 | 0.217 ± 0.033 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L3·D1 | 0.370 ± 0.024 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L3·D2 | 0.242 ± 0.017 | <fill> | <fill> | <fill> | <fires/flat/rises> |
| L3·D3 | 0.249 ± 0.007 | <fill> | <fill> | <fill> | <fires/flat/rises> |

(Verdict rule per spec §2: **fires** if Δ ≥ 0.05 AND non-overlapping ±1σ bands; **rises** if Δ ≤ −0.05 AND non-overlapping ±1σ bands; **flat** otherwise.)

**Diagnosis:** <one paragraph naming the dominant outcome (fires / flat / rises) per axis; comment on whether L3·D3 changes direction>.

**Phase 2 recommendation update:** <one paragraph. If "fires" predominates: physics IS the lever; proceed to the mask-attack sub-project next. If "flat" predominates: physics alone insufficient at this scale; escalate to real-data integration as the dominant lever. If "rises" predominates: stop and audit the new physics; do not ship as production.>
```

Fill in every `<fill_*>` from the Step 3 output. The git SHA for `<fill_v2_sha>` is from any one v2 JSON's `git_sha` field.

- [ ] **Step 5: Append the one-line roadmap update**

In `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md`, append at the end:

```markdown

---

## 2026-05-22 update — v2 print physics sweep

Print attack upgraded to v2 (halftoning + ICC). 27-cell v1-vs-v2 sweep at D1–D3. **Verdict: <fires / flat / rises>.** See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"v2 print physics result" for the per-cell deltas and the updated Phase 2 prioritization.
```

Replace `<fires / flat / rises>` with the dominant verdict from Step 4.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v2/ \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_v2.csv \
        docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md
git commit -m "report(pad-print-v2): v1-vs-v2 sweep result — <verdict>"
```

Replace `<verdict>` with the dominant outcome (`physics fires`, `physics flat`, or `physics rises`).

- [ ] **Step 7: Final full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `154 passed, 1 skipped, 4 warnings`.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose — physics-axis intervention | Tasks 1, 2, 4 (the actual physics) + Task 8/9 (the measurement) |
| §2 measurement question + threshold rule | Task 9 Step 4 (per-cell verdict table applies the §2 rule) |
| §3 halftoning algorithm | Task 1 (`_to_cmyk`, `_inv_cmyk`, `_dot_screen`, `_halftone_channel`, `_apply_halftone`) |
| §4 ICC algorithm | Task 2 (`_ICC_PARAMS`, `_apply_icc`) |
| §5 ontology v2 bump + new axis (axis appended at end so existing samples unchanged) | Task 3 |
| §6 modify-in-place + golden regen | Tasks 4 + 5 |
| §7 measurement plan | Tasks 6, 7, 8, 9 |
| §8 architecture boundaries | All tasks honor the "modify print.py in place" rule; no new modules outside the spec |
| §9 non-goals | None violated: no specular, no mask, no real-data, no replay change, no littlecms |
| §10 success criteria | Task 5 (golden regen + full suite green), Task 9 (report committed with v1-vs-v2 verdict) |

**Placeholder scan:** Every `<fill_*>` and `<verdict>` in the plan is inside the report template that the implementer populates from real run data in Task 9 Step 3 — they are explicitly to be filled, not implementation placeholders. No "TBD/TODO/implement later" anywhere else. All code blocks complete.

**Type consistency:** `_to_cmyk(rgb) -> np.ndarray(H,W,4)`, `_inv_cmyk(cmyk) -> np.ndarray(H,W,3)`, `_dot_screen(h, w, cell_px, angle_deg)`, `_halftone_channel(channel, cell_px, angle_deg)`, `_apply_halftone(rgb, print_dpi)`, `_apply_icc(rgb, paper_type, strength)`, `_ICC_PARAMS: dict[str, tuple[float, tuple[float, float], float]]` — names consistent across Tasks 1, 2, 4. `icc_profile_strength` ontology axis name consistent across Tasks 3, 4, and the spec §5. JSON keys in the v2 sweep results unchanged from the parent project's schema.
