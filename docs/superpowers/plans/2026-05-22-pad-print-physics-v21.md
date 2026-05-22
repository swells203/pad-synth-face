# PAD Print Physics v2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-channel per-sample jitter to the v2 halftoning algorithm — sub-pixel offset, angle (σ=3°), cell-size (±10%) — to break the deterministic-halftone watermark that produced the v2 sweep's 6/9 cells of 0.000 cross-domain EER, then re-measure with a 27-cell Spark sweep and decide whether physics is genuinely the lever.

**Architecture:** Extend three internal helpers in `pad-synth-face/src/pad_synth_face/attacks/print.py` (`_apply_halftone`, `_dot_screen`, `_halftone_channel`) with optional jitter parameters. When `_apply_halftone` receives an RNG, each channel samples its own `(cell_px multiplier k, angle jitter Δθ, sub-pixel offset dx, dy)` per sample. When called with no RNG, the deterministic v2 behavior is preserved (so existing v2 unit tests cover the no-jitter path unchanged). `PrintAttack.simulate` passes its existing per-sample RNG through (one-line change). Ontology version bumps to `2026-05-23`; golden regenerated. Same 27-cell measurement framework as v2.

**Tech Stack:** Same as the parent v2 project — Python 3.11+ (laptop) / 3.12 (Spark), numpy, opencv (existing), PyTorch nightly cu128 (Spark), pytest. No new external dependencies.

---

## Reference: facts the engineer needs

**Current state (verified, on main).** Branch `main` has v2 merged at `7de325e` (no, that was D4) and v2 print physics at the latest merge. The `print.py` file already contains `_HALFTONE_ANGLES_DEG`, `_to_cmyk`, `_inv_cmyk`, `_dot_screen(h, w, cell_px, angle_deg)`, `_halftone_channel(channel, cell_px, angle_deg)`, `_apply_halftone(rgb, print_dpi)`, `_ICC_PARAMS`, `_apply_icc(rgb, paper_type, strength)`. The `PrintAttack.simulate` body calls `_apply_halftone(img, params["print_dpi"])` (no rng). Existing 5-test files: `test_print_halftone.py` (CMYK + DPI scaling + determinism + shape, all called WITHOUT rng so they exercise the no-jitter path), `test_print_icc.py` (5 tests), `test_print_v2_integration.py` (5 tests including the DPI-discriminator test that DOES go through `simulate` → `_apply_halftone` with rng once wired).

**v2 sweep result already committed.** `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` has the v2 section (artifact diagnosis). v1 and v2 result JSONs are in `runs/` and `runs_v2/` respectively. v2.1 JSONs go to a new `runs_v21/` sibling subdir.

**Existing v1 means** (for the three-way comparison table in T8): see the v1 column in the v2 report's "v1 → v2 effect" table.

**Existing v2 means** (mostly 0.000): same source.

**Spec.** `docs/superpowers/specs/2026-05-22-pad-print-physics-v21-design.md`. §3 jitter table; §6 determinism; §7 measurement plan; §10 success criteria including "no v2.1 cross-domain cell ≤ 0.001 EER" and "at least one D3 cell shows v2.1 ≤ v1 − 0.05 with non-overlapping bands."

**Per-channel jitter draw order** (locked for determinism): for each channel `c ∈ {C, M, Y, K}` in that order, the rng is consumed in this exact sequence:
1. `k = rng.uniform(0.90, 1.10)` (cell-size multiplier)
2. `Δθ = rng.normal(0.0, 3.0)` (angle jitter, degrees)
3. `dx = rng.uniform(-cell_px/2, +cell_px/2)` (where `cell_px = max(2.0, base_cell * k)`)
4. `dy = rng.uniform(-cell_px/2, +cell_px/2)`

Total: 4 draws × 4 channels = 16 per `_apply_halftone` call. Order matters for byte-identical reproducibility across the determinism golden.

---

## Task 1: Halftone jitter helpers + unit tests

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py` (extend `_dot_screen`, `_halftone_channel`, `_apply_halftone` signatures + bodies; do NOT yet wire into `simulate`)
- Create: `pad-synth-face/tests/test_print_halftone_jitter.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_print_halftone_jitter.py`:
```python
import numpy as np

from pad_synth_face.attacks.print import _apply_halftone


def test_jitter_different_rng_states_produce_different_outputs():
    """The load-bearing invariant: two different rngs -> two different halftone
    outputs. Without this, the watermark survives the v2.1 work."""
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    out_a = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(1))
    out_b = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(2))
    assert not np.array_equal(out_a, out_b)


def test_jitter_same_rng_state_produces_identical_output():
    """Determinism: same rng seed -> byte-identical output (pipeline invariant)."""
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    out_a = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(7))
    out_b = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(7))
    assert np.array_equal(out_a, out_b)


def test_no_rng_preserves_deterministic_v2_path():
    """When rng=None, behavior is byte-identical to v2 (no jitter, deterministic
    screen). This keeps the existing v2 unit tests passing unchanged."""
    rgb = np.full((32, 32, 3), 0.4, dtype=np.float32)
    a = _apply_halftone(rgb, print_dpi=300)
    b = _apply_halftone(rgb, print_dpi=300)
    assert np.array_equal(a, b)


def test_jitter_preserves_shape_dtype_and_range():
    rgb = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    out = _apply_halftone(rgb, print_dpi=300, rng=np.random.default_rng(11))
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_jitter_consumes_rng_state_across_calls():
    """Two consecutive halftone calls using the SAME rng object yield different
    outputs (proves the rng is being advanced by ~16 draws per call)."""
    rgb = np.full((32, 32, 3), 0.5, dtype=np.float32)
    rng = np.random.default_rng(42)
    out1 = _apply_halftone(rgb, print_dpi=300, rng=rng)
    out2 = _apply_halftone(rgb, print_dpi=300, rng=rng)
    assert not np.array_equal(out1, out2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/stuartwells/test && .venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone_jitter.py -q 2>&1 | tail -8`
Expected: at minimum `test_jitter_different_rng_states_produce_different_outputs` and `test_jitter_consumes_rng_state_across_calls` fail with `TypeError: _apply_halftone() got an unexpected keyword argument 'rng'`.

- [ ] **Step 3: Extend the three helpers**

In `pad-synth-face/src/pad_synth_face/attacks/print.py`, replace the existing `_dot_screen`, `_halftone_channel`, and `_apply_halftone` functions with these versions (do NOT touch any other helper, the `PrintAttack` class, or the module docstring in this task):

```python
def _dot_screen(
    h: int,
    w: int,
    cell_px: float,
    angle_deg: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """2D rotated cosine dot screen, range [0,1]. Deterministic — no RNG.
    Optional (dx, dy) shift the screen origin by sub-pixel amounts."""
    theta = np.deg2rad(angle_deg)
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xv - float(dx)
    ys = yv - float(dy)
    xp = xs * np.cos(theta) + ys * np.sin(theta)
    yp = -xs * np.sin(theta) + ys * np.cos(theta)
    grid = np.cos(2.0 * np.pi * xp / cell_px) * np.cos(2.0 * np.pi * yp / cell_px)
    return (0.5 + 0.5 * grid).astype(np.float32)


def _halftone_channel(
    channel: np.ndarray,
    cell_px: float,
    angle_deg: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Binary halftone: pixel ON where channel value > screen threshold."""
    screen = _dot_screen(channel.shape[0], channel.shape[1], cell_px, angle_deg, dx, dy)
    return (channel > screen).astype(np.float32)


def _apply_halftone(
    rgb: np.ndarray,
    print_dpi: int | float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-channel AM halftoning at the standard rosette angles.

    rgb: float [0,1] (H,W,3); print_dpi drives the base dot-cell frequency.

    When rng is None (v2 behavior): deterministic screen, no jitter.
    When rng is provided (v2.1 behavior): per-channel per-sample jitter from
    the spec — cell-size ~U(0.90, 1.10), angle ~N(0, 3°), sub-pixel offset
    (dx, dy) ~U(-cell/2, +cell/2). The draw order per channel C/M/Y/K is
    (k, Δθ, dx, dy) — see spec §3 and the plan reference block.

    Returns float [0,1] (H,W,3).
    """
    base_cell = max(2.0, round(8.0 * 150.0 / float(print_dpi)))
    cmyk = _to_cmyk(rgb)
    out = np.empty_like(cmyk)
    for i, base_angle in enumerate(_HALFTONE_ANGLES_DEG):
        if rng is None:
            cell_px, angle, dx, dy = base_cell, base_angle, 0.0, 0.0
        else:
            k = float(rng.uniform(0.90, 1.10))
            cell_px = max(2.0, base_cell * k)
            angle = base_angle + float(rng.normal(0.0, 3.0))
            dx = float(rng.uniform(-cell_px / 2.0, cell_px / 2.0))
            dy = float(rng.uniform(-cell_px / 2.0, cell_px / 2.0))
        out[..., i] = _halftone_channel(cmyk[..., i], cell_px, angle, dx, dy)
    return _inv_cmyk(out)
```

- [ ] **Step 4: Run the new jitter tests to verify they pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone_jitter.py -q 2>&1 | tail -5`
Expected: 5 passed.

- [ ] **Step 5: Run the EXISTING v2 halftone unit tests — they must still pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone.py -q 2>&1 | tail -5`
Expected: 5 passed. (These tests call `_apply_halftone(rgb, print_dpi)` without rng, hitting the deterministic v2 path unchanged.)

- [ ] **Step 6: Confirm `PrintAttack.simulate` still works (it calls `_apply_halftone` without rng — that's T2's wiring)**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_attack.py pad-synth-face/tests/test_print_v2_integration.py -q 2>&1 | tail -5`
Expected: 8 passed (3 + 5 — the integration tests still pass because `simulate` hasn't been updated yet; `_apply_halftone` is called without rng so the deterministic path runs).

- [ ] **Step 7: Confirm the golden still passes (no behavioral change yet)**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 passed. (Crucial: the no-rng path is byte-identical to v2; the golden does NOT need regeneration in this task. It'll need regeneration in T4 after T2 wires the jitter through `simulate`.)

- [ ] **Step 8: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/print.py pad-synth-face/tests/test_print_halftone_jitter.py
git commit -m "feat(pad-face): halftone jitter (optional rng-driven cell/angle/offset)"
```

---

## Task 2: Wire rng into `PrintAttack.simulate`

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/print.py` (one-line change in `simulate`; update top-of-file docstring to mention v2.1)

- [ ] **Step 1: Update `simulate` body**

In `pad-synth-face/src/pad_synth_face/attacks/print.py`, locate the `PrintAttack.simulate` method. Find the line:
```python
        # v2: halftone (driven by print_dpi).
        img = _apply_halftone(img, params["print_dpi"])
```
Replace with:
```python
        # v2.1: halftone with per-sample jitter (driven by print_dpi + rng).
        img = _apply_halftone(img, params["print_dpi"], rng)
```

Also update the top-of-file docstring. Find the line in the existing docstring that says:
```
  2. Halftone — per-channel AM dot screening at standard rosette angles
     (C=15°, M=75°, Y=0°, K=45°); dot-cell frequency driven by print_dpi.
```
Replace with:
```
  2. Halftone — per-channel AM dot screening at standard rosette angles
     (C=15°, M=75°, Y=0°, K=45°); dot-cell frequency driven by print_dpi
     with per-sample jitter on cell-size (±10%), angle (σ=3°), and
     sub-pixel offset to break deterministic-pattern artifacts (v2.1).
```

And in the same docstring's closing paragraph, append a sentence about v2.1 ontology version. Find:
```
The v1 single-tier-physics version is captured by ontology_version
2026-05-11; this module corresponds to ontology_version 2026-05-22.
```
Replace with:
```
The v1 single-tier-physics version is captured by ontology_version
2026-05-11; the v2 deterministic-halftone version by 2026-05-22; this
module (v2.1, jittered halftone) corresponds to ontology_version
2026-05-23.
```

- [ ] **Step 2: Run the v2 integration tests — they should still pass**

The 5 tests in `test_print_v2_integration.py` cover shape, params, DPI-discriminator, determinism, and image-modification. Determinism (same seed → same output) still holds because the per-sample seed produces the same rng draws. The DPI-discriminator test passes a seeded `sample_rng(7)` so it's stable.

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_v2_integration.py pad-synth-face/tests/test_print_attack.py -q 2>&1 | tail -5`
Expected: 8 passed (5 + 3). If `test_simulate_low_dpi_has_more_dot_structure_than_high_dpi` becomes flaky (the seeded jitter happens to invert the typical ordering at this particular seed), STOP and report DONE_WITH_CONCERNS — do not weaken the assertion; fix in a follow-up.

- [ ] **Step 3: Confirm the v2 halftone unit tests still pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_print_halftone.py -q 2>&1 | tail -3`
Expected: 5 passed. (Those tests call `_apply_halftone` directly without rng — unchanged path.)

- [ ] **Step 4: Confirm the golden NOW fails (because `simulate` is now jittered)**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 failed. The new jitter inside `simulate` changes the print samples' bytes. This failure is INTENTIONAL — T4 regenerates the golden after T3 bumps the ontology version.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/print.py
git commit -m "feat(pad-face): wire rng into PrintAttack.simulate halftone call (v2.1)"
```

(Note: at this commit the suite has 1 failing test — the determinism golden — which is expected and fixed in T4. T2 and T3 represent an atomic physics+ontology change that T4 then stamps as canonical.)

---

## Task 3: Ontology version bump to 2026-05-23

**Files:**
- Modify: `ontology/face/print.yaml` (version only)

- [ ] **Step 1: Update the version line**

In `ontology/face/print.yaml`, change:
```yaml
version: "2026-05-22"
```
to:
```yaml
version: "2026-05-23"
```

(No axes added or removed. The jitter is algorithm-level, not ontology-driven — per spec §5.)

- [ ] **Step 2: Run ontology tests**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_ontology.py -q 2>&1 | tail -3`
Expected: all pass (just a version-string change; the loader doesn't care about the value).

- [ ] **Step 3: Sanity-check the new version is exposed**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
from pad_synth_core.ontology import load_ontology
ont = load_ontology(Path('ontology/face/print.yaml'))
print('version:', ont.version)
print('axes:', list(ont.axes.keys()) if hasattr(ont, 'axes') else 'no axes attr')
"
```
Expected: `version: 2026-05-23`; axes list unchanged from v2 (same 6 axes).

- [ ] **Step 4: Golden still fails (expected — fixed in T4)**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 failed. (The combined v2.1 physics + ontology version change.)

- [ ] **Step 5: Commit**

```bash
git add ontology/face/print.yaml
git commit -m "feat(pad-face): bump print ontology to 2026-05-23 (v2.1 physics revision)"
```

---

## Task 4: Regenerate the determinism golden

**Files:**
- Modify: `tests/golden/golden_hashes.json` (regenerated)

- [ ] **Step 1: Verify the golden currently fails**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 failed (the deliberate intermediate state).

- [ ] **Step 2: Regenerate the golden**

Run: `PAD_SYNTH_UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 passed (the update branch ran).

- [ ] **Step 3: Verify the golden now passes on a fresh run**

Run: `.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 4: Spot-check the regenerated file**

Run:
```bash
python3 -c "
import json
d = json.load(open('tests/golden/golden_hashes.json'))
print('entries:', len(d))
print('bonafide count:', sum(1 for k in d if 'bonafide' in k))
print('print count:', sum(1 for k in d if 'print' in k))
print('replay count:', sum(1 for k in d if 'replay' in k))
"
```
Expected: 32 entries; 16 bonafide; 8 print; 8 replay. Same shape as before. The 4 `face-print-*` hashes will differ from the v2 file (because jitter changed bytes); the 16 bonafide and 8 replay hashes should be **unchanged** from v2 (per-attack RNG isolation — the new ontology axis is still absent here, only the halftone algorithm changed for print attacks).

Verify the isolation: `git diff HEAD~1 tests/golden/golden_hashes.json | grep -E "^[+-]" | grep -v "^[+-]{3}" | wc -l` — expect `8` (4 deletions + 4 additions, all `face-print-*`).

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `159 passed, 1 skipped, 4 warnings` (154 prior + 5 new jitter unit tests from T1).

- [ ] **Step 6: Commit**

```bash
git add tests/golden/golden_hashes.json
git commit -m "fix(pad-face): regenerate determinism golden for v2.1 jittered halftone"
```

---

## Task 5: Six v2.1 measurement configs

**Files:**
- Create: `configs/runs/v21_seta_d1.yaml`, `v21_seta_d2.yaml`, `v21_seta_d3.yaml`
- Create: `configs/runs/v21_setb_d1.yaml`, `v21_setb_d2.yaml`, `v21_setb_d3.yaml`
- Create: `pad-synth-face/tests/test_v21_configs.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_v21_configs.py`:
```python
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "configs" / "runs"

EXPECTED = {
    "v21_seta_d1.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 6),
    "v21_seta_d2.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 32),
    "v21_seta_d3.yaml": (20260522, "mobile-front-2024", "./datasets/_fixtures/digiface", 256),
    "v21_setb_d1.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 4),
    "v21_setb_d2.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 32),
    "v21_setb_d3.yaml": (20260523, "webcam-1080p", "./datasets/_fixtures/extended_fixture", 256),
}


def test_v21_configs_present_and_well_formed():
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

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_v21_configs.py -q 2>&1 | tail -3`
Expected: 1 failed (file not found).

- [ ] **Step 3: Create all six configs**

`configs/runs/v21_seta_d1.yaml`:
```yaml
run:
  name: v21_seta_d1
  output: ./datasets/v21_seta_d1
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

`configs/runs/v21_seta_d2.yaml`: identical to `v21_seta_d1.yaml` except `name: v21_seta_d2`, `output: ./datasets/v21_seta_d2`, `samples_per_bonafide: 32`.

`configs/runs/v21_seta_d3.yaml`: identical to `v21_seta_d1.yaml` except `name: v21_seta_d3`, `output: ./datasets/v21_seta_d3`, `samples_per_bonafide: 256`.

`configs/runs/v21_setb_d1.yaml`:
```yaml
run:
  name: v21_setb_d1
  output: ./datasets/v21_setb_d1
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

`configs/runs/v21_setb_d2.yaml`: identical to `v21_setb_d1.yaml` except `name: v21_setb_d2`, `output: ./datasets/v21_setb_d2`, `samples_per_bonafide: 32`.

`configs/runs/v21_setb_d3.yaml`: identical to `v21_setb_d1.yaml` except `name: v21_setb_d3`, `output: ./datasets/v21_setb_d3`, `samples_per_bonafide: 256`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_v21_configs.py -q 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 160 passed, 1 skipped, 4 warnings (159 prior + 1 new).

- [ ] **Step 6: Commit**

```bash
git add configs/runs/v21_seta_d1.yaml configs/runs/v21_seta_d2.yaml configs/runs/v21_seta_d3.yaml configs/runs/v21_setb_d1.yaml configs/runs/v21_setb_d2.yaml configs/runs/v21_setb_d3.yaml pad-synth-face/tests/test_v21_configs.py
git commit -m "feat(pad-spark): six v2.1 measurement configs (D1/D2/D3 on jittered halftone)"
```

---

## Task 6: Generate v2.1 datasets locally

**Files:** none (output to gitignored `datasets/`).

- [ ] **Step 1: Generate all six**

```bash
cd /Users/stuartwells/test
for f in v21_seta_d1 v21_seta_d2 v21_seta_d3 v21_setb_d1 v21_setb_d2 v21_setb_d3; do
  echo "=== generating $f ==="
  .venv/bin/python -m pad_synth_face.cli generate --config configs/runs/${f}.yaml | tail -3
done
```

Expected: each prints a JSON summary with `"failed": 0`. Wall-time roughly similar to v2 (jitter adds 16 rng draws per print sample — negligible).

- [ ] **Step 2: Verify counts**

```bash
for d in datasets/v21_seta_d{1,2,3} datasets/v21_setb_d{1,2,3}; do
  n=$(wc -l < "$d/manifest.jsonl")
  bona=$(grep -c '"label":"bonafide"' "$d/manifest.jsonl")
  attack=$(grep -c '"label":"attack"' "$d/manifest.jsonl")
  printf "%-22s total=%5d  bonafide=%5d  attack=%5d\n" "$d" "$n" "$bona" "$attack"
done
```

Expected (exact):
```
datasets/v21_seta_d1    total=   96  bonafide=   48  attack=   48
datasets/v21_seta_d2    total=  512  bonafide=  256  attack=  256
datasets/v21_seta_d3    total= 4096  bonafide= 2048  attack= 2048
datasets/v21_setb_d1    total=  128  bonafide=   64  attack=   64
datasets/v21_setb_d2    total= 1024  bonafide=  512  attack=  512
datasets/v21_setb_d3    total= 8192  bonafide= 4096  attack= 4096
```

- [ ] **Step 3: Verify the watermark is broken (the load-bearing sanity check)**

Pick two print samples from `v21_seta_d3` that have the same `print_dpi` and visually-inspectable structure, and confirm they're byte-different. The same comparison on a v2 dataset would have shown effectively identical halftone patterns (modulo small differences from texture/warp).

```bash
.venv/bin/python - <<'PY'
import json, hashlib
from pathlib import Path
mani = Path("datasets/v21_seta_d3/manifest.jsonl")
# Find two print samples with the same print_dpi.
records = [json.loads(line) for line in mani.read_text().splitlines() if '"label":"attack"' in line]
print_recs = [r for r in records if r.get("attack_type") == "print"]
by_dpi = {}
for r in print_recs:
    dpi = r["attack_params"].get("print_dpi")
    by_dpi.setdefault(dpi, []).append(r)
# Take the first DPI value with >=2 samples.
for dpi, group in by_dpi.items():
    if len(group) >= 2:
        a, b = group[0], group[1]
        ha = hashlib.sha256(Path("datasets/v21_seta_d3") .joinpath(a["output_path"]).read_bytes()).hexdigest()
        hb = hashlib.sha256(Path("datasets/v21_seta_d3").joinpath(b["output_path"]).read_bytes()).hexdigest()
        print(f"dpi={dpi}: {a['sample_id']}={ha[:12]}... vs {b['sample_id']}={hb[:12]}...")
        assert ha != hb, "Watermark not broken: two same-DPI print samples have identical bytes"
        print("watermark broken: same-DPI samples differ at byte level")
        break
else:
    print("warning: no two same-DPI samples found to compare")
PY
```

Expected: prints `watermark broken: same-DPI samples differ at byte level`. If the assertion fails, STOP and report BLOCKED — the jitter isn't taking effect through the pipeline.

- [ ] **Step 4: No commit** (datasets gitignored; the configs from T5 are the regenerable spec).

---

## Task 7: rsync to Spark + run 27-cell v2.1 sweep

**Files:** none (remote operations).

- [ ] **Step 1: Sync the latest code to the Spark**

```bash
rsync -a --delete \
  --exclude='.venv' --exclude='__pycache__' --exclude='datasets' \
  --exclude='.superpowers' --exclude='.git/objects/pack' \
  /Users/stuartwells/test/ \
  swells@spark-50d2.local:~/ml/projects/pad-spark/
```

- [ ] **Step 2: rsync the six v2.1 datasets**

```bash
for d in v21_seta_d1 v21_seta_d2 v21_seta_d3 v21_setb_d1 v21_setb_d2 v21_setb_d3; do
  rsync -a --partial \
    "/Users/stuartwells/test/datasets/${d}/" \
    "swells@spark-50d2.local:~/ml/datasets/${d}/"
done
```

- [ ] **Step 3: Run the 27-cell v2.1 sweep**

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)_v21
echo "$ts" > /tmp/padspark_v21_ts
ssh swells@spark-50d2.local "cd ~/ml/projects/pad-spark && .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 ~/ml/datasets/v21_seta_d1 --set-b-d1 ~/ml/datasets/v21_setb_d1 \
  --set-a-d2 ~/ml/datasets/v21_seta_d2 --set-b-d2 ~/ml/datasets/v21_setb_d2 \
  --set-a-d3 ~/ml/datasets/v21_seta_d3 --set-b-d3 ~/ml/datasets/v21_setb_d3 \
  --set-a-d4 ~/ml/datasets/spark_seta_d4 --set-b-d4 ~/ml/datasets/spark_setb_d4 \
  --output-dir ~/ml/logs/pad-spark/${ts} \
  --device cuda --epochs 10 --batch-size 32 \
  --cells L1:D1:0,L1:D1:1,L1:D1:2,L1:D2:0,L1:D2:1,L1:D2:2,L1:D3:0,L1:D3:1,L1:D3:2,L2:D1:0,L2:D1:1,L2:D1:2,L2:D2:0,L2:D2:1,L2:D2:2,L2:D3:0,L2:D3:1,L2:D3:2,L3:D1:0,L3:D1:1,L3:D1:2,L3:D2:0,L3:D2:1,L3:D2:2,L3:D3:0,L3:D3:1,L3:D3:2" 2>&1 | tail -32
```

(D4 args pass the existing v1 D4 datasets as placeholders; `--cells` excludes any D4 cell so they aren't loaded.)

Expected: 27 lines `L? D? seed=?  eer_in=0.??  eer_cross=0.??  ??.?s`. Wall-time ~5 minutes.

**Watch for**: any cell printing `eer_cross=0.000` is a red flag — the watermark may have survived. Multiple 0.000 cells would mean v2.1 didn't break the artifact (spec §10 success criterion fails).

- [ ] **Step 4: Confirm 27 JSONs**

```bash
ssh swells@spark-50d2.local "ls ~/ml/logs/pad-spark/$(cat /tmp/padspark_v21_ts)/runs/ | wc -l"
```
Expected: `27`.

- [ ] **Step 5: No commit** (results land in the report directory in T8).

---

## Task 8: rsync results back, author v1/v2/v2.1 three-way report append, commit

**Files:**
- Add (rsync'd): 27 JSONs at `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v21/L{1,2,3}_D{1,2,3}_{0,1,2}.json`
- Add: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_v21.csv`
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (append v2.1 section)
- Modify: `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` (one-line append)

- [ ] **Step 1: rsync the 27 v2.1 JSONs into a new subdir**

```bash
ts=$(cat /tmp/padspark_v21_ts)
mkdir -p docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v21
rsync -av "swells@spark-50d2.local:~/ml/logs/pad-spark/${ts}/runs/" \
  docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v21/
ls docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v21/ | wc -l
```
Expected: 27.

- [ ] **Step 2: Build `summary_v21.csv` from the v2.1 JSONs**

```bash
.venv/bin/python - <<'PY'
import csv, json
from pathlib import Path
runs_dir = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v21")
out = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_v21.csv")
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

- [ ] **Step 3: Compute v1-vs-v2-vs-v2.1 cell aggregates and watermark check**

```bash
.venv/bin/python - <<'PY'
import json, statistics as st
from pathlib import Path
report_dir = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results")

def load(subdir, level_filter):
    cells = {}
    for p in sorted((report_dir / subdir).glob("*.json")):
        r = json.loads(p.read_text())
        if r["data_level"] not in level_filter:
            continue
        cells.setdefault((r["capacity"], r["data_level"]), []).append(r)
    return cells

v1 = load("runs", {"D1", "D2", "D3"})
v2 = load("runs_v2", {"D1", "D2", "D3"})
v21 = load("runs_v21", {"D1", "D2", "D3"})

print(f"{'cell':<8} {'v1':<18}  {'v2':<18}  {'v21':<18}  {'v21 ≤ 0.001?':>13}")
watermark_broken = True
for L in ("L1", "L2", "L3"):
    for D in ("D1", "D2", "D3"):
        v1g = [r["eer_cross_domain"] for r in v1[(L, D)]]
        v2g = [r["eer_cross_domain"] for r in v2[(L, D)]]
        v21g = [r["eer_cross_domain"] for r in v21[(L, D)]]
        m1, s1 = st.mean(v1g), st.stdev(v1g)
        m2, s2 = st.mean(v2g), st.stdev(v2g)
        m21, s21 = st.mean(v21g), st.stdev(v21g)
        broken = m21 > 0.001
        if not broken:
            watermark_broken = False
        print(f"{L} {D}    {m1:.3f}+-{s1:.3f}    {m2:.3f}+-{s2:.3f}    "
              f"{m21:.3f}+-{s21:.3f}    {'yes' if broken else 'NO!!':>10}")

print()
print(f"Watermark broken (all v2.1 cross-domain cells > 0.001): {watermark_broken}")
print()
# Check the gain-survives criterion at D3.
print("D3 v1-vs-v2.1 comparison:")
for L in ("L1", "L2", "L3"):
    v1g = [r["eer_cross_domain"] for r in v1[(L, "D3")]]
    v21g = [r["eer_cross_domain"] for r in v21[(L, "D3")]]
    m1, s1 = st.mean(v1g), st.stdev(v1g)
    m21, s21 = st.mean(v21g), st.stdev(v21g)
    delta = m1 - m21
    overlap = not (m1 - s1 > m21 + s21 or m21 - s21 > m1 + s1)
    fires = delta >= 0.05 and not overlap
    print(f"  {L} D3: v1={m1:.3f}+-{s1:.3f} v21={m21:.3f}+-{s21:.3f} delta={delta:+.3f} overlap={overlap} fires={fires}")
PY
```

Record:
- The 9-cell table (v1 / v2 / v2.1 means and stds).
- Whether `watermark_broken` is True (the load-bearing diagnostic).
- The D3 v1-vs-v2.1 comparison (the "gain survives" check).

- [ ] **Step 4: Append the v2.1 section to the existing report**

Open `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` and append at the end:

```markdown

---

## 2026-05-22 update — v2.1 result (halftone jitter)

The v2 watermark was diagnosed as the deterministic halftone screen geometry. v2.1 adds per-channel per-sample jitter to break it: sub-pixel screen offset ~U(−cell/2, +cell/2), angle ~N(0, 3°), cell-size multiplier ~U(0.90, 1.10). No new ontology axes; ontology version bumped to `2026-05-23`. Regenerated 6 D1–D3 datasets (`datasets/v21_set{a,b}_d{1,2,3}/`) and ran the 27-cell sweep on the same GB10. Code SHA at sweep time: <fill_v21_sha>. Torch: `2.12.0.dev20260407+cu128`.

**Three-way cross-domain EER comparison (mean ± std):**

| Cell | v1 | v2 | v2.1 |
|---|---|---|---|
| L1·D1 | 0.396 ± 0.033 | 0.240 ± 0.070 | <fill_v21_L1_D1> |
| L1·D2 | 0.441 ± 0.029 | 0.240 ± 0.050 | <fill_v21_L1_D2> |
| L1·D3 | 0.228 ± 0.022 | 0.000 ± 0.000 | <fill_v21_L1_D3> |
| L2·D1 | 0.354 ± 0.070 | 0.130 ± 0.059 | <fill_v21_L2_D1> |
| L2·D2 | 0.214 ± 0.005 | 0.000 ± 0.000 | <fill_v21_L2_D2> |
| L2·D3 | 0.217 ± 0.033 | 0.000 ± 0.000 | <fill_v21_L2_D3> |
| L3·D1 | 0.370 ± 0.024 | 0.120 ± 0.153 | <fill_v21_L3_D1> |
| L3·D2 | 0.242 ± 0.017 | 0.000 ± 0.000 | <fill_v21_L3_D2> |
| L3·D3 | 0.249 ± 0.007 | 0.000 ± 0.000 | <fill_v21_L3_D3> |

**Watermark check (spec §10 criterion):** all v2.1 cross-domain cells must have mean > 0.001 EER. Verdict: <fill: BROKEN / SURVIVED — list any cells with mean ≤ 0.001>.

**Gain-survives check (spec §10):** D3 v1-vs-v2.1 delta with non-overlapping ±1σ bands; at least one capacity tier must show Δ ≥ 0.05 with no band overlap.

| L | v1 D3 mean ± std | v2.1 D3 mean ± std | Δ (v1 − v21) | Bands overlap? | Fires (Δ ≥ 0.05, no overlap)? |
|---|---|---|---|---|---|
| L1 | 0.228 ± 0.022 | <fill> | <fill> | <fill> | <fill> |
| L2 | 0.217 ± 0.033 | <fill> | <fill> | <fill> | <fill> |
| L3 | 0.249 ± 0.007 | <fill> | <fill> | <fill> | <fill> |

**Diagnosis:** <one paragraph. If watermark broken AND at least one tier fires the gain test: "physics IS the lever — v2.1 ships as the production print attack." If watermark broken but no tier fires: "the v2 'improvement' was entirely the artifact; physics-axis improvements are not enough. Escalate to real-data integration." If watermark survived: "v2.1 jitter insufficient; investigate which deterministic feature still leaks before another iteration.">

**Phase 2 recommendation update:** <one paragraph reflecting the diagnosis. Specifically: should v2.1 replace v2 in `print.py` as the production halftone? Does the mask-attack sub-project proceed as planned, accelerate, or wait for real-data integration first?>
```

Fill every `<fill>` from the Step 3 output. The git SHA is from any v2.1 result JSON's `git_sha` field.

- [ ] **Step 5: Append the one-line roadmap update**

In `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md`, append at the end:

```markdown

---

## 2026-05-22 update — v2.1 jittered-halftone sweep

v2.1 adds per-sample jitter to halftoning to break the v2 watermark. 27-cell D1–D3 sweep on the Spark. **Watermark verdict: <broken/survived>. Gain verdict: <fires/flat>.** See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"v2.1 result" for the three-way v1/v2/v2.1 comparison and the updated Phase 2 prioritization.
```

Fill `<broken/survived>` and `<fires/flat>` with the Step 3 verdicts.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_v21/ \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_v21.csv \
        docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md
git commit -m "report(pad-print-v21): jittered-halftone sweep — <watermark verdict> / <gain verdict>"
```

Replace the placeholders with the actual verdicts (e.g., `watermark broken / gain fires`).

- [ ] **Step 7: Full suite green (final)**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `160 passed, 1 skipped, 4 warnings`.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose — break watermark while preserving gain | Tasks 1, 2 (jitter implementation) + T7/T8 (measurement) |
| §2 question + 0.000 threshold + Δ ≥ 0.05 D3 rule | Task 8 Step 3 (watermark + gain checks) |
| §3 three jitter sources (offset, angle, cell-size) with exact distributions and draw order | Task 1 (implementation) + plan's Reference block (draw order locked) |
| §4 API change — optional rng param, deterministic default | Task 1 (signature change) + Task 2 (wiring) |
| §5 ontology version bump, no new axes | Task 3 |
| §6 determinism preserved + golden regen | Task 4 |
| §7 measurement plan (6 configs, generate, rsync, run, report) | Tasks 5, 6, 7, 8 |
| §8 architecture boundaries (print.py + ontology + golden + configs + report only) | All tasks honor this; ICC/texture/warp/cutout untouched; replay/sensor/bonafide/pipeline.py/cli unmodified |
| §9 non-goals | None violated: no new ontology axes, no dot-shape categorical, no dot-gain noise added, no replay/ICC changes, no pipeline.py hardcode fix |
| §10 success criteria | Task 8 Step 3 watermark + gain checks |

**Placeholder scan:** Every `<fill_*>`, `<broken/survived>`, `<fires/flat>` is in the report template the implementer populates from real run data in T8 Step 3. No "TBD/TODO/implement later" elsewhere. All code blocks complete.

**Type consistency:** `_apply_halftone(rgb, print_dpi, rng=None)`, `_dot_screen(h, w, cell_px, angle_deg, dx=0.0, dy=0.0)`, `_halftone_channel(channel, cell_px, angle_deg, dx=0.0, dy=0.0)` — consistent across T1, T2. Per-channel draw order `(k, Δθ, dx, dy)` consistent between the Reference block and T1 Step 3 implementation. JSON keys in v2.1 sweep unchanged from parent project's schema.
