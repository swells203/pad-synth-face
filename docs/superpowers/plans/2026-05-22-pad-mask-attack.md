# PAD Mask-Attack Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third face-attack module — `mask` (paper / silicone / resin) — to the v2.1 + DigiFace baseline, producing a clean mask-only cross-domain EER and an integrated print+replay+mask EER.

**Architecture:** A new `MaskAttack` class implements the existing `FaceAttackModule` protocol with a 2D image-space `simulate()`. A `mask_type` ontology axis selects a material-property bundle (Approach A — a module-level dict, mirroring `PrintAttack`'s `_PAPER_TINTS`/`_ICC_PARAMS`); six continuous axes modulate it per sample. The v2/v2.1 artifact lesson is designed in: continuous float until the final `uint8` cast (no binary threshold / no quantization), and per-sample rng jitter on every spatial pattern.

**Tech Stack:** Python 3.12+, NumPy, OpenCV (`cv2`), Pydantic ontology loader, pytest. Sweep runs on the DGX Spark (GB10, CUDA 12.8) via the existing attack-agnostic `scripts/spark_sweep.py`.

**Spec:** [`../specs/2026-05-22-pad-mask-attack-design.md`](../specs/2026-05-22-pad-mask-attack-design.md)

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `ontology/face/mask.yaml` | Mask attack ontology (1 categorical + 6 uniform axes, all with provenance) | Create |
| `pad-synth-face/src/pad_synth_face/attacks/mask.py` | `MaskAttack` class + material bundle + image-space physics stages | Create |
| `pad-synth-face/src/pad_synth_face/pipeline.py` | Register `"mask"`; robustify canonical-version derivation (spec §8) | Modify |
| `pad-synth-face/tests/test_mask_attack.py` | Unit tests: shape/dtype, determinism, jitter, anti-palette, material differentiation | Create |
| `tests/test_ontology_files.py` | Add `test_mask_ontology_loads` | Modify |
| `pad-synth-face/tests/test_pipeline_e2e.py` | Add a mask-inclusive e2e test | Modify |
| `pad-synth-face/tests/test_pipeline_mask_only.py` | Mask-only config exercises the canonical-version fix (no `print` attack) | Create |
| `tests/test_determinism_golden.py` | Add `mask` to the golden config; regenerate golden | Modify |
| `configs/runs/mask_set{a,b}_d{1,2,3}.yaml` | 6 mask-only sweep configs | Create |
| `configs/runs/mix_set{a,b}_d{1,2,3}.yaml` | 6 integrated print+replay+mask sweep configs | Create |
| `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` | Append mask-only + integrated result sections | Modify |

---

## Task 1: Mask ontology YAML

**Files:**
- Create: `ontology/face/mask.yaml`
- Modify (test): `tests/test_ontology_files.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ontology_files.py`:

```python
def test_mask_ontology_loads():
    ont = load_ontology(REPO_ROOT / "ontology" / "face" / "mask.yaml")
    assert ont.attack_type == "mask"
    assert ont.version == "2026-05-22"
    assert set(ont.axes) == {
        "mask_type",
        "light_azimuth_deg",
        "light_elevation_deg",
        "specular_strength",
        "aperture_misalignment_px",
        "surface_warp",
        "seam_visibility",
    }
    assert ont.axes["mask_type"].values == ["paper", "silicone", "resin"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ontology_files.py::test_mask_ontology_loads -v`
Expected: FAIL — `FileNotFoundError` for `mask.yaml`.

- [ ] **Step 3: Create the ontology file**

Create `ontology/face/mask.yaml`:

```yaml
version: "2026-05-22"
attack_type: mask
axes:
  mask_type:
    type: categorical
    values: [paper, silicone, resin]
    weights: [0.30, 0.45, 0.25]
    provenance:
      paper: "Liu et al., 'CASIA-SURF / HiFiMask: A Large-Scale High-Fidelity Mask Dataset', CVPR 2021 (material range and prevalence)"
      doi: "10.1109/CVPR46437.2021.00616"
  light_azimuth_deg:
    type: uniform
    low: -180.0
    high: 180.0
    provenance:
      paper: "Erdogmus & Marcel, 'Spoofing Face Recognition with 3D Masks', IEEE TIFS 2014 (3DMAD capture lighting geometry)"
      doi: "10.1109/TIFS.2014.2322255"
  light_elevation_deg:
    type: uniform
    low: 10.0
    high: 80.0
    provenance:
      paper: "Erdogmus & Marcel, 'Spoofing Face Recognition with 3D Masks', IEEE TIFS 2014 (3DMAD capture lighting geometry)"
      doi: "10.1109/TIFS.2014.2322255"
  specular_strength:
    type: uniform
    low: 0.0
    high: 1.0
    provenance:
      paper: "Manjani et al., 'Detecting Silicone Mask-Based Presentation Attack via Deep Dictionary Learning', IEEE TIFS 2017 (SMAD silicone gloss appearance)"
      doi: "10.1109/TIFS.2017.2676720"
  aperture_misalignment_px:
    type: uniform
    low: 0.0
    high: 4.0
    provenance:
      paper: "Erdogmus & Marcel, 'Spoofing Face Recognition with 3D Masks', IEEE TIFS 2014 (eye/mouth aperture mismatch cue)"
      doi: "10.1109/TIFS.2014.2322255"
  surface_warp:
    type: uniform
    low: 0.0
    high: 1.0
    provenance:
      paper: "Manjani et al., 'Detecting Silicone Mask-Based Presentation Attack via Deep Dictionary Learning', IEEE TIFS 2017 (flexible-mask drape deformation)"
      doi: "10.1109/TIFS.2017.2676720"
  seam_visibility:
    type: uniform
    low: 0.0
    high: 1.0
    provenance:
      paper: "Liu et al., 'CASIA-SURF / HiFiMask', CVPR 2021 (mask-boundary / seam artifacts)"
      doi: "10.1109/CVPR46437.2021.00616"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ontology_files.py::test_mask_ontology_loads -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ontology/face/mask.yaml tests/test_ontology_files.py
git commit -m "feat(pad-mask): mask attack ontology (paper/silicone/resin, 7 axes)"
```

---

## Task 2: MaskAttack skeleton — protocol conformance

Build the class with a pass-through `simulate()` first, proving shape/dtype/determinism before adding physics.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/attacks/mask.py`
- Create (test): `pad-synth-face/tests/test_mask_attack.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_mask_attack.py`:

```python
from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.mask import MaskAttack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ontology():
    return load_ontology(REPO_ROOT / "ontology" / "face" / "mask.yaml")


def test_mask_attack_returns_same_shape_uint8():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = MaskAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.dtype == np.uint8
    assert out.shape == bonafide.shape


def test_mask_attack_is_deterministic():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    rng1 = sample_rng(123)
    p1 = attack.sample_params(rng1)
    out1 = attack.simulate(bonafide, p1, rng1)

    rng2 = sample_rng(123)
    p2 = attack.sample_params(rng2)
    out2 = attack.simulate(bonafide, p2, rng2)

    assert p1 == p2
    assert np.array_equal(out1, out2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pad-synth-face/tests/test_mask_attack.py -v`
Expected: FAIL — `ModuleNotFoundError: pad_synth_face.attacks.mask`.

- [ ] **Step 3: Create the skeleton module**

Create `pad-synth-face/src/pad_synth_face/attacks/mask.py`:

```python
"""Phase 2 mask-attack simulator (worn paper / silicone / resin masks).

2D image-space approximation of a worn face mask. No real 3D geometry: the
"3D-ness" is faked with an analytic elliptical-dome shading field, a soft
specular term, eye/mouth aperture misregistration, a non-rigid drape warp,
and a perimeter seam.

Artifact discipline (the v2/v2.1 print lesson, designed in): the pipeline
stays in continuous float until the final uint8 cast -- NO binary
thresholding or colour quantisation anywhere -- and every spatial pattern is
per-sample jittered from the rng so Set A and Set B never share a fixed
geometry a detector could memorise.

mask_type selects a material-property bundle; the continuous ontology axes
modulate it per sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pad_synth_core.ontology import Ontology


class MaskAttack:
    name = "mask"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "mask"
        self.ontology = ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.ontology.sample_params(rng)

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Pass-through placeholder; physics added in Task 3.
        img = bonafide.astype(np.float32) / 255.0
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pad-synth-face/tests/test_mask_attack.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/mask.py pad-synth-face/tests/test_mask_attack.py
git commit -m "feat(pad-mask): MaskAttack skeleton (protocol conformance)"
```

---

## Task 3: Mask physics — material bundle + image-space stages

Replace the pass-through with the full continuous, jittered pipeline (spec §5–§7).

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/attacks/mask.py`
- Modify (test): `pad-synth-face/tests/test_mask_attack.py`

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-face/tests/test_mask_attack.py`:

```python
def test_mask_jitter_different_seeds_differ():
    """Load-bearing anti-watermark invariant: two rngs -> two outputs."""
    bonafide = np.full((64, 64, 3), 150, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    rng1 = sample_rng(1)
    out1 = attack.simulate(bonafide, attack.sample_params(rng1), rng1)
    rng2 = sample_rng(2)
    out2 = attack.simulate(bonafide, attack.sample_params(rng2), rng2)

    assert not np.array_equal(out1, out2)


def test_mask_output_is_not_quantised():
    """Anti-palette guard (the exact v2 halftone mistake): continuous output
    must have far more than the 16-colour palette that produced the watermark."""
    rng = sample_rng(0)
    bonafide = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    attack = MaskAttack(_ontology())
    srng = sample_rng(5)
    out = attack.simulate(bonafide, attack.sample_params(srng), srng)
    n_colors = np.unique(out.reshape(-1, 3), axis=0).shape[0]
    assert n_colors > 1000


def test_mask_materials_are_distinguishable():
    """The three mask_type bundles must produce measurably different images."""
    bonafide = np.full((64, 64, 3), 140, dtype=np.uint8)
    attack = MaskAttack(_ontology())

    outs = {}
    for mat in ("paper", "silicone", "resin"):
        rng = sample_rng(42)
        params = attack.sample_params(rng)
        params["mask_type"] = mat  # hold all other axes fixed
        outs[mat] = attack.simulate(bonafide, params, rng)

    assert not np.array_equal(outs["paper"], outs["silicone"])
    assert not np.array_equal(outs["silicone"], outs["resin"])
    means = {m: float(o.mean()) for m, o in outs.items()}
    # Materials differ by more than rounding noise.
    assert max(means.values()) - min(means.values()) > 1.0


def test_mask_preserves_shape_and_range_on_random_input():
    rng = sample_rng(9)
    bonafide = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    attack = MaskAttack(_ontology())
    srng = sample_rng(3)
    out = attack.simulate(bonafide, attack.sample_params(srng), srng)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pad-synth-face/tests/test_mask_attack.py -v`
Expected: FAIL — `test_mask_jitter_different_seeds_differ` (pass-through ignores rng) and `test_mask_materials_are_distinguishable` (pass-through ignores `mask_type`).

- [ ] **Step 3: Implement the physics**

Replace the entire contents of `pad-synth-face/src/pad_synth_face/attacks/mask.py` with:

```python
"""Phase 2 mask-attack simulator (worn paper / silicone / resin masks).

2D image-space approximation of a worn face mask. No real 3D geometry: the
"3D-ness" is faked with an analytic elliptical-dome shading field, a soft
Blinn-Phong specular term, eye/mouth aperture misregistration, a non-rigid
drape warp, and a perimeter seam.

Artifact discipline (the v2/v2.1 print lesson, designed in): the pipeline
stays in continuous float until the final uint8 cast -- NO binary
thresholding or colour quantisation anywhere -- and every spatial pattern is
per-sample jittered from the rng so Set A and Set B never share a fixed
geometry a detector could memorise.

mask_type selects a material-property bundle (colour cast, specular scale,
texture-loss sigma, subsurface tint); the continuous ontology axes modulate
it per sample. This module corresponds to mask ontology_version 2026-05-22.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pad_synth_core.ontology import Ontology


@dataclass(frozen=True)
class MaskMaterial:
    color_cast: tuple[float, float, float]
    specular_scale: float
    texture_loss_sigma: float
    subsurface_tint: tuple[float, float, float]


# Material constants only; ALL per-sample variation comes from the jittered
# ontology axes, never from this table. Values informed by SMAD (silicone
# gloss/translucency) and HiFiMask (paper/resin appearance) -- see ontology
# provenance.
_MASK_MATERIALS: dict[str, MaskMaterial] = {
    "paper":    MaskMaterial((0.98, 0.97, 0.94), 0.15, 0.6, (0.00, 0.00, 0.00)),
    "silicone": MaskMaterial((1.02, 0.98, 0.95), 0.55, 1.0, (0.06, 0.02, 0.02)),
    "resin":    MaskMaterial((0.97, 0.98, 1.03), 0.85, 1.4, (0.00, 0.00, 0.00)),
}


def _texture_loss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass: masks lack fine skin-pore detail. Continuous."""
    k = max(3, int(2 * round(sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def _dome_normals(h: int, w: int) -> np.ndarray:
    """Unit normals (H,W,3) of an elliptical dome over the frame. Image-space
    parametric field -- NOT a per-face depth estimate."""
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    nx = (xv - cx) / (w * 0.5)
    ny = (yv - cy) / (h * 0.5)
    z = np.sqrt(np.clip(1.0 - nx**2 - ny**2, 0.0, 1.0))
    n = np.stack([nx, ny, z], axis=-1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def _light_dir(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float32,
    )


def _shading(img: np.ndarray, normals: np.ndarray, light: np.ndarray,
             strength: float = 0.35) -> np.ndarray:
    """Lambertian shading gradient lit by `light`."""
    lam = np.clip((normals * light).sum(axis=-1), 0.0, 1.0)
    factor = (1.0 - strength) + strength * lam[..., None]
    return img * factor


def _specular(img: np.ndarray, normals: np.ndarray, light: np.ndarray,
              intensity: float) -> np.ndarray:
    """Soft Blinn-Phong highlight; view fixed at +z. Continuous additive."""
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = light + view
    half = half / (np.linalg.norm(half) + 1e-6)
    spec = np.clip((normals * half).sum(axis=-1), 0.0, 1.0) ** 32
    return np.clip(img + intensity * spec[..., None], 0.0, 1.0)


def _aperture_mismatch(img: np.ndarray, offset_px: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Soft darkening at eye/mouth regions, centre offset by offset_px in a
    jittered direction (models the wearer's features not lining up)."""
    h, w = img.shape[:2]
    angle = float(rng.uniform(-np.pi, np.pi))
    dy = offset_px * np.sin(angle)
    dx = offset_px * np.cos(angle)
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    factor = np.ones((h, w, 1), dtype=np.float32)
    for cy_f, cx_f, ry_f, rx_f in (
        (0.36, 0.30, 0.06, 0.10),  # left eye
        (0.36, 0.70, 0.06, 0.10),  # right eye
        (0.70, 0.50, 0.08, 0.16),  # mouth
    ):
        cy = cy_f * h + dy
        cx = cx_f * w + dx
        r = ((yv - cy) / (ry_f * h)) ** 2 + ((xv - cx) / (rx_f * w)) ** 2
        factor[..., 0] *= 1.0 - 0.5 * np.exp(-r)
    return img * factor


def _drape_warp(img: np.ndarray, amount: float,
                rng: np.random.Generator) -> np.ndarray:
    """Mild non-rigid perspective warp for the mask draping over the face."""
    h, w = img.shape[:2]
    m = amount * 0.08 * w
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + rng.uniform(-m, m, size=(4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _seam(img: np.ndarray, visibility: float) -> np.ndarray:
    """Darkened elliptical ring at the mask perimeter."""
    h, w = img.shape[:2]
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yv - cy) / (h * 0.46)) ** 2 + ((xv - cx) / (w * 0.40)) ** 2)
    ring = np.exp(-((r - 1.0) ** 2) / (2.0 * 0.08**2))
    factor = 1.0 - visibility * 0.5 * ring[..., None]
    return img * factor


class MaskAttack:
    name = "mask"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "mask"
        self.ontology = ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.ontology.sample_params(rng)

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        h, w = bonafide.shape[:2]
        img = bonafide.astype(np.float32) / 255.0
        mat = _MASK_MATERIALS[params["mask_type"]]

        # 1. Texture loss (jittered sigma).
        sigma = max(0.3, mat.texture_loss_sigma * float(rng.uniform(0.85, 1.15)))
        img = _texture_loss(img, sigma)

        # 2. Material colour cast + subsurface tint (continuous, no quantise).
        cast = np.array(mat.color_cast, dtype=np.float32)
        sub = np.array(mat.subsurface_tint, dtype=np.float32)
        img = np.clip(img * cast + sub, 0.0, 1.0)

        # 3 + 4. Pseudo-3D shading and specular from the dome normals.
        normals = _dome_normals(h, w)
        light = _light_dir(
            float(params["light_azimuth_deg"]),
            float(params["light_elevation_deg"]),
        )
        img = _shading(img, normals, light)
        spec_intensity = (
            float(params["specular_strength"])
            * mat.specular_scale
            * float(rng.uniform(0.85, 1.15))
        )
        img = _specular(img, normals, light, spec_intensity)

        # 5. Aperture mismatch (jittered offset direction).
        img = _aperture_mismatch(img, float(params["aperture_misalignment_px"]), rng)

        # 6. Drape warp.
        img = _drape_warp(img, float(params["surface_warp"]), rng)

        # 7. Perimeter seam.
        img = _seam(img, float(params["seam_visibility"]))

        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run the full mask test file to verify it passes**

Run: `python -m pytest pad-synth-face/tests/test_mask_attack.py -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/mask.py pad-synth-face/tests/test_mask_attack.py
git commit -m "feat(pad-mask): image-space mask physics (continuous, jittered, anti-palette)"
```

---

## Task 4: Robustify canonical-version derivation (spec §8)

`pipeline.py:112` does `attack_modules["print"].ontology.version`, which `KeyError`s for a mask-only config. Fix before registering mask.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/pipeline.py`
- Create (test): `pad-synth-face/tests/test_pipeline_canonical_version.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-face/tests/test_pipeline_canonical_version.py`:

```python
from pad_synth_face.pipeline import _canonical_ontology_version


class _FakeOnt:
    def __init__(self, version):
        self.version = version


class _FakeModule:
    def __init__(self, version):
        self.ontology = _FakeOnt(version)


def test_prefers_print_when_present():
    mods = {"print": _FakeModule("P"), "replay": _FakeModule("R")}
    assert _canonical_ontology_version(mods) == "P"


def test_falls_back_to_priority_then_alpha():
    # No print: priority order is print -> replay -> mask.
    mods = {"mask": _FakeModule("M"), "replay": _FakeModule("R")}
    assert _canonical_ontology_version(mods) == "R"


def test_mask_only_does_not_raise():
    mods = {"mask": _FakeModule("M")}
    assert _canonical_ontology_version(mods) == "M"


def test_unknown_attack_uses_alphabetical_first():
    mods = {"zeta": _FakeModule("Z"), "alpha": _FakeModule("A")}
    assert _canonical_ontology_version(mods) == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pad-synth-face/tests/test_pipeline_canonical_version.py -v`
Expected: FAIL — `ImportError: cannot import name '_canonical_ontology_version'`.

- [ ] **Step 3: Add the helper and use it**

In `pad-synth-face/src/pad_synth_face/pipeline.py`, add this function just above `run_pipeline` (after the `_record_ontology_citations` function, ~line 68):

```python
def _canonical_ontology_version(attack_modules: dict[str, Any]) -> str:
    """Pick one canonical ontology version to stamp on every sample record.

    Bonafide records have no attack ontology of their own, so the dataset
    borrows one attack's version. Prefer print (the dominant version-tracked
    component historically), then replay, then mask; otherwise fall back to
    the alphabetically-first attack present. Robust to mask-only configs that
    have no print attack.
    """
    for preferred in ("print", "replay", "mask"):
        if preferred in attack_modules:
            return attack_modules[preferred].ontology.version
    first = sorted(attack_modules)[0]
    return attack_modules[first].ontology.version
```

Then replace line 112:

```python
    _ontology_version = attack_modules["print"].ontology.version
```

with:

```python
    _ontology_version = _canonical_ontology_version(attack_modules)
```

- [ ] **Step 4: Run test + full suite to verify pass and no regressions**

Run: `python -m pytest pad-synth-face/tests/test_pipeline_canonical_version.py pad-synth-face/tests/test_pipeline_dynamic_ontology_version.py -v`
Expected: PASS (new tests + the existing dynamic-version test unchanged).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/pipeline.py pad-synth-face/tests/test_pipeline_canonical_version.py
git commit -m "fix(pad-face): robustify canonical ontology-version derivation for mask-only configs"
```

---

## Task 5: Register mask + mask-only pipeline e2e

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/pipeline.py`
- Create (test): `pad-synth-face/tests/test_pipeline_mask_only.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-face/tests/test_pipeline_mask_only.py`:

```python
import json
from pathlib import Path

import yaml

from pad_synth_face.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mask_only_pipeline_runs(fixture_bonafide_dir: Path, tmp_path: Path):
    config = {
        "run": {
            "name": "mask_only",
            "output": str(tmp_path / "out"),
            "seed": 7,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 1},
        "attacks": {
            "mask": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "mask.yaml"),
            }
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "mask_only.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    assert summary["samples_generated"] == 8
    assert summary["bonafide_emitted"] == 8

    manifest = (Path(config["run"]["output"]) / "manifest.jsonl").read_text()
    recs = [json.loads(line) for line in manifest.splitlines()]
    attack_types = {r["attack_type"] for r in recs if r["label"] == "attack"}
    assert attack_types == {"mask"}
    # Canonical version came from the mask ontology (no print present).
    assert any(r["ontology_version"] == "2026-05-22" for r in recs)
    # Mask images landed under face/mask/.
    assert list((Path(config["run"]["output"]) / "face" / "mask").glob("*.jpg"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pad-synth-face/tests/test_pipeline_mask_only.py -v`
Expected: FAIL — `KeyError: 'mask'` in `_ATTACK_REGISTRY`.

- [ ] **Step 3: Register the attack**

In `pad-synth-face/src/pad_synth_face/pipeline.py`, add the import (with the other attack imports, ~line 31-32):

```python
from pad_synth_face.attacks.mask import MaskAttack
```

and update the registry (~line 37):

```python
_ATTACK_REGISTRY = {"print": PrintAttack, "replay": ReplayAttack, "mask": MaskAttack}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pad-synth-face/tests/test_pipeline_mask_only.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/pipeline.py pad-synth-face/tests/test_pipeline_mask_only.py
git commit -m "feat(pad-mask): register mask in attack registry + mask-only e2e test"
```

---

## Task 6: Mask in multi-attack e2e + golden regeneration

**Files:**
- Modify: `pad-synth-face/tests/test_pipeline_e2e.py`
- Modify: `tests/test_determinism_golden.py`
- Modify: `tests/golden/golden_hashes.json` (regenerated)

- [ ] **Step 1: Write the failing test**

Append to `pad-synth-face/tests/test_pipeline_e2e.py`:

```python
def test_run_pipeline_with_three_attacks(
    fixture_bonafide_dir: Path, tmp_path: Path
):
    config = {
        "run": {
            "name": "three_attacks",
            "output": str(tmp_path / "out"),
            "seed": 2024,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 2},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml"),
            },
            "mask": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "mask.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "three.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    # 8 identities x samples_per_bonafide=2 = 16 attack slots, 16 bonafide.
    # (Attack type per slot is an independent weighted draw — orchestrator.py
    # line ~49 — so we assert the three-attack path runs cleanly and only valid
    # types appear, not that mask appears for this specific seed. Task 5's
    # mask-only test already proves mask renders end-to-end.)
    assert summary["samples_generated"] == 16
    assert summary["bonafide_emitted"] == 16
    assert summary["samples_failed"] == 0

    manifest = (Path(config["run"]["output"]) / "manifest.jsonl").read_text()
    attack_types = {
        json.loads(line)["attack_type"]
        for line in manifest.splitlines()
        if json.loads(line)["label"] == "attack"
    }
    assert attack_types.issubset({"print", "replay", "mask"})
    assert attack_types  # at least one attack was emitted
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest "pad-synth-face/tests/test_pipeline_e2e.py::test_run_pipeline_with_three_attacks" -v`
Expected: PASS (the registry already supports mask after Task 5). This test documents the three-attack path.

- [ ] **Step 3: Add mask to the golden config**

In `tests/test_determinism_golden.py`, add a mask entry to the `attacks` dict in `_run` (after the `replay` entry, ~line 36):

```python
            "mask": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "mask.yaml"),
            },
```

- [ ] **Step 4: Regenerate the golden and verify it locks**

Run:
```bash
PAD_SYNTH_UPDATE_GOLDEN=1 python -m pytest tests/test_determinism_golden.py -v
python -m pytest tests/test_determinism_golden.py -v
```
Expected: first run rewrites `tests/golden/golden_hashes.json` and passes; second run passes against the regenerated golden. (The golden now covers all three attacks; hash changes are intentional — adding an attack changes the work distribution.)

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/tests/test_pipeline_e2e.py tests/test_determinism_golden.py tests/golden/golden_hashes.json
git commit -m "test(pad-mask): three-attack e2e + regenerate determinism golden with mask"
```

---

## Task 7: Run the full test suite (checkpoint)

**Files:** none (verification only).

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass (the prior baseline was `171 passed, 1 skipped`; this adds the new mask tests, still 1 skipped for no-CUDA). If anything fails, fix before proceeding — do not generate datasets on a red suite.

- [ ] **Step 2: Lint**

Run: `ruff check pad-synth-face ontology tests`
Expected: clean. Fix any findings and re-run.

- [ ] **Step 3: Commit (only if lint fixes were needed)**

```bash
git add -A
git commit -m "style(pad-mask): ruff fixes"
```

---

## Task 8: Mask-only sweep configs (deliverable 1)

Six configs mirroring the existing `real_set{a,b}_d{1,2,3}.yaml`, but with the `mask` attack only. They reuse the pinned DigiFace identity lists and the v2.1 bonafide base.

**Files:**
- Create: `configs/runs/mask_seta_d1.yaml`, `mask_seta_d2.yaml`, `mask_seta_d3.yaml`, `mask_setb_d1.yaml`, `mask_setb_d2.yaml`, `mask_setb_d3.yaml`

- [ ] **Step 1: Generate each config from its `real_*` sibling**

For each of the six `(set, d)` combinations, copy the matching `real_<set>_<d>.yaml`, change the `run.name`/`run.output` to the `mask_*` name, and replace the entire `attacks:` block with a mask-only block. Concretely, for `configs/runs/mask_seta_d3.yaml`:

```yaml
run:
  name: mask_seta_d3
  output: ./datasets/mask_seta_d3
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/digiface_118k_64
  samples_per_bonafide: 256
  identities_file: ./configs/digiface_identities_seta.txt
  splits:
    train: 0.0
    dev: 0.0
    test: 1.0

attacks:
  mask:
    weight: 1.0
    ontology: ./ontology/face/mask.yaml

sensor_preset: mobile-front-2024
```

For the other five files: take the `run`/`bonafide`/`sensor_preset` blocks verbatim from the corresponding `real_<set>_<d>.yaml` (so `samples_per_bonafide`, `identities_file`, and `splits` exactly match the published sweep), change only `run.name` and `run.output` to `mask_<set>_<d>` / `./datasets/mask_<set>_<d>`, and use the same mask-only `attacks:` block shown above.

- [ ] **Step 2: Verify each config loads and is mask-only**

Run:
```bash
python - <<'PY'
import yaml, glob
for f in sorted(glob.glob("configs/runs/mask_set*_d*.yaml")):
    c = yaml.safe_load(open(f))
    assert list(c["attacks"]) == ["mask"], f
    assert c["run"]["output"] == f"./datasets/{c['run']['name']}", f
    print("ok", f)
PY
```
Expected: `ok` for all six files.

- [ ] **Step 3: Commit**

```bash
git add configs/runs/mask_set*_d*.yaml
git commit -m "feat(pad-mask): six mask-only sweep configs (DigiFace base, Set A/B, D1-D3)"
```

---

## Task 9: Integrated print+replay+mask sweep configs (deliverable 2)

Six configs identical to the `real_*` ones but with `mask` added alongside the existing `print` + `replay`.

**Files:**
- Create: `configs/runs/mix_seta_d1.yaml` … `mix_setb_d3.yaml` (six)

- [ ] **Step 1: Generate each config from its `real_*` sibling**

For each `(set, d)`, copy `real_<set>_<d>.yaml`, change `run.name`/`run.output` to `mix_<set>_<d>` / `./datasets/mix_<set>_<d>`, and add the mask attack to the existing `attacks:` block. For `configs/runs/mix_seta_d3.yaml` the `attacks:` block is:

```yaml
attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml
  mask:
    weight: 1.0
    ontology: ./ontology/face/mask.yaml
```

All other blocks (`run` except name/output, `bonafide`, `sensor_preset`) are copied verbatim from `real_<set>_<d>.yaml`.

- [ ] **Step 2: Verify each config loads with all three attacks**

Run:
```bash
python - <<'PY'
import yaml, glob
for f in sorted(glob.glob("configs/runs/mix_set*_d*.yaml")):
    c = yaml.safe_load(open(f))
    assert set(c["attacks"]) == {"print", "replay", "mask"}, f
    assert c["run"]["output"] == f"./datasets/{c['run']['name']}", f
    print("ok", f)
PY
```
Expected: `ok` for all six.

- [ ] **Step 3: Commit**

```bash
git add configs/runs/mix_set*_d*.yaml
git commit -m "feat(pad-mask): six integrated print+replay+mask sweep configs"
```

---

## Task 10: Generate datasets + run the sweeps on the Spark, append report

The sweep needs the GB10 (CUDA 12.8). Datasets are generated by the pipeline (CPU) then swept on the Spark. The host/bootstrap workflow is the standard one for this repo.

**Files:**
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`

- [ ] **Step 1: Generate the twelve datasets**

For each mask-only and integrated config, run the pipeline to materialise the dataset directory. Example (one cell):

```bash
python -m pad_synth_face.cli generate --config configs/runs/mask_seta_d1.yaml
```
Repeat for all six `mask_*` and six `mix_*` configs (or script the loop). Confirm each `./datasets/<name>/manifest.jsonl` exists and has both `bonafide` and `attack` labels.

- [ ] **Step 2: Byte-level anti-watermark sanity check (mirrors the v2.1 T6 check)**

Before trusting any EER, confirm two same-`mask_type` mask attacks are byte-different:

```bash
python - <<'PY'
import json, hashlib, pathlib
root = pathlib.Path("datasets/mask_seta_d3")
recs = [json.loads(l) for l in (root/"manifest.jsonl").read_text().splitlines()]
masks = [r for r in recs if r.get("attack_type") == "mask"
         and r.get("attack_params", {}).get("mask_type") == "silicone"][:2]
hs = [hashlib.sha256((root/r["output_path"]).read_bytes()).hexdigest() for r in masks]
assert hs[0] != hs[1], "WATERMARK: two same-mask_type samples are byte-identical"
print("jitter OK:", hs[0][:8], hs[1][:8])
PY
```
Expected: `jitter OK:` with two different hashes.

- [ ] **Step 3: Sync to the Spark and run the mask-only sweep**

Per the repo's Spark workflow (`ssh swells@spark-50d2.local`, bootstrap via `scripts/setup_spark.sh`, rsync repo to `~/ml/projects/pad-spark/`). Then run the attack-agnostic sweep, pointing Set A → train, Set B → eval. Mask-only uses D1–D3 (D4 not generated):

```bash
python scripts/spark_sweep.py \
  --set-a-d1 datasets/mask_seta_d1 --set-b-d1 datasets/mask_setb_d1 \
  --set-a-d2 datasets/mask_seta_d2 --set-b-d2 datasets/mask_setb_d2 \
  --set-a-d3 datasets/mask_seta_d3 --set-b-d3 datasets/mask_setb_d3 \
  --set-a-d4 datasets/mask_seta_d3 --set-b-d4 datasets/mask_setb_d3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask \
  --cells "$(python - <<'PY'
print(",".join(f"{L}:{D}:{s}" for L in ("L1","L2","L3") for D in ("D1","D2","D3") for s in (0,1,2)))
PY
)" --device cuda
```
(The `--set-*-d4` flags are required by the parser; pointing them at the D3 dirs satisfies the argument without sweeping D4 since no `D4` cells are requested.) Expected: 27 per-cell JSONs + `summary.csv` under `runs_mask/`.

- [ ] **Step 4: Run the integrated sweep**

Same invocation with `mix_*` datasets and `--output-dir …/runs_mix`.

- [ ] **Step 5: Append both result sections to the report**

Add two new dated `## 2026-05-…` sections to `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`, following the format of the existing real-bonafide section: a cross-domain EER table (mean ± std across 3 seeds), an in-domain table, the artifact verdict against the `no cell ≤ 0.001` rule, the headline mask-only number (compare to print's L1·D3 = 0.178), and how the integrated number compares to the per-attack numbers. Link the raw JSON/CSV dirs.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix
git commit -m "report(pad-mask): mask-only + integrated cross-domain sweep results"
```

---

## Self-review notes

- **Spec coverage:** ontology (§4) → Task 1; module + material bundle + 7 stages (§5–§6) → Tasks 2–3; artifact discipline (§7) → Task 3 tests + Task 10 Step 2; registry + eval auto-discovery (§3) → Task 5; `pipeline.py:112` fix (§8) → Task 4; deliverable 1 mask-only (§9) → Tasks 8, 10; deliverable 2 integrated (§9) → Tasks 9, 10; report append (§9) → Task 10; testing (§10) → Tasks 1–6.
- **Decision rule** (`no cross-domain cell ≤ 0.001` = artifact-free) is carried into Task 10 Step 5.
- **No new eval / sweep code** — `scripts/spark_sweep.py` and `TinyPADDataset` are unchanged, as the spec promised.
