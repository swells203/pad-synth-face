# PAD Synthetic Dataset — Phase 1.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up a synthetic cross-domain proxy eval (Set A → Set B) so the project gets a generalization signal that decides Phase 2 priorities (go deep on physics vs go wide on attack types).

**Architecture:** Three small additions to the existing repo — a `WEBCAM_1080P` sensor preset, an extended procedural bonafide fixture with skin-tone color stats and face-like silhouette, and a `train_and_cross_domain_eval()` entry point that evaluates a Set-A-trained detector on a Set-B held-out set. No new packages, no architectural changes. The existing `DigiFaceLoader` works on both fixtures because the directory layout is identical.

**Tech Stack:** Python 3.11, numpy, Pillow, PyTorch (already in the `eval` extras), pytest. All deps already pinned in the workspace.

---

## Spec reference

The full design is at `docs/superpowers/specs/2026-05-11-pad-synth-phase1_5-design.md`. Strategic context is at `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md`.

## File structure changes

| File | Action | Responsibility |
|---|---|---|
| `pad-synth-face/src/pad_synth_face/sensor.py` | Modify | Add `WEBCAM_1080P` preset |
| `pad-synth-face/src/pad_synth_face/_fixtures.py` | Modify | Add `build_extended_fixture_bonafide()` |
| `pad-synth-face/src/pad_synth_face/pipeline.py` | Modify | Register webcam preset in `_SENSOR_REGISTRY` |
| `pad-synth-face/src/pad_synth_face/cli.py` | Modify | Add `eval` subcommand |
| `pad-synth-core/src/pad_synth_core/eval/baseline.py` | Modify | Add `train_and_cross_domain_eval()`, keep old API as wrapper |
| `pad-synth-face/tests/test_sensor.py` | Modify | Test webcam preset |
| `pad-synth-face/tests/test_fixtures.py` | Create | Test extended fixture |
| `pad-synth-face/tests/test_bonafide.py` | Modify | Test (0,0,1) split edge case |
| `pad-synth-core/tests/test_eval_baseline.py` | Modify | Test cross-domain eval |
| `pad-synth-face/tests/test_cli.py` | Create | Test new `eval` subcommand |
| `tests/test_phase15_integration.py` | Create | End-to-end smoke run of Phase 1.5 |
| `configs/runs/phase15_setb.yaml` | Create | Set B generation config |
| `LIMITATIONS.md` | Create | Document Set B as synthetic proxy |

Total: 7 modified files, 6 new files.

---

## Task 1: Add `WEBCAM_1080P` sensor preset

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/sensor.py`
- Modify: `pad-synth-face/tests/test_sensor.py`

- [ ] **Step 1: Write the failing test**

Append to `pad-synth-face/tests/test_sensor.py`:
```python
def test_webcam_1080p_preset_exists_and_applies():
    from pad_synth_face.sensor import WEBCAM_1080P, apply_sensor

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    rng = sample_rng(0)
    out, params = apply_sensor(img, WEBCAM_1080P, rng)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert params["preset"] == "webcam-1080p"


def test_webcam_1080p_has_distinct_iso_range_from_mobile():
    from pad_synth_face.sensor import MOBILE_FRONT_2024, WEBCAM_1080P

    # Webcams typically work in lower light → higher max ISO.
    assert WEBCAM_1080P.iso_range[1] > MOBILE_FRONT_2024.iso_range[1]
    # Webcams typically lossier JPEG → lower max QF.
    assert WEBCAM_1080P.jpeg_qf_range[1] <= MOBILE_FRONT_2024.jpeg_qf_range[1]
    # Webcams typically less vignette than tight phone front cam.
    assert WEBCAM_1080P.vignette_strength < MOBILE_FRONT_2024.vignette_strength
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/stuartwells/test
.venv/bin/python -m pytest pad-synth-face/tests/test_sensor.py::test_webcam_1080p_preset_exists_and_applies -v 2>&1 | tail -10
```

Expected: ImportError on `WEBCAM_1080P`.

- [ ] **Step 3: Add the preset constant**

Append to `pad-synth-face/src/pad_synth_face/sensor.py` (immediately after the existing `MOBILE_FRONT_2024 = ...` block):

```python
WEBCAM_1080P = SensorPreset(
    name="webcam-1080p",
    iso_range=(200, 1600),
    jpeg_qf_range=(70, 92),
    wb_k_range=(3200, 6000),
    vignette_strength=0.20,
)
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_sensor.py -v 2>&1 | tail -15
```

Expected: all sensor tests pass (4 existing + 2 new = 6 passed).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor.py
git commit -m "feat(face): add webcam-1080p sensor preset for Phase 1.5 Set B"
```

---

## Task 2: Register `WEBCAM_1080P` in pipeline registry

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/pipeline.py`
- Modify: `pad-synth-face/tests/test_pipeline_e2e.py`

- [ ] **Step 1: Write the failing test**

Append to `pad-synth-face/tests/test_pipeline_e2e.py`:
```python
def test_pipeline_accepts_webcam_1080p_preset(
    fixture_bonafide_dir: Path, tmp_path: Path
):
    config = {
        "run": {
            "name": "webcam_smoke",
            "output": str(tmp_path / "out"),
            "seed": 1,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 1},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
        },
        "sensor_preset": "webcam-1080p",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    assert summary["samples_generated"] == 8
    assert summary["bonafide_emitted"] == 8

    # Spot-check that one manifest record records the webcam preset.
    manifest_path = Path(config["run"]["output"]) / "manifest.jsonl"
    first = json.loads(manifest_path.read_text().splitlines()[0])
    assert first["sensor_preset"] == "webcam-1080p"
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_e2e.py::test_pipeline_accepts_webcam_1080p_preset -v 2>&1 | tail -10
```

Expected: KeyError or similar — `webcam-1080p` not in `_SENSOR_REGISTRY`.

- [ ] **Step 3: Register the preset**

In `pad-synth-face/src/pad_synth_face/pipeline.py`, find the line:
```python
_SENSOR_REGISTRY = {"mobile-front-2024": MOBILE_FRONT_2024}
```

Replace with:
```python
_SENSOR_REGISTRY = {
    "mobile-front-2024": MOBILE_FRONT_2024,
    "webcam-1080p": WEBCAM_1080P,
}
```

And update the import on the line above. Find:
```python
from pad_synth_face.sensor import MOBILE_FRONT_2024, apply_sensor
```

Replace with:
```python
from pad_synth_face.sensor import MOBILE_FRONT_2024, WEBCAM_1080P, apply_sensor
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_e2e.py -v 2>&1 | tail -10
```

Expected: 4 pipeline e2e tests pass (3 existing + 1 new).

- [ ] **Step 5: Verify no regression in full suite**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```

Expected: 57 passed (55 existing + 2 from Task 1 + 1 from Task 2 = 58 actually; verify count matches your local).

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/pipeline.py pad-synth-face/tests/test_pipeline_e2e.py
git commit -m "feat(face): register webcam-1080p in pipeline sensor registry"
```

---

## Task 3: Verify `identity_disjoint_split` handles `(0.0, 0.0, 1.0)` ratios

The Set B config will use `splits: {train: 0.0, dev: 0.0, test: 1.0}` to put every identity in the test split. The current `int(round(n * 0.0))` arithmetic should yield this naturally, but add an explicit test so we catch regressions.

**Files:**
- Modify: `pad-synth-face/tests/test_bonafide.py`

- [ ] **Step 1: Write the failing/passing test**

Append to `pad-synth-face/tests/test_bonafide.py`:
```python
def test_identity_disjoint_split_all_to_test(fixture_bonafide_dir: Path):
    """When ratios are (0, 0, 1), every identity must land in the test split."""
    loader = DigiFaceLoader(fixture_bonafide_dir)
    train, dev, test = loader.identity_disjoint_split(
        seed=0, ratios=(0.0, 0.0, 1.0)
    )
    assert train == []
    assert dev == []
    assert set(test) == set(loader.list_identities())
    assert len(test) == 8
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_bonafide.py::test_identity_disjoint_split_all_to_test -v 2>&1 | tail -10
```

Expected: PASS immediately (the current implementation should handle this; the test pins the behavior).

- [ ] **Step 3: If the test fails, fix the loader**

If `int(round(n * 0.0))` doesn't yield 0 (it should), update `identity_disjoint_split` in `pad-synth-face/src/pad_synth_face/bonafide.py` to special-case zero ratios. (Most likely no change needed.)

- [ ] **Step 4: Commit**

```bash
git add pad-synth-face/tests/test_bonafide.py
git commit -m "test(face): pin (0,0,1) split-ratio behavior for Phase 1.5 Set B"
```

---

## Task 4: Build extended procedural fixture

New `build_extended_fixture_bonafide()` produces 16 identities × 4 PNG samples each. Each identity has a base skin-tone color drawn from a small Fitzpatrick-inspired palette; each PNG is a 64×64 RGB image with an oval face silhouette (Gaussian falloff) and darker "eye region" patches.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/_fixtures.py`
- Create: `pad-synth-face/tests/test_fixtures.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_fixtures.py`:
```python
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_face._fixtures import (
    build_extended_fixture_bonafide,
    build_fixture_bonafide,
)


def test_extended_fixture_creates_16_identities(tmp_path: Path):
    root = build_extended_fixture_bonafide(tmp_path / "extended")
    identity_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(identity_dirs) == 16


def test_extended_fixture_has_four_samples_per_identity(tmp_path: Path):
    root = build_extended_fixture_bonafide(tmp_path / "extended")
    for identity_dir in root.iterdir():
        if identity_dir.is_dir():
            pngs = list(identity_dir.glob("*.png"))
            assert len(pngs) == 4


def test_extended_fixture_pixel_stats_differ_from_basic(tmp_path: Path):
    """Set A and Set B fixtures must have visibly different pixel distributions
    so the cross-domain eval has actual domain shift to measure."""
    basic_root = build_fixture_bonafide(tmp_path / "basic")
    ext_root = build_extended_fixture_bonafide(tmp_path / "extended")

    def _mean_color(root: Path) -> np.ndarray:
        all_pixels = []
        for png in sorted(root.rglob("*.png")):
            arr = np.array(Image.open(png).convert("RGB"))
            all_pixels.append(arr.reshape(-1, 3))
        stacked = np.concatenate(all_pixels, axis=0)
        return stacked.mean(axis=0)

    basic_mean = _mean_color(basic_root)
    ext_mean = _mean_color(ext_root)
    # Different distributions → mean RGB must differ by at least 10 units in
    # some channel (out of 0-255). This is a coarse but objective check.
    assert np.any(np.abs(basic_mean - ext_mean) > 10)


def test_extended_fixture_is_deterministic(tmp_path: Path):
    """Same call → byte-identical output."""
    import hashlib

    def _hash_tree(root: Path) -> str:
        h = hashlib.sha256()
        for png in sorted(root.rglob("*.png")):
            h.update(png.read_bytes())
        return h.hexdigest()

    a = build_extended_fixture_bonafide(tmp_path / "a")
    b = build_extended_fixture_bonafide(tmp_path / "b")
    assert _hash_tree(a) == _hash_tree(b)


def test_extended_fixture_images_are_64x64_rgb(tmp_path: Path):
    root = build_extended_fixture_bonafide(tmp_path / "extended")
    first = next(root.rglob("*.png"))
    arr = np.array(Image.open(first).convert("RGB"))
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_fixtures.py -v 2>&1 | tail -10
```

Expected: ImportError on `build_extended_fixture_bonafide`.

- [ ] **Step 3: Implement the function**

Append to `pad-synth-face/src/pad_synth_face/_fixtures.py`:
```python
# Fitzpatrick-inspired skin-tone base colors (RGB). Not personally identifying;
# these are abstract palette anchors approximating ranges documented in
# Krishnapriya et al., "Issues Related to Face Recognition Accuracy Varying
# Based on Race and Skin Tone", IEEE Trans. Tech. Soc. 2020.
_SKIN_TONE_PALETTE: list[tuple[int, int, int]] = [
    (244, 219, 196),  # very light
    (224, 192, 165),
    (199, 158, 125),
    (170, 124, 92),
    (133, 90, 60),
    (95, 60, 38),
    (215, 175, 135),  # warm light
    (185, 140, 105),
    (155, 110, 80),
    (120, 85, 60),
    (235, 200, 170),
    (205, 165, 130),
    (175, 130, 95),
    (145, 105, 75),
    (115, 85, 65),
    (90, 65, 45),
]


def _oval_mask(h: int, w: int) -> np.ndarray:
    """Face-shaped Gaussian falloff: 1.0 at center, ~0.3 at the corners."""
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Wider horizontally is less face-shaped; use slight ovalness.
    ry, rx = h * 0.42, w * 0.36
    r = np.sqrt(((yv - cy) / ry) ** 2 + ((xv - cx) / rx) ** 2)
    mask = np.exp(-(r**2) * 1.4)
    return np.clip(mask, 0.3, 1.0)


def _eye_region_darken(h: int, w: int) -> np.ndarray:
    """Darken patches at expected eye y-band (~30-45% from top)."""
    out = np.ones((h, w), dtype=np.float32)
    y_eye_top, y_eye_bot = int(h * 0.30), int(h * 0.45)
    # Left and right eye patches.
    for x_lo, x_hi in [(int(w * 0.22), int(w * 0.40)),
                       (int(w * 0.60), int(w * 0.78))]:
        out[y_eye_top:y_eye_bot, x_lo:x_hi] *= 0.65
    return out


def build_extended_fixture_bonafide(root: Path) -> Path:
    """Phase 1.5 Set B bonafide fixture.

    16 identities × 4 samples each. Each identity has a base skin-tone color
    drawn from a Fitzpatrick-inspired palette. Each image is a 64x64 RGB image
    with an oval face silhouette (Gaussian falloff from center) and darker
    eye-region patches. Per-sample noise gives 4 distinct images per identity.

    This is a procedural fixture for the synthetic cross-domain eval proxy
    — not a substitute for real face data. See LIMITATIONS.md.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)  # different from basic fixture's seed (0)
    oval = _oval_mask(64, 64)
    eye = _eye_region_darken(64, 64)
    for identity in range(16):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir(exist_ok=True)
        base = np.array(_SKIN_TONE_PALETTE[identity], dtype=np.float32)
        for sample in range(4):
            # Background base × oval × eye attenuation, then per-sample noise.
            face = np.tile(base, (64, 64, 1))  # (h, w, 3)
            face = face * oval[:, :, None] * eye[:, :, None]
            # Background outside the oval falls toward neutral grey.
            background = np.full((64, 64, 3), 90.0, dtype=np.float32)
            blend = oval[:, :, None]
            arr = face * blend + background * (1.0 - blend)
            noise = rng.integers(-15, 15, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root
```

Note: the existing imports at the top of `_fixtures.py` already include `numpy as np`, `PIL.Image`, and `Path`. No new imports needed.

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_fixtures.py -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/_fixtures.py pad-synth-face/tests/test_fixtures.py
git commit -m "feat(face): extended procedural bonafide fixture (16 IDs, skin-tone, oval)"
```

---

## Task 5: Add `train_and_cross_domain_eval` to baseline.py

`train_and_eval_tiny_cnn` currently does train + in-domain eval. Phase 1.5 needs a wrapper that can additionally evaluate on a separate held-out dataset (Set B). The existing single-root entry point stays as a thin wrapper for backward compatibility — the integration test from Task 13 of Phase 1 must continue to pass.

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/eval/baseline.py`
- Modify: `pad-synth-core/tests/test_eval_baseline.py`

- [ ] **Step 1: Write the failing tests**

Append to `pad-synth-core/tests/test_eval_baseline.py`:
```python
def test_train_and_cross_domain_eval_in_domain_mode(fixture_pad_dataset_root):
    """When eval_root is None, behavior matches train_and_eval_tiny_cnn."""
    from pad_synth_core.eval.baseline import train_and_cross_domain_eval

    result = train_and_cross_domain_eval(
        train_root=fixture_pad_dataset_root,
        eval_root=None,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert "eer_in_domain" in result
    assert "val_accuracy_in_domain" in result
    assert "n_train" in result
    assert "n_val_in_domain" in result
    # Cross-domain fields should be present but None.
    assert result["eer_cross_domain"] is None
    assert result["n_val_cross_domain"] is None


def test_train_and_cross_domain_eval_with_separate_eval_root(
    fixture_pad_dataset_root, tmp_path
):
    """When eval_root is provided, the result includes cross-domain numbers."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from pad_synth_core.eval.baseline import train_and_cross_domain_eval

    # Build a second tiny PAD-shaped dataset as the cross-domain eval set.
    eval_root = tmp_path / "ds_b"
    (eval_root / "face" / "bonafide").mkdir(parents=True)
    (eval_root / "face" / "print").mkdir(parents=True)
    rng = np.random.default_rng(99)
    for i in range(6):
        b = rng.integers(100, 220, size=(64, 64, 3), dtype=np.uint8)
        a = rng.integers(10, 90, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(b).save(eval_root / "face" / "bonafide" / f"{i}.jpg")
        Image.fromarray(a).save(eval_root / "face" / "print" / f"{i}.jpg")

    result = train_and_cross_domain_eval(
        train_root=fixture_pad_dataset_root,
        eval_root=eval_root,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert result["eer_cross_domain"] is not None
    assert 0.0 <= result["eer_cross_domain"] <= 1.0
    assert result["n_val_cross_domain"] == 12  # 6 bonafide + 6 print
    assert isinstance(result["val_accuracy_cross_domain"], float)


def test_train_and_eval_tiny_cnn_still_works_after_refactor(
    fixture_pad_dataset_root,
):
    """The original entry point must remain functional and return the
    documented field names ('eer', 'val_accuracy', etc.)."""
    from pad_synth_core.eval.baseline import train_and_eval_tiny_cnn

    result = train_and_eval_tiny_cnn(
        dataset_root=fixture_pad_dataset_root,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert "eer" in result
    assert "val_accuracy" in result
    assert "n_train" in result
    assert "n_val" in result
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline.py -v 2>&1 | tail -15
```

Expected: ImportError on `train_and_cross_domain_eval` from at least two of the three new tests; the third (`test_train_and_eval_tiny_cnn_still_works_after_refactor`) should pass as-is since we haven't touched the existing function yet.

- [ ] **Step 3: Implement the cross-domain function**

In `pad-synth-core/src/pad_synth_core/eval/baseline.py`, find the existing `train_and_eval_tiny_cnn` function and REPLACE the entire function with these two:

```python
def train_and_cross_domain_eval(
    train_root: Path,
    eval_root: Path | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a TinyCNN on train_root; eval in-domain (held-out split) and
    optionally cross-domain (full eval_root if provided).

    Returns a dict with keys:
        eer_in_domain (float)
        val_accuracy_in_domain (float)
        n_train (int)
        n_val_in_domain (int)
        eer_cross_domain (float | None)
        val_accuracy_cross_domain (float | None)
        n_val_cross_domain (int | None)
    """
    torch.manual_seed(seed)
    train_ds_full = TinyPADDataset(train_root)
    n_val = max(1, len(train_ds_full) // 4)
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    in_eer, in_acc = _eval_loader(model, val_dl)

    cross_eer: float | None = None
    cross_acc: float | None = None
    n_val_cross: int | None = None
    if eval_root is not None:
        cross_ds = TinyPADDataset(eval_root)
        cross_dl = DataLoader(cross_ds, batch_size=batch_size)
        cross_eer, cross_acc = _eval_loader(model, cross_dl)
        n_val_cross = len(cross_ds)

    return {
        "eer_in_domain": in_eer,
        "val_accuracy_in_domain": in_acc,
        "n_train": n_train,
        "n_val_in_domain": n_val,
        "eer_cross_domain": cross_eer,
        "val_accuracy_cross_domain": cross_acc,
        "n_val_cross_domain": n_val_cross,
    }


def _eval_loader(model: nn.Module, dl: DataLoader) -> tuple[float, float]:
    """Run a model over a dataloader; return (EER, accuracy)."""
    scores: list[float] = []
    labels: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].tolist()
            scores.extend(probs)
            labels.extend(y.tolist())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.numel())
    return compute_eer(scores, labels), correct / max(total, 1)


def train_and_eval_tiny_cnn(
    dataset_root: Path,
    epochs: int = 1,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Backward-compatible wrapper around train_and_cross_domain_eval.

    Returns the original field names: eer, val_accuracy, n_train, n_val.
    """
    full = train_and_cross_domain_eval(
        train_root=dataset_root,
        eval_root=None,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    return {
        "eer": full["eer_in_domain"],
        "val_accuracy": full["val_accuracy_in_domain"],
        "n_train": full["n_train"],
        "n_val": full["n_val_in_domain"],
    }
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline.py -v 2>&1 | tail -15
```

Expected: 6 passed (3 existing + 3 new).

- [ ] **Step 5: Verify no regression in full suite**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```

Expected: all tests still pass (no regressions in pipeline/integration tests that use the old `train_and_eval_tiny_cnn`).

- [ ] **Step 6: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval/baseline.py pad-synth-core/tests/test_eval_baseline.py
git commit -m "feat(eval): add train_and_cross_domain_eval; keep old API as wrapper"
```

---

## Task 6: Create Set B smoke config

**Files:**
- Create: `configs/runs/phase15_setb.yaml`
- Modify: `tests/test_phase15_integration.py` (created in Task 8 — for now just create the config)

- [ ] **Step 1: Create the config file**

Create `configs/runs/phase15_setb.yaml`:
```yaml
run:
  name: phase15_setb
  output: ./datasets/phase15_setb
  seed: 20260512
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

- [ ] **Step 2: Verify config is well-formed YAML**

```bash
.venv/bin/python -c "import yaml; print(yaml.safe_load(open('configs/runs/phase15_setb.yaml')))"
```

Expected: prints a dict matching the config above.

- [ ] **Step 3: Commit**

```bash
git add configs/runs/phase15_setb.yaml
git commit -m "feat(config): Phase 1.5 Set B smoke config (webcam preset, all-test split)"
```

---

## Task 7: Add `eval` CLI subcommand

The existing CLI has a single `generate` subcommand. Phase 1.5 adds `eval` for running `train_and_cross_domain_eval` over two pre-generated datasets.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/cli.py`
- Create: `pad-synth-face/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-face/tests/test_cli.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _build_pad_dataset(root: Path, seed: int, n: int = 6) -> Path:
    """Build a minimal PAD-shaped on-disk dataset for eval testing."""
    (root / "face" / "bonafide").mkdir(parents=True)
    (root / "face" / "print").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        b = rng.integers(100, 220, size=(64, 64, 3), dtype=np.uint8)
        a = rng.integers(10, 90, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(b).save(root / "face" / "bonafide" / f"{i}.jpg")
        Image.fromarray(a).save(root / "face" / "print" / f"{i}.jpg")
    return root


def test_cli_eval_subcommand_runs_in_domain_only(tmp_path: Path):
    train_root = _build_pad_dataset(tmp_path / "train", seed=0, n=8)

    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "eval",
         "--train-root", str(train_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert "eer_in_domain" in output
    assert output["eer_cross_domain"] is None


def test_cli_eval_subcommand_runs_cross_domain(tmp_path: Path):
    train_root = _build_pad_dataset(tmp_path / "train", seed=0, n=8)
    eval_root = _build_pad_dataset(tmp_path / "eval", seed=99, n=6)

    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "eval",
         "--train-root", str(train_root),
         "--eval-root", str(eval_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["eer_in_domain"] is not None
    assert output["eer_cross_domain"] is not None
    assert 0.0 <= output["eer_cross_domain"] <= 1.0
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_cli.py -v 2>&1 | tail -15
```

Expected: nonzero return code from subprocess; `eval` is not a recognized subcommand.

- [ ] **Step 3: Extend the CLI**

Replace the entire contents of `pad-synth-face/src/pad_synth_face/cli.py` with:
```python
"""Minimal Phase-1+1.5 CLI: `pad-synth-face {generate,eval} --config <yaml>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pad_synth_face.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pad-synth-face")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate a synthetic PAD dataset")
    gen.add_argument("--config", required=True, type=Path)

    ev = sub.add_parser(
        "eval",
        help="Train a baseline PAD detector on train-root, optionally evaluate cross-domain on eval-root",
    )
    ev.add_argument("--train-root", required=True, type=Path)
    ev.add_argument("--eval-root", required=False, type=Path, default=None)
    ev.add_argument("--epochs", type=int, default=8)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        summary = run_pipeline(args.config)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "eval":
        # Lazy import — torch is heavy and only required for eval.
        from pad_synth_core.eval.baseline import train_and_cross_domain_eval

        result = train_and_cross_domain_eval(
            train_root=args.train_root,
            eval_root=args.eval_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_cli.py -v 2>&1 | tail -15
```

Expected: 2 passed.

- [ ] **Step 5: Smoke-check via subprocess**

```bash
.venv/bin/python -m pad_synth_face.cli --help 2>&1 | tail -10
.venv/bin/python -m pad_synth_face.cli eval --help 2>&1 | tail -10
```

Expected: both print help text mentioning both `generate` and `eval` subcommands.

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/cli.py pad-synth-face/tests/test_cli.py
git commit -m "feat(cli): add 'eval' subcommand for cross-domain detector eval"
```

---

## Task 8: End-to-end Phase 1.5 integration test

Builds the extended fixture, generates Set A and Set B via the CLI, then runs cross-domain eval and asserts the result contains the expected keys with sensible values.

**Files:**
- Create: `tests/test_phase15_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_phase15_integration.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import yaml

from pad_synth_face._fixtures import (
    build_extended_fixture_bonafide,
    build_fixture_bonafide,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config))


def _generate(cfg_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "generate",
         "--config", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_phase15_cross_domain_eval_end_to_end(tmp_path: Path):
    # --- Build both fixtures ---
    set_a_fixture = build_fixture_bonafide(tmp_path / "fixture_a")
    set_b_fixture = build_extended_fixture_bonafide(tmp_path / "fixture_b")

    # --- Generate Set A (mobile-front sensor) ---
    set_a_config = {
        "run": {"name": "set_a", "output": str(tmp_path / "set_a"),
                "seed": 20260511, "deterministic": True},
        "modality": "face",
        "bonafide": {
            "root": str(set_a_fixture),
            "samples_per_bonafide": 4,
            "splits": {"train": 0.5, "dev": 0.25, "test": 0.25},
        },
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
            "replay": {"weight": 1.0,
                       "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    set_a_cfg = tmp_path / "set_a.yaml"
    _write_config(set_a_cfg, set_a_config)
    summary_a = _generate(set_a_cfg)
    # 8 IDs × 4 samples each, separately for bonafide and attacks
    assert summary_a["samples_generated"] == 32
    assert summary_a["bonafide_emitted"] == 32

    # --- Generate Set B (webcam-1080p sensor, all-test split, extended fixture) ---
    set_b_config = {
        "run": {"name": "set_b", "output": str(tmp_path / "set_b"),
                "seed": 20260512, "deterministic": True},
        "modality": "face",
        "bonafide": {
            "root": str(set_b_fixture),
            "samples_per_bonafide": 4,
            "splits": {"train": 0.0, "dev": 0.0, "test": 1.0},
        },
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
            "replay": {"weight": 1.0,
                       "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml")},
        },
        "sensor_preset": "webcam-1080p",
    }
    set_b_cfg = tmp_path / "set_b.yaml"
    _write_config(set_b_cfg, set_b_config)
    summary_b = _generate(set_b_cfg)
    # 16 IDs × 4 samples
    assert summary_b["samples_generated"] == 64
    assert summary_b["bonafide_emitted"] == 64

    # --- Cross-domain eval ---
    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "eval",
         "--train-root", str(tmp_path / "set_a"),
         "--eval-root", str(tmp_path / "set_b"),
         "--epochs", "5",
         "--seed", "0"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["eer_in_domain"] is not None
    assert out["eer_cross_domain"] is not None
    assert 0.0 <= out["eer_in_domain"] <= 1.0
    assert 0.0 <= out["eer_cross_domain"] <= 1.0
    assert out["n_val_cross_domain"] == 128  # 64 bonafide + 64 attack in Set B
```

- [ ] **Step 2: Run the integration test**

```bash
.venv/bin/python -m pytest tests/test_phase15_integration.py -v 2>&1 | tail -20
```

Expected: 1 passed (may take 10-30 seconds due to the end-to-end generation + training).

- [ ] **Step 3: Run the full suite**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase15_integration.py
git commit -m "test: Phase 1.5 end-to-end cross-domain eval integration"
```

---

## Task 9: Write LIMITATIONS.md

Document explicitly that Set B is a synthetic-to-synthetic proxy.

**Files:**
- Create: `LIMITATIONS.md` (at repo root)

- [ ] **Step 1: Write the LIMITATIONS file**

Create `LIMITATIONS.md`:
```markdown
# Limitations

## Phase 1.5 cross-domain eval is a synthetic-to-synthetic proxy

The cross-domain EER reported by `pad-synth-face eval --train-root A --eval-root B`
is a synthetic-to-synthetic generalization test, not a synthetic-to-real-world test.

### What it does measure

- Whether the PAD detector overfits to a specific synthetic distribution
- Robustness to bonafide-source and sensor-preset variation, holding attack
  ontologies stable
- A floor indicator: if cross-domain synthetic generalization fails, real-world
  performance will also fail. If it succeeds, real-world performance is undetermined.

### What it does NOT measure

- Real-world deployment performance against actual print/replay attacks
- Generalization to attack types not present in either set (mask, deepfake)
- Generalization to capture devices not modeled by `mobile-front-2024` or
  `webcam-1080p` sensor presets
- Any production claim about detector quality

### Phase 2 follow-on

Real PAD dataset integration (CelebA-Spoof, MSU-MFSD, Idiap Replay-Attack) is
tracked as a Phase 2 sub-task. Until that lands, treat the cross-domain EER as
a directional signal, not a benchmark number.

### Other current limitations

- Only 2 of 4 face attack types implemented (print, replay; mask and deepfake
  are Phase 2 work)
- No voice modality (Phase 3)
- Bonafide source is procedural; real face data integration is Phase 2
- Datasets are small (under 200 samples in any single config)
- Phase 1 simulator uses simplified physics — full halftoning, ICC profiling,
  per-device subpixel models, and refresh-rate banding are Phase 2 work
```

- [ ] **Step 2: Commit**

```bash
git add LIMITATIONS.md
git commit -m "docs: LIMITATIONS.md documenting Phase 1.5 cross-domain eval as a synthetic proxy"
```

---

## Final verification

After all tasks land, run the full suite and exercise the end-to-end CLI flow manually.

- [ ] **Step 1: Full suite passes**

```bash
cd /Users/stuartwells/test
.venv/bin/python -m pytest -q 2>&1 | tail -5
```

Expected: all tests pass (target: 55 + 2 from Task 1 + 1 from Task 2 + 1 from Task 3 + 5 from Task 4 + 3 from Task 5 + 2 from Task 7 + 1 from Task 8 = ~70 passed).

- [ ] **Step 2: Manual end-to-end flow**

```bash
# Build both fixtures (uses build_fixture_bonafide for Set A, build_extended_fixture_bonafide for Set B)
.venv/bin/python -c "
from pathlib import Path
from pad_synth_face._fixtures import build_fixture_bonafide, build_extended_fixture_bonafide
build_fixture_bonafide(Path('datasets/_fixtures/digiface'))
build_extended_fixture_bonafide(Path('datasets/_fixtures/extended_fixture'))
print('Fixtures built.')
"

# Generate Set A using the existing smoke config
.venv/bin/python -m pad_synth_face.cli generate --config configs/runs/phase1_smoke.yaml | tail -10

# Generate Set B using the new Phase 1.5 config
.venv/bin/python -m pad_synth_face.cli generate --config configs/runs/phase15_setb.yaml | tail -10

# Run cross-domain eval
.venv/bin/python -m pad_synth_face.cli eval \
    --train-root datasets/phase1_smoke \
    --eval-root datasets/phase15_setb \
    --epochs 10 --seed 0 | tail -15
```

Expected: prints a JSON dict with both `eer_in_domain` and `eer_cross_domain` populated. Record the numbers and use them with the spec §7 decision matrix to set Phase 2's direction.

- [ ] **Step 3: Append result to history log**

```bash
mkdir -p datasets/phase1_smoke/qc/cross_domain_eval
.venv/bin/python -m pad_synth_face.cli eval \
    --train-root datasets/phase1_smoke \
    --eval-root datasets/phase15_setb \
    --epochs 10 --seed 0 \
    > datasets/phase1_smoke/qc/cross_domain_eval/$(date -u +%Y%m%d-%H%M%SZ).json
```

This records each cross-domain eval run with a UTC timestamp for future trend tracking.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2.1 Goals: Set B different bonafide + sensor | Tasks 4 (fixture) + 1 (sensor) |
| §2.1 Goals: `webcam-1080p` preset | Task 1 |
| §2.1 Goals: `train_and_cross_domain_eval` entry point | Task 5 |
| §2.1 Goals: Log + LIMITATIONS.md | Task 9, Final Verification Step 3 |
| §2.3 All 55 existing tests pass | Verified after every task |
| §3 Architecture (file structure) | All file paths match the spec |
| §4.1 WEBCAM_1080P with exact values | Task 1 Step 3 (exact match) |
| §4.2 build_extended_fixture_bonafide | Task 4 |
| §4.3 train_and_cross_domain_eval signature | Task 5 Step 3 |
| §4.4 Set B smoke config | Task 6 |
| §4.5 CLI eval subcommand | Task 7 |
| §5 Data flow | Final Verification Step 2 |
| §6 Tests: all enumerated tests | Task-by-task (Tasks 1, 3, 4, 5, 7, 8) |
| §6 Tests: determinism golden still passes | Implicitly verified — Set A pipeline unchanged |
| §7 Decision matrix used after Final Verification | User responsibility post-execution |
| §8 Risks: (0,0,1) ratio edge case | Task 3 |
| §10 LIMITATIONS.md | Task 9 |

All spec sections covered.

**Placeholder scan:** No "TBD", "TODO", "implement later", or "similar to Task N" without code. Every step that changes code shows the complete code.

**Type consistency:**
- `SensorPreset` fields match across Task 1, Task 2, and the existing sensor.py.
- `train_and_cross_domain_eval` returns `dict[str, Any]` with documented keys; the wrapper `train_and_eval_tiny_cnn` maps those keys to the original API.
- `BonafideSample` and `DigiFaceLoader` are unchanged; both fixtures emit the same directory layout.
- `_FIXED_IMAGE_SHAPE = (64, 64, 3)` from Phase 1 is preserved; Task 4's images are 64×64×3 to match.
- The integration test asserts `n_val_cross_domain == 128` based on `samples_per_bonafide=4 × 16 identities × 2 (bonafide + attack)`. Verified.

No issues found.
