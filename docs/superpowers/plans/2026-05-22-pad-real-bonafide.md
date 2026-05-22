# PAD Real-Bonafide Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Microsoft Research's DigiFace-1M 118k aligned subset as a bonafide source for the PAD pipeline, regenerate the existing 27-cell sweep on real-bonafide datasets with v2.1 print physics, and write the verdict on whether real-face textures break the v2 / v2.1 binary-halftone-palette artifact.

**Architecture:** Two small additive changes (DigiFaceLoader gains `restrict_to`; pipeline.py reads dynamic `ontology_version` and a new `bonafide.identities_file`). A few new scripts handle download, resize, and identity-list selection. Six new configs point the existing pipeline at the prepared real-bonafide directory. The Spark sweep is identical to v2.1 — only `bonafide.root` and the identities file change.

**Tech Stack:** Python 3.11+ (laptop) / 3.12 (Spark), numpy, Pillow, PyTorch nightly cu128 (Spark), pytest. No new external dependencies. DigiFace-1M 118k subset (MIT, Microsoft Research).

---

## Reference: facts the engineer needs

**Current `bonafide.py`** (`pad-synth-face/src/pad_synth_face/bonafide.py`) has `class DigiFaceLoader(self, root)` with methods `list_identities()`, `samples_for_identity(identity) -> list[BonafideSample]` (currently globs only `*.png`), `load(sample) -> np.ndarray`, `identity_disjoint_split(seed, ratios) -> (train, dev, test)`. The class is reused by both the fixture (PNGs) and the planned DigiFace path.

**Current `pipeline.py`** lines 181 and 238 hardcode `ontology_version="2026-05-11"` in the bonafide and attack `SampleRecord` constructions respectively. The loaded attack modules (each with `.ontology.version`) are in scope at both sites. The pipeline instantiates `loader = DigiFaceLoader(Path(cfg["bonafide"]["root"]))` — no `restrict_to` yet.

**`tests/golden/golden_hashes.json`** stores `sample_id → output_sha256` for 32 entries. The hashes are sha256 of the output JPEG bytes, NOT of the manifest. The manifest's `ontology_version` field has no effect on JPEG bytes; the hardcode fix should therefore NOT require golden regeneration.

**Existing tests** that must continue passing unchanged:
- `pad-synth-face/tests/test_bonafide.py` (DigiFaceLoader basics)
- `pad-synth-face/tests/test_pipeline_e2e.py`
- `pad-synth-face/tests/test_print_attack.py`, `test_print_v2_integration.py`, `test_print_halftone.py`, `test_print_halftone_jitter.py`, `test_print_icc.py`
- `tests/test_determinism_golden.py`

**Spec.** `docs/superpowers/specs/2026-05-22-pad-real-bonafide-design.md`. §3 dataset choice; §4 resize-to-64; §5 identity selection; §6 configs; §7 loader/pipeline extensions; §8 measurement plan; §12 operational risk on download auth.

**Operational risk (spec §12):** DigiFace-1M may require Microsoft account auth / terms-of-use agreement. T3's download task surfaces this; if blocked, the user fetches manually and we resume from a pre-staged directory.

---

## Task 1: DigiFaceLoader gains `restrict_to` + glob both `.png` and `.jpg`

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/bonafide.py`
- Create: `pad-synth-face/tests/test_bonafide_restrict.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_bonafide_restrict.py`:
```python
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pad_synth_face.bonafide import DigiFaceLoader


def _seed_real_layout(root: Path, identities: int, samples: int, ext: str) -> Path:
    """Build a DigiFace-shaped fake dataset: <root>/<id>/<i>.{png|jpg}."""
    for i in range(identities):
        d = root / f"{i:08d}"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(samples):
            arr = (np.random.default_rng(i * 100 + s).random((16, 16, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{s:03d}.{ext}", format="PNG" if ext == "png" else "JPEG")
    return root


def test_loader_default_lists_all_identities(tmp_path):
    root = _seed_real_layout(tmp_path / "src", identities=5, samples=2, ext="png")
    loader = DigiFaceLoader(root)
    assert loader.list_identities() == [f"{i:08d}" for i in range(5)]


def test_restrict_to_filters_identities(tmp_path):
    root = _seed_real_layout(tmp_path / "src", identities=5, samples=2, ext="png")
    loader = DigiFaceLoader(root, restrict_to=["00000001", "00000003"])
    assert loader.list_identities() == ["00000001", "00000003"]


def test_restrict_to_intersects_with_on_disk(tmp_path):
    """Identities in restrict_to but not on disk are silently dropped."""
    root = _seed_real_layout(tmp_path / "src", identities=3, samples=1, ext="png")
    loader = DigiFaceLoader(root, restrict_to=["00000001", "99999999"])
    assert loader.list_identities() == ["00000001"]


def test_glob_picks_up_jpg_files(tmp_path):
    """Real DigiFace may ship .jpg; loader must find them."""
    root = _seed_real_layout(tmp_path / "src", identities=2, samples=3, ext="jpg")
    loader = DigiFaceLoader(root)
    samples = loader.samples_for_identity("00000000")
    assert len(samples) == 3
    assert all(s.path.suffix == ".jpg" for s in samples)


def test_glob_mixes_png_and_jpg(tmp_path):
    """A directory containing both extensions should yield both."""
    base = tmp_path / "src" / "00000000"
    base.mkdir(parents=True)
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(base / "a.png")
    Image.fromarray(arr).save(base / "b.jpg")
    loader = DigiFaceLoader(tmp_path / "src")
    samples = loader.samples_for_identity("00000000")
    assert len(samples) == 2
    assert {s.path.suffix for s in samples} == {".png", ".jpg"}


def test_no_restrict_argument_preserves_v1_default(tmp_path):
    """Backwards compat: positional-only root still works."""
    root = _seed_real_layout(tmp_path / "src", identities=3, samples=1, ext="png")
    loader = DigiFaceLoader(root)
    assert loader.list_identities() == ["00000000", "00000001", "00000002"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/stuartwells/test && .venv/bin/python -m pytest pad-synth-face/tests/test_bonafide_restrict.py -q 2>&1 | tail -8`
Expected: at least `test_restrict_to_filters_identities`, `test_restrict_to_intersects_with_on_disk`, and `test_glob_picks_up_jpg_files` fail. The default-listing tests may pass.

- [ ] **Step 3: Extend `DigiFaceLoader`**

In `pad-synth-face/src/pad_synth_face/bonafide.py`, replace the `class DigiFaceLoader:` definition (keeping the `BonafideSample` dataclass and the file's docstring/imports unchanged) with:

```python
class DigiFaceLoader:
    def __init__(self, root: Path, restrict_to: list[str] | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._restrict_to: set[str] | None = (
            set(restrict_to) if restrict_to is not None else None
        )

    def list_identities(self) -> list[str]:
        ids = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if self._restrict_to is not None:
            ids = [i for i in ids if i in self._restrict_to]
        return ids

    def samples_for_identity(self, identity: str) -> list[BonafideSample]:
        identity_dir = self.root / identity
        return [
            BonafideSample(identity=identity, path=p)
            for p in sorted(identity_dir.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

    def load(self, sample: BonafideSample) -> np.ndarray:
        img = Image.open(sample.path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def identity_disjoint_split(
        self, seed: int, ratios: tuple[float, float, float]
    ) -> tuple[list[str], list[str], list[str]]:
        ids = self.list_identities()
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids)).tolist()
        shuffled = [ids[i] for i in order]
        n = len(shuffled)
        n_train = int(round(n * ratios[0]))
        n_dev = int(round(n * ratios[1]))
        train = shuffled[:n_train]
        dev = shuffled[n_train : n_train + n_dev]
        test = shuffled[n_train + n_dev :]
        return train, dev, test
```

- [ ] **Step 4: Run to verify new tests pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_bonafide_restrict.py -q 2>&1 | tail -5`
Expected: 6 passed.

- [ ] **Step 5: Run the existing bonafide tests — they must still pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_bonafide.py -q 2>&1 | tail -3`
Expected: all pass. (The default behavior — no `restrict_to`, sorted identity listing, PNG samples — is unchanged.)

- [ ] **Step 6: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 166 passed, 1 skipped (160 prior + 6 new).

- [ ] **Step 7: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/bonafide.py pad-synth-face/tests/test_bonafide_restrict.py
git commit -m "feat(pad-face): DigiFaceLoader gains restrict_to and jpg/png glob"
```

---

## Task 2: Pipeline reads dynamic `ontology_version` and `identities_file`

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/pipeline.py`
- Create: `pad-synth-face/tests/test_pipeline_dynamic_ontology_version.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_pipeline_dynamic_ontology_version.py`:
```python
import json
from pathlib import Path

import yaml

from pad_synth_face._fixtures import build_fixture_bonafide
from pad_synth_face.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]


def test_manifest_records_dynamic_ontology_version(tmp_path: Path):
    """The manifest's ontology_version must match the loaded print ontology's
    version (currently 2026-05-23 post v2.1), not a hardcoded string."""
    fixture_root = build_fixture_bonafide(tmp_path / "fixture")
    config = {
        "run": {
            "name": "dyn_ver_test",
            "output": str(tmp_path / "out"),
            "seed": 20260522,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_root), "samples_per_bonafide": 1},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO / "ontology" / "face" / "replay.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_pipeline(cfg_path)

    # Read the actual print ontology version from disk to compare against.
    print_ont_yaml = yaml.safe_load((REPO / "ontology" / "face" / "print.yaml").read_text())
    expected_version = print_ont_yaml["version"]
    assert expected_version != "2026-05-11", (
        "test guard: print ontology has been bumped past v1; "
        "if you see this, the v2/v2.1 bumps were reverted somehow"
    )

    manifest = (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    assert manifest, "manifest is empty"
    for line in manifest:
        rec = json.loads(line)
        assert rec["ontology_version"] == expected_version, (
            f"sample {rec['sample_id']} has ontology_version={rec['ontology_version']!r}, "
            f"expected {expected_version!r}"
        )


def test_pipeline_honors_bonafide_identities_file(tmp_path: Path):
    """When bonafide.identities_file is set, the pipeline restricts iteration
    to those identities."""
    fixture_root = build_fixture_bonafide(tmp_path / "fixture")
    # Pick 2 of the 8 fixture identities.
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("00000002\n00000005\n")
    config = {
        "run": {
            "name": "restrict_test",
            "output": str(tmp_path / "out"),
            "seed": 20260522,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_root),
            "samples_per_bonafide": 2,
            "identities_file": str(ids_file),
        },
        "attacks": {
            "print": {"weight": 1.0, "ontology": str(REPO / "ontology" / "face" / "print.yaml")},
            "replay": {"weight": 1.0, "ontology": str(REPO / "ontology" / "face" / "replay.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_pipeline(cfg_path)

    manifest = (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    sources = {json.loads(line)["bonafide_source"]["id"] for line in manifest}
    assert sources == {"00000002", "00000005"}, f"expected only those 2 IDs, got {sources}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_dynamic_ontology_version.py -q 2>&1 | tail -5`
Expected: both fail. `test_manifest_records_dynamic_ontology_version` fails because the hardcode stamps `2026-05-11`; `test_pipeline_honors_bonafide_identities_file` fails because `identities_file` is ignored.

- [ ] **Step 3: Modify `pipeline.py`**

In `pad-synth-face/src/pad_synth_face/pipeline.py`, locate the section where `loader` is constructed and `attack_modules` is built. The current line `loader = DigiFaceLoader(Path(cfg["bonafide"]["root"]))` becomes:

```python
        _bonafide_cfg = cfg["bonafide"]
        if "identities_file" in _bonafide_cfg:
            _ids_path = Path(_bonafide_cfg["identities_file"])
            _restrict = [
                line.strip() for line in _ids_path.read_text().splitlines() if line.strip()
            ]
            loader = DigiFaceLoader(Path(_bonafide_cfg["root"]), restrict_to=_restrict)
        else:
            loader = DigiFaceLoader(Path(_bonafide_cfg["root"]))
```

Then, AFTER `attack_modules` is built (the line `attack_modules = {name: _ATTACK_REGISTRY[name](...)}` etc.) and BEFORE the first `SampleRecord` construction at line 168, insert a single canonical-version assignment:

```python
        # Single canonical ontology_version for all sample records in this run.
        # Bonafide records share the print attack's version since bonafide has
        # no attack ontology of its own; the print ontology is the dominant
        # version-tracked component of the dataset.
        _ontology_version = attack_modules["print"].ontology.version
```

Finally, find BOTH occurrences of `ontology_version="2026-05-11",` (currently at lines 181 and 238) and replace each with:

```python
                    ontology_version=_ontology_version,
```

(Indentation: match the surrounding SampleRecord field indentation — currently 20 spaces of leading indent. Use the same.)

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_dynamic_ontology_version.py -q 2>&1 | tail -5`
Expected: 2 passed.

- [ ] **Step 5: Run the existing pipeline-touching tests — they must still pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_e2e.py pad-synth-face/tests/test_bonafide.py pad-synth-face/tests/test_print_attack.py pad-synth-face/tests/test_print_v2_integration.py pad-synth-face/tests/test_print_halftone.py pad-synth-face/tests/test_print_halftone_jitter.py pad-synth-face/tests/test_print_icc.py -q 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 6: The determinism golden should also still pass**

The pipeline.py change affects the manifest's `ontology_version` field but NOT the output JPEG bytes (the golden hashes JPEGs, not the manifest). Run:
```bash
.venv/bin/python -m pytest tests/test_determinism_golden.py -q 2>&1 | tail -3
```
Expected: 1 passed. If it FAILS, STOP and report DONE_WITH_CONCERNS — the assumption is wrong and we need to understand which bytes shifted before regenerating.

- [ ] **Step 7: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 168 passed, 1 skipped (166 prior + 2 new).

- [ ] **Step 8: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/pipeline.py pad-synth-face/tests/test_pipeline_dynamic_ontology_version.py
git commit -m "feat(pad-face): pipeline reads dynamic ontology_version + identities_file"
```

---

## Task 3: Download DigiFace-1M 118k subset (OPERATIONAL — may BLOCK)

**Files:**
- Create: `scripts/fetch_digiface_subset.sh`

- [ ] **Step 1: Create the download script**

`scripts/fetch_digiface_subset.sh`:
```bash
#!/usr/bin/env bash
# Fetch the Microsoft Research DigiFace-1M 118k aligned subset.
# Outputs to ./datasets/_real/digiface_118k_raw/ (gitignored).
# Idempotent: skips if already present.
set -euo pipefail

OUT_DIR="datasets/_real/digiface_118k_raw"
mkdir -p "$OUT_DIR"

# Marker file indicates successful previous download.
MARKER="$OUT_DIR/.fetch_complete"
if [[ -f "$MARKER" ]]; then
  echo "DigiFace 118k subset already present at $OUT_DIR (marker exists)."
  exit 0
fi

# DigiFace-1M is hosted by Microsoft Research. The 118k aligned subset URL
# may require an MS account or terms-of-use agreement. The implementer
# resolves the exact URL at execution time. If the resource requires
# interactive auth, this script should print clear guidance and exit non-zero
# so the operational task can fall back to a pre-staged directory.

# Resolution path:
# 1. Try the canonical Microsoft Research DigiFace download URL.
# 2. If that requires browser-based auth, instruct the user.

cat <<EOF
This script attempts to download the DigiFace-1M 118k aligned subset
to $OUT_DIR.

DigiFace-1M is published at: https://microsoft.github.io/DigiFace1M/
The 118k aligned subset is the smaller release suitable for our 27-cell sweep
(~few-hundred MB compared to the ~6GB full release).

If automated download fails because the host requires browser auth or a
terms-of-use click-through, please:
  1. Visit the DigiFace-1M page and download the 118k aligned subset.
  2. Extract to: $OUT_DIR/
  3. Confirm layout: each identity is a subdirectory named like '00000001'
     containing PNG or JPG files named like '0.png', '1.png', etc.
  4. touch $MARKER  to skip this script on re-runs.

Attempting automated download...
EOF

# Attempt the download. The exact URL is filled in at execution time by
# checking the Microsoft Research DigiFace page; if no public direct-download
# URL is available, this falls through to the failure message above.

# Placeholder for the actual curl/wget invocation; the implementer fills in
# the resolved URL at execution time.
echo "WARNING: actual download not yet implemented in this script."
echo "Please follow the manual instructions above and re-run."
exit 2
EOF
```

Mark executable:
```bash
chmod +x scripts/fetch_digiface_subset.sh
```

- [ ] **Step 2: Attempt the download**

```bash
cd /Users/stuartwells/test
./scripts/fetch_digiface_subset.sh
```

**This step is expected to fail.** Reasons:
- If Microsoft Research has a direct-download URL: update the script to call `curl -L -o "$OUT_DIR/digiface_118k.zip" <URL>` and re-run.
- If it requires browser auth: the user must download manually. Provide them this guidance:
  1. Visit https://microsoft.github.io/DigiFace1M/
  2. Download the 118k aligned subset.
  3. Extract to `/Users/stuartwells/test/datasets/_real/digiface_118k_raw/`.
  4. Confirm directory structure: `<root>/<identity_id>/<sample>.png` (or `.jpg`).
  5. `touch /Users/stuartwells/test/datasets/_real/digiface_118k_raw/.fetch_complete`

If the user needs to fetch manually, **REPORT BLOCKED** with the exact manual-steps message above and wait for confirmation that the directory is staged.

- [ ] **Step 3: Verify the on-disk layout**

After the download (automated or manual) completes, run:
```bash
test -d datasets/_real/digiface_118k_raw && \
  echo "identities: $(find datasets/_real/digiface_118k_raw -mindepth 1 -maxdepth 1 -type d | wc -l)" && \
  echo "first id's samples: $(find datasets/_real/digiface_118k_raw -mindepth 2 -maxdepth 2 -type f | head -5)"
```
Expected: identities count is ≥ 100 (the 118k subset has many more, around 10k); each identity dir contains image files (`.png` or `.jpg`).

- [ ] **Step 4: Commit the script** (regardless of whether download succeeded)

```bash
git add scripts/fetch_digiface_subset.sh
git commit -m "feat(pad-spark): DigiFace-1M 118k fetch script (manual fallback documented)"
```

The actual dataset is gitignored (it lives under `datasets/`).

---

## Task 4: Preprocess to 64×64

**Files:**
- Create: `scripts/prepare_digiface_64.py`
- Create: `pad-synth-face/tests/test_prepare_digiface.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_prepare_digiface.py`:
```python
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def _seed_raw(root: Path, identities: int, samples: int, size: int) -> Path:
    for i in range(identities):
        d = root / f"{i:08d}"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(samples):
            arr = (np.random.default_rng(i * 100 + s).random((size, size, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{s:03d}.png")
    return root


def test_prepare_digiface_resizes_to_64_and_preserves_layout(tmp_path):
    src = _seed_raw(tmp_path / "raw", identities=3, samples=4, size=112)
    dst = tmp_path / "out_64"

    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_digiface_64.py"),
         "--src", str(src), "--dst", str(dst)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr

    # Verify layout preserved.
    ids = sorted(p.name for p in dst.iterdir() if p.is_dir())
    assert ids == ["00000000", "00000001", "00000002"]
    for i in ids:
        samples = sorted((dst / i).glob("*.png"))
        assert len(samples) == 4
        for s in samples:
            with Image.open(s) as im:
                assert im.size == (64, 64), f"{s} is {im.size}, expected (64, 64)"

    # _meta.json present and well-formed.
    import json
    meta = json.loads((dst / "_meta.json").read_text())
    assert meta["target_size"] == 64
    assert meta["identities"] == 3
    assert meta["samples_total"] == 12


def test_prepare_digiface_is_idempotent(tmp_path):
    """Re-running on an existing dst dir should be a no-op (skip already-done)."""
    src = _seed_raw(tmp_path / "raw", identities=2, samples=2, size=112)
    dst = tmp_path / "out_64"

    r1 = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_digiface_64.py"),
         "--src", str(src), "--dst", str(dst)],
        capture_output=True, text=True, check=False,
    )
    assert r1.returncode == 0

    # Touch the destination's mtimes; second run should not overwrite.
    sample_path = dst / "00000000" / "000.png"
    original_mtime = sample_path.stat().st_mtime

    r2 = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_digiface_64.py"),
         "--src", str(src), "--dst", str(dst)],
        capture_output=True, text=True, check=False,
    )
    assert r2.returncode == 0
    assert sample_path.stat().st_mtime == original_mtime, "idempotent: should not re-write existing files"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_prepare_digiface.py -q 2>&1 | tail -5`
Expected: 2 failed (script missing).

- [ ] **Step 3: Create the preprocessing script**

`scripts/prepare_digiface_64.py`:
```python
#!/usr/bin/env python3
"""Resize DigiFace-1M images to 64x64, preserving <root>/<id>/<sample> layout.

Idempotent: skips files that already exist at the destination. Uses PIL's
LANCZOS resampling for quality. Writes a `_meta.json` recording the
operation summary (counts, target size).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TARGET_SIZE = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source DigiFace root: <src>/<identity>/<sample>.{png,jpg}")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Destination root for resized images")
    args = ap.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst
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
            # Always write as PNG for consistency at destination.
            out_path = out_dir / f"{sample_path.stem}.png"
            if out_path.exists():
                n_skipped += 1
                continue
            with Image.open(sample_path) as im:
                im = im.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                im.save(out_path, format="PNG")
            n_samples += 1

    meta = {
        "target_size": TARGET_SIZE,
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

Mark executable: `chmod +x scripts/prepare_digiface_64.py`.

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_prepare_digiface.py -q 2>&1 | tail -5`
Expected: 2 passed.

- [ ] **Step 5: Run the script on the real downloaded data**

```bash
cd /Users/stuartwells/test
.venv/bin/python scripts/prepare_digiface_64.py \
  --src datasets/_real/digiface_118k_raw \
  --dst datasets/_real/digiface_118k_64
```

Expected: prints the `_meta.json` summary with `target_size: 64`, `identities` count matching the source (likely ~10k for the 118k subset), and a non-zero `samples_written`. Wall-time: a few minutes for ~118k images.

If this step is skipped because T3 was BLOCKED, stop here and report BLOCKED on this task too.

- [ ] **Step 6: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 170 passed, 1 skipped (168 prior + 2 new).

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_digiface_64.py pad-synth-face/tests/test_prepare_digiface.py
git commit -m "feat(pad-spark): prepare_digiface_64.py (112x112 -> 64x64 resize)"
```

The actual resized data is gitignored.

---

## Task 5: Select 8/16 identities and commit the pinned lists

**Files:**
- Create: `scripts/select_digiface_identities.py`
- Create (committed): `configs/digiface_identities_seta.txt`, `configs/digiface_identities_setb.txt`

- [ ] **Step 1: Create the selection script**

`scripts/select_digiface_identities.py`:
```python
#!/usr/bin/env python3
"""Deterministically select 8 identities for Set A and 16 disjoint identities
for Set B from a DigiFace-1M root directory. Writes the two committed text
files. Seeded by the master Set A/B seeds (20260522 / 20260523)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="DigiFace root (resized 64x64 dir)")
    ap.add_argument("--seta-out", type=Path,
                    default=REPO / "configs" / "digiface_identities_seta.txt")
    ap.add_argument("--setb-out", type=Path,
                    default=REPO / "configs" / "digiface_identities_setb.txt")
    ap.add_argument("--seta-count", type=int, default=8)
    ap.add_argument("--setb-count", type=int, default=16)
    args = ap.parse_args()

    # Enumerate identities (sorted; the dataset directory listing is stable
    # because each ID is its own subdir).
    all_ids = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    assert len(all_ids) >= args.seta_count + args.setb_count, (
        f"need at least {args.seta_count + args.setb_count} identities; "
        f"found {len(all_ids)}"
    )

    # Seeded permutation, then carve disjoint slices.
    rng = np.random.default_rng(20260522)
    order = rng.permutation(len(all_ids)).tolist()
    shuffled = [all_ids[i] for i in order]
    seta = sorted(shuffled[: args.seta_count])
    setb = sorted(shuffled[args.seta_count : args.seta_count + args.setb_count])

    args.seta_out.write_text("\n".join(seta) + "\n")
    args.setb_out.write_text("\n".join(setb) + "\n")

    print(f"Wrote {len(seta)} Set A identities to {args.seta_out}")
    print(f"Wrote {len(setb)} Set B identities to {args.setb_out}")
    print(f"Set A: {seta}")
    print(f"Set B: {setb}")


if __name__ == "__main__":
    main()
```

Mark executable: `chmod +x scripts/select_digiface_identities.py`.

- [ ] **Step 2: Run it against the resized DigiFace root**

```bash
cd /Users/stuartwells/test
.venv/bin/python scripts/select_digiface_identities.py \
  --root datasets/_real/digiface_118k_64
```

Expected: writes `configs/digiface_identities_seta.txt` (8 lines, one identity name per line) and `configs/digiface_identities_setb.txt` (16 lines). Prints both lists for inspection.

- [ ] **Step 3: Verify the lists are disjoint**

```bash
comm -12 \
  <(sort configs/digiface_identities_seta.txt) \
  <(sort configs/digiface_identities_setb.txt)
```
Expected: empty output (no common identities). If any output appears, the selection has a bug.

- [ ] **Step 4: Verify the listed identities actually exist on disk**

```bash
for id in $(cat configs/digiface_identities_seta.txt configs/digiface_identities_setb.txt); do
  test -d "datasets/_real/digiface_118k_64/$id" || echo "MISSING: $id"
done
```
Expected: no `MISSING` output. All 24 identity directories should exist.

- [ ] **Step 5: Commit the script + the two pinned lists**

```bash
git add scripts/select_digiface_identities.py configs/digiface_identities_seta.txt configs/digiface_identities_setb.txt
git commit -m "feat(pad-spark): pin 8/16 DigiFace identities for Set A/B"
```

---

## Task 6: Six real-bonafide configs + validation test

**Files:**
- Create: `configs/runs/real_seta_d1.yaml`, `real_seta_d2.yaml`, `real_seta_d3.yaml`
- Create: `configs/runs/real_setb_d1.yaml`, `real_setb_d2.yaml`, `real_setb_d3.yaml`
- Create: `pad-synth-face/tests/test_real_configs.py`

- [ ] **Step 1: Write the failing test**

`pad-synth-face/tests/test_real_configs.py`:
```python
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "configs" / "runs"

EXPECTED = {
    "real_seta_d1.yaml": (20260522, "mobile-front-2024", "./datasets/_real/digiface_118k_64",
                          6, "./configs/digiface_identities_seta.txt"),
    "real_seta_d2.yaml": (20260522, "mobile-front-2024", "./datasets/_real/digiface_118k_64",
                          32, "./configs/digiface_identities_seta.txt"),
    "real_seta_d3.yaml": (20260522, "mobile-front-2024", "./datasets/_real/digiface_118k_64",
                          256, "./configs/digiface_identities_seta.txt"),
    "real_setb_d1.yaml": (20260523, "webcam-1080p", "./datasets/_real/digiface_118k_64",
                          4, "./configs/digiface_identities_setb.txt"),
    "real_setb_d2.yaml": (20260523, "webcam-1080p", "./datasets/_real/digiface_118k_64",
                          32, "./configs/digiface_identities_setb.txt"),
    "real_setb_d3.yaml": (20260523, "webcam-1080p", "./datasets/_real/digiface_118k_64",
                          256, "./configs/digiface_identities_setb.txt"),
}


def test_real_configs_present_and_well_formed():
    for fname, (seed, sensor, fixture, spb, ids_file) in EXPECTED.items():
        cfg = yaml.safe_load((CFG_DIR / fname).read_text())
        assert cfg["run"]["seed"] == seed, fname
        assert cfg["run"]["deterministic"] is True, fname
        assert cfg["run"]["output"] == f"./datasets/{Path(fname).stem}", fname
        assert cfg["modality"] == "face", fname
        assert cfg["sensor_preset"] == sensor, fname
        assert cfg["bonafide"]["root"] == fixture, fname
        assert cfg["bonafide"]["samples_per_bonafide"] == spb, fname
        assert cfg["bonafide"]["identities_file"] == ids_file, fname
        assert set(cfg["attacks"].keys()) == {"print", "replay"}, fname
        assert cfg["attacks"]["print"]["weight"] == 1.0, fname
        assert cfg["attacks"]["replay"]["weight"] == 1.0, fname
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_real_configs.py -q 2>&1 | tail -3`
Expected: 1 failed (file not found).

- [ ] **Step 3: Create all six configs**

`configs/runs/real_seta_d1.yaml`:
```yaml
run:
  name: real_seta_d1
  output: ./datasets/real_seta_d1
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/digiface_118k_64
  samples_per_bonafide: 6
  identities_file: ./configs/digiface_identities_seta.txt
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

`configs/runs/real_seta_d2.yaml`: identical to `real_seta_d1.yaml` except `name: real_seta_d2`, `output: ./datasets/real_seta_d2`, `samples_per_bonafide: 32`.

`configs/runs/real_seta_d3.yaml`: identical to `real_seta_d1.yaml` except `name: real_seta_d3`, `output: ./datasets/real_seta_d3`, `samples_per_bonafide: 256`.

`configs/runs/real_setb_d1.yaml`:
```yaml
run:
  name: real_setb_d1
  output: ./datasets/real_setb_d1
  seed: 20260523
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/digiface_118k_64
  samples_per_bonafide: 4
  identities_file: ./configs/digiface_identities_setb.txt
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

`configs/runs/real_setb_d2.yaml`: identical to `real_setb_d1.yaml` except `name: real_setb_d2`, `output: ./datasets/real_setb_d2`, `samples_per_bonafide: 32`.

`configs/runs/real_setb_d3.yaml`: identical to `real_setb_d1.yaml` except `name: real_setb_d3`, `output: ./datasets/real_setb_d3`, `samples_per_bonafide: 256`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_real_configs.py -q 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 171 passed, 1 skipped (170 prior + 1 new).

- [ ] **Step 6: Commit**

```bash
git add configs/runs/real_seta_d1.yaml configs/runs/real_seta_d2.yaml configs/runs/real_seta_d3.yaml configs/runs/real_setb_d1.yaml configs/runs/real_setb_d2.yaml configs/runs/real_setb_d3.yaml pad-synth-face/tests/test_real_configs.py
git commit -m "feat(pad-spark): six real-bonafide measurement configs (DigiFace-1M, 8/16 IDs)"
```

---

## Task 7: Generate the six real-bonafide datasets locally

**Files:** none (output to gitignored `datasets/`).

- [ ] **Step 1: Generate all six**

```bash
cd /Users/stuartwells/test
for f in real_seta_d1 real_seta_d2 real_seta_d3 real_setb_d1 real_setb_d2 real_setb_d3; do
  echo "=== generating $f ==="
  .venv/bin/python -m pad_synth_face.cli generate --config configs/runs/${f}.yaml | tail -3
done
```

Expected: each prints a JSON summary with `"failed": 0`. Wall-time ~10–15 minutes (real-face image loading is slower than the procedural fixture's in-memory generation).

- [ ] **Step 2: Verify counts**

```bash
for d in datasets/real_seta_d{1,2,3} datasets/real_setb_d{1,2,3}; do
  n=$(wc -l < "$d/manifest.jsonl")
  bona=$(grep -c '"label":"bonafide"' "$d/manifest.jsonl")
  attack=$(grep -c '"label":"attack"' "$d/manifest.jsonl")
  printf "%-22s total=%5d  bonafide=%5d  attack=%5d\n" "$d" "$n" "$bona" "$attack"
done
```

Expected (exact):
```
datasets/real_seta_d1   total=   96  bonafide=   48  attack=   48
datasets/real_seta_d2   total=  512  bonafide=  256  attack=  256
datasets/real_seta_d3   total= 4096  bonafide= 2048  attack= 2048
datasets/real_setb_d1   total=  128  bonafide=   64  attack=   64
datasets/real_setb_d2   total= 1024  bonafide=  512  attack=  512
datasets/real_setb_d3   total= 8192  bonafide= 4096  attack= 4096
```

If counts diverge: STOP and report BLOCKED — most likely the identity-restrict logic or pipeline wiring has a bug.

- [ ] **Step 3: Verify the manifest stamps the correct ontology_version**

```bash
.venv/bin/python -c "
import json
with open('datasets/real_seta_d1/manifest.jsonl') as fh:
    rec = json.loads(fh.readline())
print('ontology_version:', rec.get('ontology_version'))
print('bonafide_source:', rec.get('bonafide_source'))
"
```
Expected: `ontology_version: 2026-05-23` (the current v2.1 print ontology version, NOT the old hardcoded `2026-05-11`); `bonafide_source` shows the actual DigiFace identity ID used.

- [ ] **Step 4: No commit** (datasets gitignored; the configs are the regenerable spec).

---

## Task 8: rsync + run 27-cell v2.1-on-real-bonafide sweep on the Spark

**Files:** none (remote operations).

- [ ] **Step 1: Sync the latest code to the Spark**

```bash
rsync -a --delete \
  --exclude='.venv' --exclude='__pycache__' --exclude='datasets' \
  --exclude='.superpowers' --exclude='.git/objects/pack' \
  /Users/stuartwells/test/ \
  swells@spark-50d2.local:~/ml/projects/pad-spark/
```

- [ ] **Step 2: rsync the six real-bonafide datasets**

```bash
for d in real_seta_d1 real_seta_d2 real_seta_d3 real_setb_d1 real_setb_d2 real_setb_d3; do
  rsync -a --partial \
    "/Users/stuartwells/test/datasets/${d}/" \
    "swells@spark-50d2.local:~/ml/datasets/${d}/"
done
```

- [ ] **Step 3: Run the 27-cell v2.1-on-real-bonafide sweep**

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)_real
echo "$ts" > /tmp/padspark_real_ts
ssh swells@spark-50d2.local "cd ~/ml/projects/pad-spark && .venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 ~/ml/datasets/real_seta_d1 --set-b-d1 ~/ml/datasets/real_setb_d1 \
  --set-a-d2 ~/ml/datasets/real_seta_d2 --set-b-d2 ~/ml/datasets/real_setb_d2 \
  --set-a-d3 ~/ml/datasets/real_seta_d3 --set-b-d3 ~/ml/datasets/real_setb_d3 \
  --set-a-d4 ~/ml/datasets/spark_seta_d4 --set-b-d4 ~/ml/datasets/spark_setb_d4 \
  --output-dir ~/ml/logs/pad-spark/${ts} \
  --device cuda --epochs 10 --batch-size 32 \
  --cells L1:D1:0,L1:D1:1,L1:D1:2,L1:D2:0,L1:D2:1,L1:D2:2,L1:D3:0,L1:D3:1,L1:D3:2,L2:D1:0,L2:D1:1,L2:D1:2,L2:D2:0,L2:D2:1,L2:D2:2,L2:D3:0,L2:D3:1,L2:D3:2,L3:D1:0,L3:D1:1,L3:D1:2,L3:D2:0,L3:D2:1,L3:D2:2,L3:D3:0,L3:D3:1,L3:D3:2" 2>&1 | tail -32
```

Expected: 27 lines `L? D? seed=?  eer_in=0.??  eer_cross=0.??  ??.?s`. Wall-time ~5–7 minutes on the GB10.

**Watch for**: any cell printing `eer_cross=0.000` is the artifact-survival signal. The whole experiment hinges on this.

- [ ] **Step 4: Confirm 27 JSONs**

```bash
ssh swells@spark-50d2.local "ls ~/ml/logs/pad-spark/$(cat /tmp/padspark_real_ts)/runs/ | wc -l"
```
Expected: `27`.

- [ ] **Step 5: No commit** (results land in the report dir in T9).

---

## Task 9: rsync results back, write three-way comparison report, commit

**Files:**
- Add (rsync'd): 27 JSONs at `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_real/L{1,2,3}_D{1,2,3}_{0,1,2}.json`
- Add: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_real.csv`
- Modify: `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` (append v2.1-on-real section)
- Modify: `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md` (one-line append)

- [ ] **Step 1: rsync the 27 real-bonafide JSONs into a new subdir**

```bash
ts=$(cat /tmp/padspark_real_ts)
mkdir -p docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_real
rsync -av "swells@spark-50d2.local:~/ml/logs/pad-spark/${ts}/runs/" \
  docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_real/
ls docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_real/ | wc -l
```
Expected: 27.

- [ ] **Step 2: Build `summary_real.csv`**

```bash
.venv/bin/python - <<'PY'
import csv, json
from pathlib import Path
runs_dir = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_real")
out = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_real.csv")
rows = []
for p in sorted(runs_dir.glob("*.json")):
    r = json.loads(p.read_text())
    rows.append([r["capacity"], r["data_level"], r["seed"],
                 r["eer_in_domain"], r["eer_cross_domain"],
                 f"{r['train_seconds']:.2f}"])
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["capacity","data_level","seed","eer_in_domain","eer_cross_domain","train_seconds"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
PY
```
Expected: `wrote 27 rows`.

- [ ] **Step 3: Compute the synthetic-v2.1 vs real-v2.1 comparison**

```bash
.venv/bin/python - <<'PY'
import json, statistics as st
from pathlib import Path
rd = Path("docs/superpowers/reports/2026-05-22-pad-spark-sweep-results")

def load(sub, lf):
    cells = {}
    for p in sorted((rd / sub).glob("*.json")):
        r = json.loads(p.read_text())
        if r["data_level"] not in lf: continue
        cells.setdefault((r["capacity"], r["data_level"]), []).append(r)
    return cells

synth = load("runs_v21", {"D1","D2","D3"})
real  = load("runs_real", {"D1","D2","D3"})

print(f"{'cell':<8} {'synth v2.1':<22}  {'real v2.1':<22}  {'real in-dom':<22}  {'real ≤ 0.001?':>13}")
broken = True
for L in ("L1","L2","L3"):
    for D in ("D1","D2","D3"):
        sg=[r["eer_cross_domain"] for r in synth[(L,D)]]
        rg=[r["eer_cross_domain"] for r in real[(L,D)]]
        ri=[r["eer_in_domain"] for r in real[(L,D)]]
        ms,ss=st.mean(sg),st.stdev(sg)
        mr,sr=st.mean(rg),st.stdev(rg)
        mi,si=st.mean(ri),st.stdev(ri)
        if mr<=0.001: broken=False
        print(f"{L} {D}    {ms:.3f}+-{ss:.3f}    {mr:.3f}+-{sr:.3f}    {mi:.3f}+-{si:.3f}    {'no!' if mr<=0.001 else 'yes':>10}")
print()
print(f"Artifact broken by real bonafide (all real-v2.1 cross-domain > 0.001): {broken}")
PY
```

Record the table and the verdict.

- [ ] **Step 4: Append the real-bonafide section to the report**

Open `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md` and append at the end:

````markdown

---

## 2026-05-22 update — real-bonafide v2.1 result (DigiFace-1M)

The v2.1 "two-strikes" finding promoted real-bonafide integration to top priority. DigiFace-1M's 118k aligned subset (Microsoft Research synthetic-but-photographic-realism faces, MIT-licensed) was ingested as a bonafide source, replacing the procedural skin-tone blob fixture. 8/16 identity-disjoint identities deterministically selected (pinned in `configs/digiface_identities_set{a,b}.txt`); all 118k images resized to 64×64 for direct apples-to-apples with synthetic-bonafide v2.1. Same v2.1 print physics, same 27-cell sweep on the same GB10. Code SHA at sweep time: <fill_real_sha>. Torch: `2.12.0.dev20260407+cu128`.

**Synthetic-v2.1 → real-v2.1 cross-domain EER comparison (mean ± std):**

| Cell | synth v2.1 | real v2.1 (DigiFace) |
|---|---|---|
| L1·D1 | 0.245 ± 0.080 | <fill> |
| L1·D2 | 0.230 ± 0.057 | <fill> |
| L1·D3 | 0.000 ± 0.000 | <fill> |
| L2·D1 | 0.130 ± 0.063 | <fill> |
| L2·D2 | 0.000 ± 0.000 | <fill> |
| L2·D3 | 0.000 ± 0.000 | <fill> |
| L3·D1 | 0.109 ± 0.109 | <fill> |
| L3·D2 | 0.000 ± 0.000 | <fill> |
| L3·D3 | 0.000 ± 0.000 | <fill> |

**Real-v2.1 in-domain EER (mean ± std):**

| | D1 | D2 | D3 |
|---|---|---|---|
| L1 | <fill> | <fill> | <fill> |
| L2 | <fill> | <fill> | <fill> |
| L3 | <fill> | <fill> | <fill> |

**Artifact verdict:** all real-v2.1 cross-domain cell means must be > 0.001 EER. Verdict: <fill: BROKEN / SURVIVED — list any cells with mean ≤ 0.001>.

**Diagnosis:** <one paragraph. If artifact broken: "shifting bonafide to real-face textures defeats the binary-palette shortcut; physics-axis improvements ARE the lever once the synthetic-bonafide confound is removed." If artifact survives: "the artifact is deeper than the bonafide-distribution component — the binary-threshold halftone output itself produces a learnable signature regardless of underlying face distribution. Synthetic halftoning of any form is unsuitable as production training data; pivot to real attack capture or move further from halftone-based attacks.">

**Phase 2 recommendation update:** <one paragraph. If broken: "ship v2.1 physics + DigiFace bonafide as the next production baseline; mask-attack sub-project proceeds on this combined base. The 'two-strikes' artifact concern is resolved." If survives: "pure-synthetic print attacks have a hard ceiling regardless of bonafide. Promote real attack capture to top Phase 2.5 priority; v2.2 (gray-level halftoning) becomes a less attractive option since the artifact appears not to be palette-driven.">

### Raw results

- Real-v2.1 per-cell JSON: [`./2026-05-22-pad-spark-sweep-results/runs_real/`](./2026-05-22-pad-spark-sweep-results/runs_real/) (27 files)
- Real-v2.1 summary CSV: [`./2026-05-22-pad-spark-sweep-results/summary_real.csv`](./2026-05-22-pad-spark-sweep-results/summary_real.csv)
````

Fill every `<fill>` from the Step 3 output. The git SHA is from any real-v2.1 JSON's `git_sha` field.

- [ ] **Step 5: Append the one-line roadmap update**

In `docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md`, append at the end:

```markdown

---

## 2026-05-22 update — real-bonafide (DigiFace-1M) v2.1 sweep

The originally-deferred Phase-1 real-data lever finally landed. DigiFace-1M 118k subset ingested (8 IDs for Set A, 16 for Set B, identity-disjoint pinned lists), v2.1 print physics applied, 27-cell sweep at D1–D3. **Artifact verdict: <broken/survived>.** See [`2026-05-22-pad-spark-sweep-results.md`](./2026-05-22-pad-spark-sweep-results.md) §"real-bonafide v2.1 result" for the synth-vs-real comparison and the updated Phase 2 prioritization.
```

Replace `<broken/survived>` with the actual verdict.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_real/ \
        docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/summary_real.csv \
        docs/superpowers/reports/2026-05-11-pad-synth-decisions-and-roadmap.md
git commit -m "report(pad-real-bonafide): v2.1-on-DigiFace sweep — <verdict>"
```

Replace `<verdict>` with `artifact broken` or `artifact survived`.

- [ ] **Step 7: Full suite green (final)**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 171 passed, 1 skipped, 4 warnings.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose — break artifact via real-face textures | Tasks 3–8 (data) + T9 (measurement) |
| §2 question + 0.001 threshold | Task 9 Step 3 (verdict logic) |
| §3 DigiFace-1M 118k subset, MIT-licensed | Task 3 |
| §4 resize to 64×64 | Task 4 |
| §5 8/16 identity-disjoint selection | Task 5 |
| §6 six new configs with identities_file field | Task 6 |
| §7 loader restrict_to + jpg/png glob + pipeline identities_file | Tasks 1 + 2 |
| §8 measurement plan (rsync, run, report) | Tasks 7, 8, 9 |
| §9 architecture boundaries; in-scope pipeline.py modifications including hardcode fix | Tasks 1, 2 |
| §10 non-goals | None violated: no real attacks, no v1/v2 reruns, no full 6GB DL, no ontology bump, no new sensor presets, no model changes, no FFHQ |
| §11 success criteria | Task 9 Step 7 (suite green) + Task 9 Step 6 (commit) |
| §12 operational download risk | Task 3 (download script + BLOCKED protocol) |

**Placeholder scan:** Every `<fill>` and `<verdict>` is in the report template that T9 Step 3 populates from real run data. No "TBD/TODO/implement later" elsewhere. All code blocks complete.

**Type consistency:** `DigiFaceLoader(root, restrict_to=None)` signature consistent across T1 (definition), T2 (pipeline use), T6 (config field shape that feeds it). The `_ontology_version` local in `pipeline.py` is read from `attack_modules["print"].ontology.version` (assumes "print" is always present — true in all six new configs and all existing configs). Manifest schema (`ontology_version: str`) unchanged. JSON keys in real-v2.1 sweep results unchanged from parent project's schema.
