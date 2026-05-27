# PAD Real-Attack Capture Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dataset-agnostic real-attack ingester + synth→real eval wiring so a licensed real-attack PAD dataset can plug into the v2.1+DigiFace+mask base — harness now, dataset later.

**Architecture:** A package function `ingest_real_attack()` reads the folder-convention source (`<src>/bonafide/**`, `<src>/attack/<type>/**`), resizes to 64×64, and writes the canonical `face/{bonafide,<type>}/*.jpg` + `manifest.jsonl` + a `RealAttackIngested` provenance event — exactly the layout `TinyPADDataset` already consumes. A thin `scripts/prepare_real_attack.py` wraps it as a CLI. The synth→real sweep runs through the existing `spark_sweep.py` unchanged. A procedural `build_fixture_real_attack` fixture makes the whole path testable with no real data.

**Tech Stack:** Python 3.12+, NumPy, Pillow (`PIL.Image`), Pydantic, pytest. Reuses `ManifestWriter`, `SampleRecord`, `BonafideSource`, `ProvenanceLedger`, `check_image_basic`.

**Spec:** [`../specs/2026-05-27-pad-real-attack-capture-design.md`](../specs/2026-05-27-pad-real-attack-capture-design.md)

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `pad-synth-core/src/pad_synth_core/provenance.py` | add `RealAttackIngested` event to the union | Modify |
| `pad-synth-core/tests/test_provenance_real_attack.py` | event serialises with required fields | Create |
| `pad-synth-face/src/pad_synth_face/_fixtures.py` | `build_fixture_real_attack(root)` procedural source | Modify |
| `pad-synth-face/src/pad_synth_face/real_attack.py` | `ingest_real_attack()` — core ingestion logic (in the package, so it's unit-testable) | Create |
| `scripts/prepare_real_attack.py` | thin CLI wrapper over `ingest_real_attack` | Create |
| `pad-synth-face/tests/test_real_attack_ingest.py` | ingestion correctness, idempotency, determinism | Create |
| `pad-synth-face/tests/test_real_attack_wiring.py` | synth→real `train_and_cross_domain_eval` runs end-to-end | Create |
| `docs/real-attack-capture.md` | folder convention + prepare/sweep commands + no-commit policy | Create |

Design note: the spec lists `scripts/prepare_real_attack.py` as "the ingester". For testability the ingestion logic lives in the package (`pad_synth_face/real_attack.py`) and the script is a thin CLI — mirroring how `DigiFaceLoader` lives in the package while `prepare_digiface_64.py` is a script. Tests import the function directly.

---

## Task 1: `RealAttackIngested` provenance event

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/provenance.py`
- Create (test): `pad-synth-core/tests/test_provenance_real_attack.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-core/tests/test_provenance_real_attack.py`:

```python
import json

from pad_synth_core.provenance import ProvenanceLedger, RealAttackIngested


def test_real_attack_ingested_serialises(tmp_path):
    ev = RealAttackIngested(
        name="MSU-MFSD",
        license="MSU research EULA",
        source_url="https://example.org/msu-mfsd",
        sha256_of_index="abc123",
        attack_types=["print", "replay"],
    )
    assert ev.type == "real_attack_dataset_ingested"

    led_path = tmp_path / "provenance.jsonl"
    with ProvenanceLedger(led_path) as led:
        led.record(ev)
    rec = json.loads(led_path.read_text().splitlines()[0])
    assert rec["type"] == "real_attack_dataset_ingested"
    assert rec["name"] == "MSU-MFSD"
    assert rec["license"] == "MSU research EULA"
    assert rec["attack_types"] == ["print", "replay"]
    assert "ingested_at" in rec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_provenance_real_attack.py -v`
Expected: FAIL — `ImportError: cannot import name 'RealAttackIngested'`.

- [ ] **Step 3: Add the event**

In `pad-synth-core/src/pad_synth_core/provenance.py`, add this class after `BonafideIngested` (it reuses the module-level `_now`):

```python
class RealAttackIngested(BaseModel):
    type: Literal["real_attack_dataset_ingested"] = "real_attack_dataset_ingested"
    name: str
    license: str
    source_url: str
    sha256_of_index: str
    attack_types: list[str]
    ingested_at: datetime = Field(default_factory=_now)
```

Then add it to the union:

```python
ProvenanceEvent = (
    BonafideIngested | GeneratorRegistered | OntologyCitation | RealAttackIngested
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_provenance_real_attack.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/provenance.py pad-synth-core/tests/test_provenance_real_attack.py
git commit -m "feat(pad-core): RealAttackIngested provenance event"
```

---

## Task 2: `build_fixture_real_attack` fixture

A procedural folder-convention source so ingestion is testable with no real data. Images are 96×96 (so the 64×64 resize is exercised) and noisy (so `check_image_basic`'s `std >= 1.0` passes).

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/_fixtures.py`
- Create (test): `pad-synth-face/tests/test_fixture_real_attack.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-face/tests/test_fixture_real_attack.py`:

```python
import numpy as np
from PIL import Image

from pad_synth_face._fixtures import build_fixture_real_attack


def test_fixture_real_attack_layout(tmp_path):
    root = build_fixture_real_attack(tmp_path / "src")

    bonafide = sorted((root / "bonafide").rglob("*.png"))
    print_a = sorted((root / "attack" / "print").rglob("*.png"))
    replay_a = sorted((root / "attack" / "replay").rglob("*.png"))

    assert len(bonafide) >= 4
    assert len(print_a) >= 4
    assert len(replay_a) >= 4

    # Images are larger than 64 (resize must do work) and non-degenerate.
    arr = np.array(Image.open(bonafide[0]).convert("RGB"))
    assert arr.shape[0] > 64 and arr.shape[1] > 64
    assert float(arr.std()) >= 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_fixture_real_attack.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_fixture_real_attack'`.

- [ ] **Step 3: Implement the fixture**

Append to `pad-synth-face/src/pad_synth_face/_fixtures.py`:

```python
def build_fixture_real_attack(root: Path) -> Path:
    """Procedural folder-convention real-attack source for tests.

    Layout: <root>/bonafide/subjectNN/*.png and
    <root>/attack/<type>/subjectNN/*.png. Images are 96x96 RGB with
    structured noise (std well above the QC floor) so the ingester's
    resize and check_image_basic both exercise real work. No real data,
    no PII -- purely synthetic stand-ins for the capture pipeline.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260527)
    size = 96

    def _emit(class_dir: Path, n_subjects: int, base_shift: int) -> None:
        for s in range(n_subjects):
            subj = class_dir / f"subject{s:02d}"
            subj.mkdir(parents=True, exist_ok=True)
            for k in range(2):
                base = rng.integers(40, 200, size=3)
                arr = np.tile(base, (size, size, 1)).astype(np.int16)
                arr += rng.integers(-30, 30, size=(size, size, 3), dtype=np.int16)
                arr += base_shift  # per-class tint so classes are separable
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(subj / f"{k}.png")

    _emit(root / "bonafide", n_subjects=3, base_shift=0)
    _emit(root / "attack" / "print", n_subjects=3, base_shift=-25)
    _emit(root / "attack" / "replay", n_subjects=3, base_shift=25)
    return root
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_fixture_real_attack.py -v`
Expected: PASS (3 subjects × 2 images = 6 per class, ≥ 4).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/_fixtures.py pad-synth-face/tests/test_fixture_real_attack.py
git commit -m "feat(pad-real-attack): procedural folder-convention test fixture"
```

---

## Task 3: `ingest_real_attack` core + tests

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/real_attack.py`
- Create (test): `pad-synth-face/tests/test_real_attack_ingest.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_real_attack_ingest.py`:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_face._fixtures import build_fixture_real_attack
from pad_synth_face.real_attack import ingest_real_attack


def _ingest(tmp_path: Path):
    src = build_fixture_real_attack(tmp_path / "src")
    out = tmp_path / "out"
    summary = ingest_real_attack(
        src=src, out=out,
        dataset_name="FIXTURE-RA", license="test-only",
        source_url="https://example.org/fixture",
    )
    return out, summary


def test_canonical_layout_and_counts(tmp_path):
    out, summary = _ingest(tmp_path)
    bona = sorted((out / "face" / "bonafide").glob("*.jpg"))
    pr = sorted((out / "face" / "print").glob("*.jpg"))
    rp = sorted((out / "face" / "replay").glob("*.jpg"))
    assert len(bona) == 6 and len(pr) == 6 and len(rp) == 6
    assert summary["counts"] == {"bonafide": 6, "print": 6, "replay": 6}
    assert sorted(summary["attack_types"]) == ["print", "replay"]
    # Images are 64x64 RGB.
    arr = np.array(Image.open(bona[0]).convert("RGB"))
    assert arr.shape == (64, 64, 3)


def test_manifest_labels_and_attack_type(tmp_path):
    out, _ = _ingest(tmp_path)
    recs = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    by_label = {}
    for r in recs:
        by_label.setdefault(r["label"], []).append(r)
    assert len(by_label["bonafide"]) == 6
    assert len(by_label["attack"]) == 12
    assert all(r["attack_type"] is None for r in by_label["bonafide"])
    assert {r["attack_type"] for r in by_label["attack"]} == {"print", "replay"}
    # Dataset attribution is recorded on every record.
    assert all(r["bonafide_source"]["dataset"] == "FIXTURE-RA" for r in recs)
    assert all(r["bonafide_source"]["license"] == "test-only" for r in recs)


def test_provenance_event_written(tmp_path):
    out, _ = _ingest(tmp_path)
    prov = [json.loads(l) for l in (out / "provenance.jsonl").read_text().splitlines()]
    ra = [e for e in prov if e["type"] == "real_attack_dataset_ingested"]
    assert len(ra) == 1
    assert ra[0]["name"] == "FIXTURE-RA"
    assert ra[0]["license"] == "test-only"
    assert sorted(ra[0]["attack_types"]) == ["print", "replay"]


def test_idempotent_and_deterministic(tmp_path):
    src = build_fixture_real_attack(tmp_path / "src")
    out = tmp_path / "out"
    common = dict(src=src, out=out, dataset_name="FIXTURE-RA",
                  license="test-only", source_url="https://example.org/fixture")
    s1 = ingest_real_attack(**common)
    # Hash the produced jpgs.
    import hashlib
    def digest():
        h = hashlib.sha256()
        for p in sorted((out / "face").rglob("*.jpg")):
            h.update(p.read_bytes())
        return h.hexdigest()
    d1 = digest()
    s2 = ingest_real_attack(**common)  # re-run
    assert s2["counts"] == {"bonafide": 0, "print": 0, "replay": 0}  # all skipped
    assert digest() == d1  # byte-identical, nothing rewritten
    assert s1["counts"] == {"bonafide": 6, "print": 6, "replay": 6}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_real_attack_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: pad_synth_face.real_attack`.

- [ ] **Step 3: Implement the ingester**

Create `pad-synth-face/src/pad_synth_face/real_attack.py`:

```python
"""Ingest a real-attack PAD dataset into the canonical eval layout.

Reads the folder convention
    <src>/bonafide/**/*.{jpg,jpeg,png}
    <src>/attack/<attack_type>/**/*.{jpg,jpeg,png}
and writes the canonical 64x64 dataset that pad_synth_core.eval consumes:
    <out>/face/bonafide/real-bonafide-NNNNNNNN.jpg
    <out>/face/<attack_type>/real-<attack_type>-NNNNNNNN.jpg
    <out>/manifest.jsonl
    <out>/provenance.jsonl   (RealAttackIngested -- records dataset + licence)

Input is extracted image frames (video decoding is the caller's pre-step).
Deterministic (sorted source order) and idempotent (existing sample IDs are
skipped). Real images are never committed -- write under datasets/_real_attack/
which the gitignored datasets/ covers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import pad_synth_core
import pad_synth_face
from pad_synth_core.manifest import BonafideSource, ManifestWriter, SampleRecord
from pad_synth_core.provenance import ProvenanceLedger, RealAttackIngested
from pad_synth_core.qc.per_sample import check_image_basic

_EXTS = {".jpg", ".jpeg", ".png"}
_TARGET = (64, 64, 3)


def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in _EXTS)


def _load_64(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((64, 64), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def ingest_real_attack(
    src: Path,
    out: Path,
    dataset_name: str,
    license: str,
    source_url: str,
    max_per_class: int | None = None,
) -> dict[str, Any]:
    src, out = Path(src), Path(out)
    face_root = out / "face"
    counts: dict[str, int] = {}
    index_paths: list[str] = []

    manifest = ManifestWriter(out / "manifest.jsonl")
    existing = manifest.existing_sample_ids()

    def _process(subdir: str, attack_type: str | None, src_dir: Path, prefix: str) -> None:
        out_dir = face_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        files = _list_images(src_dir)
        if max_per_class is not None:
            files = files[:max_per_class]
        written = 0
        for i, fp in enumerate(files):
            index_paths.append(str(fp.relative_to(src)))
            sid = f"{prefix}-{i:08d}"
            if sid in existing:
                continue
            arr = _load_64(fp)
            if not check_image_basic(arr, _TARGET).ok:
                continue
            out_rel = f"face/{subdir}/{sid}.jpg"
            Image.fromarray(arr).save(out / out_rel, format="JPEG", quality=92)
            sha = hashlib.sha256((out / out_rel).read_bytes()).hexdigest()
            manifest.append(SampleRecord(
                sample_id=sid,
                modality="face",
                label="bonafide" if attack_type is None else "attack",
                attack_type=attack_type,
                bonafide_source=BonafideSource(
                    dataset=dataset_name, id=str(fp.relative_to(src)), license=license
                ),
                pipeline_version=f"pad-synth-face@{pad_synth_face.__version__}",
                core_version=f"pad-synth-core@{pad_synth_core.__version__}",
                ontology_version="real-attack-capture",
                seed=0,
                output_path=out_rel,
                output_sha256=sha,
            ))
            written += 1
        counts[subdir] = written

    bonafide_dir = src / "bonafide"
    if bonafide_dir.is_dir():
        _process("bonafide", None, bonafide_dir, "real-bonafide")

    attack_types: list[str] = []
    attack_root = src / "attack"
    if attack_root.is_dir():
        for tdir in sorted(p for p in attack_root.iterdir() if p.is_dir()):
            attack_types.append(tdir.name)
            _process(tdir.name, tdir.name, tdir, f"real-{tdir.name}")

    manifest.close()

    with ProvenanceLedger(out / "provenance.jsonl") as led:
        led.record(RealAttackIngested(
            name=dataset_name,
            license=license,
            source_url=source_url,
            sha256_of_index=hashlib.sha256(
                "|".join(sorted(index_paths)).encode()
            ).hexdigest(),
            attack_types=attack_types,
        ))

    return {"out": str(out), "counts": counts, "attack_types": attack_types}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_real_attack_ingest.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/real_attack.py pad-synth-face/tests/test_real_attack_ingest.py
git commit -m "feat(pad-real-attack): folder-convention ingester -> canonical eval layout"
```

---

## Task 4: CLI wrapper script

**Files:**
- Create: `scripts/prepare_real_attack.py`

- [ ] **Step 1: Write the script**

Create `scripts/prepare_real_attack.py`:

```python
#!/usr/bin/env python3
"""CLI wrapper: ingest a real-attack PAD dataset into the canonical eval layout.

Thin shim over pad_synth_face.real_attack.ingest_real_attack. See
docs/real-attack-capture.md for the folder convention and the synth->real
sweep command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.real_attack import ingest_real_attack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source: <src>/bonafide/** and <src>/attack/<type>/**")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination canonical dataset dir (under datasets/_real_attack/)")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--license", required=True, help="Dataset licence / EULA string")
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--max-per-class", type=int, default=None)
    args = ap.parse_args()

    summary = ingest_real_attack(
        src=args.src, out=args.out,
        dataset_name=args.dataset_name, license=args.license,
        source_url=args.source_url, max_per_class=args.max_per_class,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI against the fixture**

Run:
```bash
.venv/bin/python - <<'PY'
import tempfile, subprocess, sys, pathlib
sys.path.insert(0, "pad-synth-face/src")
from pad_synth_face._fixtures import build_fixture_real_attack
d = pathlib.Path(tempfile.mkdtemp())
build_fixture_real_attack(d / "src")
r = subprocess.run([".venv/bin/python", "scripts/prepare_real_attack.py",
    "--src", str(d/"src"), "--out", str(d/"out"),
    "--dataset-name", "SMOKE", "--license", "x", "--source-url", "y"],
    capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, r.stderr
assert (d/"out"/"face"/"bonafide").is_dir()
print("CLI OK")
PY
```
Expected: prints a summary JSON and `CLI OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/prepare_real_attack.py
git commit -m "feat(pad-real-attack): prepare_real_attack.py CLI wrapper"
```

---

## Task 5: Synth→real eval wiring test

Proves a synthetic train set + an ingested real eval set run end-to-end through `train_and_cross_domain_eval` and that both real classes are read.

**Files:**
- Create (test): `pad-synth-face/tests/test_real_attack_wiring.py`

- [ ] **Step 1: Write the test**

Create `pad-synth-face/tests/test_real_attack_wiring.py`:

```python
from pathlib import Path

import yaml

from pad_synth_core.eval.baseline import train_and_cross_domain_eval
from pad_synth_face._fixtures import build_fixture_real_attack
from pad_synth_face.pipeline import run_pipeline
from pad_synth_face.real_attack import ingest_real_attack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synth_train(fixture_bonafide_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "synth"
    cfg = {
        "run": {"name": "synth", "output": str(out), "seed": 1, "deterministic": True},
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 2},
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "synth.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_pipeline(cfg_path)
    return out


def test_synth_to_real_eval_runs(fixture_bonafide_dir: Path, tmp_path: Path):
    train_root = _make_synth_train(fixture_bonafide_dir, tmp_path)

    real_src = build_fixture_real_attack(tmp_path / "real_src")
    real_out = tmp_path / "real"
    ingest_real_attack(
        src=real_src, out=real_out,
        dataset_name="FIXTURE-RA", license="test-only",
        source_url="https://example.org/fixture",
    )

    result = train_and_cross_domain_eval(
        train_root=train_root,
        eval_root=real_out,
        epochs=1,
        batch_size=8,
        seed=0,
        device="cpu",
    )
    # Cross-domain EER is finite and the real eval set was actually read
    # (6 bonafide + 12 attack = 18 real samples).
    assert 0.0 <= float(result["eer_cross_domain"]) <= 1.0
    assert result["n_val_cross_domain"] == 18
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_real_attack_wiring.py -v`
Expected: PASS. (If `train_and_cross_domain_eval`'s parameter names differ, open `pad-synth-core/src/pad_synth_core/eval/baseline.py` and match the signature exactly — do not change the eval code.)

- [ ] **Step 3: Commit**

```bash
git add pad-synth-face/tests/test_real_attack_wiring.py
git commit -m "test(pad-real-attack): synth->real eval wiring end-to-end on fixtures"
```

---

## Task 6: Doc + full-suite / lint checkpoint

**Files:**
- Create: `docs/real-attack-capture.md`

- [ ] **Step 1: Write the doc**

Create `docs/real-attack-capture.md`:

````markdown
# Real-attack capture: ingesting a real-attack PAD dataset

Harness for the synth→real generalisation test. Train on the synthetic
production base (v2.1 print + DigiFace bonafide + replay + mask), evaluate on
a real-attack dataset.

## 1. Arrange the source (folder convention)

Extract frames from the dataset (video decoding is your pre-step) and lay them
out as:

```
<src>/
  bonafide/**/*.{jpg,jpeg,png}
  attack/<attack_type>/**/*.{jpg,jpeg,png}   # e.g. attack/print, attack/replay
```

`<attack_type>` is any string; it becomes the attack-class subdir in the output.

## 2. Ingest → canonical 64×64 eval dataset

```bash
.venv/bin/python scripts/prepare_real_attack.py \
  --src /path/to/<src> \
  --out datasets/_real_attack/<dataset> \
  --dataset-name "MSU-MFSD" \
  --license "MSU research EULA" \
  --source-url "https://.../msu-mfsd"
```

Writes `datasets/_real_attack/<dataset>/face/{bonafide,<type>}/*.jpg`, a
`manifest.jsonl`, and a `provenance.jsonl` recording the dataset name + licence.

**Real data is never committed.** `datasets/` is gitignored; keep ingested real
datasets under `datasets/_real_attack/`. Only the script, fixture, tests, and
this doc are committed.

## 3. Run the synth→real sweep

Generate the synthetic production base first (Set A), then point the sweep's
eval side at the ingested real dir for every data level:

```bash
.venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mix_seta_d1 --set-b-d1 datasets/_real_attack/<dataset> \
  --set-a-d2 datasets/mix_seta_d2 --set-b-d2 datasets/_real_attack/<dataset> \
  --set-a-d3 datasets/mix_seta_d3 --set-b-d3 datasets/_real_attack/<dataset> \
  --set-a-d4 datasets/mix_seta_d3 --set-b-d4 datasets/_real_attack/<dataset> \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_synth2real \
  --cells "$(python3 -c "print(','.join(f'{L}:{D}:{s}' for L in ('L1','L2','L3') for D in ('D1','D2','D3') for s in (0,1,2)))")" \
  --device cuda
```

This trains on synthetic at increasing data scale and evaluates on the fixed
real set — the headline synth→real EER curve. Append the result table to the
sweep-results report.
````

- [ ] **Step 2: Full suite + lint checkpoint**

Run:
```bash
.venv/bin/python -m pytest -q
uvx ruff check --select E,F,B,UP --line-length 100 --ignore E501 \
  pad-synth-face/src/pad_synth_face/real_attack.py \
  pad-synth-core/src/pad_synth_core/provenance.py \
  scripts/prepare_real_attack.py \
  pad-synth-face/tests/test_real_attack_ingest.py \
  pad-synth-face/tests/test_real_attack_wiring.py \
  pad-synth-face/tests/test_fixture_real_attack.py \
  pad-synth-core/tests/test_provenance_real_attack.py
```
Expected: suite green (prior baseline 185 passed, 1 skipped, plus the new tests). Ruff: `All checks passed!`.

Note on ruff: run it via `uvx ruff` with `--select E,F,B,UP` (correctness lints). Do NOT run the `I`/isort rule from the repo root — `uvx ruff` misclassifies the `src`-layout packages (`pad_synth_core`/`pad_synth_face`) as third-party and would spuriously rewrite import blocks across the whole codebase. Match the existing files' import style (blank line before first-party imports) by hand.

- [ ] **Step 3: Commit**

```bash
git add docs/real-attack-capture.md
git commit -m "docs(pad-real-attack): folder convention + synth->real sweep guide"
```

---

## Self-review notes

- **Spec coverage:** ingester + canonical output (§3/§4) → Task 3; folder convention (§3) → Tasks 2,3; `RealAttackIngested` provenance + licence capture (§5/§7) → Task 1, used in Task 3; fixture (§5) → Task 2; CLI (§5) → Task 4; synth→real wiring through unchanged `spark_sweep` (§6) → Task 5 (eval call) + Task 6 (doc command); testing (§8) → Tasks 3,5; data no-commit policy (§7) → Task 6 doc + relies on existing gitignored `datasets/`.
- **Scope boundary honoured:** no real dataset is downloaded and no real EER is produced — the harness + fixture + doc are the deliverable (spec §2).
- **No eval/sweep code changes** — `train_and_cross_domain_eval` and `spark_sweep.py` are used as-is.
- **`SampleRecord` required fields:** `bonafide_source` is populated as dataset attribution for every record (the dataset is the source of both real bonafide and real attack frames); `ontology_version="real-attack-capture"` sentinel, `seed=0` (deterministic, no rng). Verified against the schema in `pad-synth-core/src/pad_synth_core/manifest.py`.
