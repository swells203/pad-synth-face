# Commercial-Bonafide Retrain Validation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scaffolding that takes a commercially-licensed bonafide face set, runs it through the existing synthetic-attack pipeline, and produces a PASS/FAIL verdict on whether cross-domain EER holds vs the DigiFace baseline — so a shippable model can be de-risked before any data purchase.

**Architecture:** Five units mirroring the DFDC-prep pattern. Ingest *logic* lives in a new `pad_synth_face.commercial_bonafide` module (testable) with a thin `scripts/prepare_commercial_bonafide.py` CLI shim — matching how `dfdc.py`/`real_attack.py` are structured. An identity-pinning script and six sweep configs clone the DFDC/real equivalents. The one genuinely new piece is `scripts/compare_bonafide_eer.py`, which reads two sweep output dirs and renders the matched-scale A/B EER-delta verdict. No changes to attack physics, sensor, model factories, or `spark_sweep.py`.

**Tech Stack:** Python 3.11, NumPy, Pillow (LANCZOS resize), pydantic (existing `pad_synth_core.provenance`), PyYAML, pytest. Spark GB10 (CUDA) for the sweep itself (operational, not part of these code tasks).

**Spec:** `docs/superpowers/specs/2026-06-02-pad-commercial-bonafide-validation-design.md`

**Branch:** `feat/pad-commercial-bonafide-validation` (already created from main; spec committed as `b8e3c19`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `pad-synth-face/src/pad_synth_face/commercial_bonafide.py` | Ingest logic: canonical `<id>/<sample>` tree → 224 bonafide root + licence provenance | **Create** |
| `scripts/prepare_commercial_bonafide.py` | Thin CLI shim over the module | **Create** |
| `pad-synth-face/tests/test_commercial_bonafide_ingest.py` | Ingest unit tests (fixture, no real faces) | **Create** |
| `scripts/pin_commercial_identities.py` | Pin disjoint Set A/B identity lists from the ingested root | **Create** |
| `configs/runs/commercial_set{a,b}_d{1,2,3}.yaml` (×6) | Sweep configs (mirror `real_set*`, bonafide → commercial) | **Create** |
| `scripts/compare_bonafide_eer.py` | NEW verdict: matched-scale A/B cross-domain EER delta + PASS/FAIL | **Create** |
| `tests/test_compare_bonafide_eer.py` | Verdict unit tests (hand-authored mini sweep dirs) | **Create** |
| `docs/commercial-bonafide.md` | Operator runbook: obtain → ingest → pin → sweep → compare | **Create** |

No existing files modified. The resize loop is duplicated (~15 lines) from `scripts/prepare_digiface.py` rather than shared, to avoid coupling a script to a script (spec §6).

---

## Task 1: Commercial-bonafide ingest module + CLI shim

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/commercial_bonafide.py`
- Create: `scripts/prepare_commercial_bonafide.py`
- Test: `pad-synth-face/tests/test_commercial_bonafide_ingest.py`

**Context:** The ingest takes the canonical input contract `<src>/<identity>/<sample>.{png,jpg,jpeg}` (one dir per subject), resizes every image to `IMAGE_SIZE` (224) with PIL LANCZOS, writes `<out>/<identity>/NNN.png`, and records a `BonafideIngested` provenance event (name, licence, source_url, sha256_of_index) to `<out>/provenance.jsonl` plus a human-readable `<out>/_meta.json`. Logic lives in the module so it is unit-testable without a subprocess; the script is a thin shim (matching `dfdc.py` + `prepare_dfdc.py`). `BonafideIngested` already exists in `pad_synth_core.provenance` — no new event type.

- [ ] **Step 1: Write the failing ingest test**

Create `pad-synth-face/tests/test_commercial_bonafide_ingest.py`:

```python
"""Commercial-bonafide ingest: canonical contract -> 224 root + provenance.

Uses only generated images — no real or licensed faces ever touch the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_face.commercial_bonafide import ingest_commercial_bonafide


def _make_canonical_tree(root: Path, n_ids: int = 3, per_id: int = 2) -> None:
    rng = np.random.default_rng(0)
    for i in range(n_ids):
        d = root / f"subj_{i:03d}"
        d.mkdir(parents=True)
        for j in range(per_id):
            arr = rng.integers(0, 256, size=(96, 80, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{j}.jpg")


def test_ingest_produces_canonical_224_root(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src)
    summary = ingest_commercial_bonafide(
        src=src, out=out,
        license="Acme commercial face licence v1",
        source_url="https://vendor.example/sample",
        vendor="acme",
    )
    # 3 identities, 2 samples each, all resized to 224x224 PNG
    id_dirs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(id_dirs) == 3
    for d in id_dirs:
        pngs = sorted(d.glob("*.png"))
        assert len(pngs) == 2
        with Image.open(pngs[0]) as im:
            assert im.size == (224, 224)
    assert summary["identities"] == 3
    assert summary["samples_written"] == 6


def test_ingest_records_licence_provenance(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src)
    ingest_commercial_bonafide(
        src=src, out=out,
        license="Acme commercial face licence v1",
        source_url="https://vendor.example/sample",
        vendor="acme",
    )
    prov_lines = (out / "provenance.jsonl").read_text().strip().splitlines()
    assert len(prov_lines) == 1
    rec = json.loads(prov_lines[0])
    assert rec["type"] == "bonafide_dataset_ingested"
    assert rec["license"] == "Acme commercial face licence v1"
    assert rec["source_url"] == "https://vendor.example/sample"
    meta = json.loads((out / "_meta.json").read_text())
    assert meta["vendor"] == "acme"
    assert meta["target_size"] == 224
    assert meta["identities"] == 3


def test_ingest_is_idempotent(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src)
    ingest_commercial_bonafide(
        src=src, out=out, license="L", source_url="U", vendor="acme",
    )
    second = ingest_commercial_bonafide(
        src=src, out=out, license="L", source_url="U", vendor="acme",
    )
    # Re-run skips already-written files; no duplicate provenance event.
    assert second["samples_written"] == 0
    assert second["samples_skipped_existing"] == 6
    prov_lines = (out / "provenance.jsonl").read_text().strip().splitlines()
    assert len(prov_lines) == 1


def test_ingest_respects_max_per_identity(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src, n_ids=2, per_id=5)
    summary = ingest_commercial_bonafide(
        src=src, out=out, license="L", source_url="U",
        vendor="acme", max_per_identity=3,
    )
    assert summary["samples_written"] == 6  # 2 ids * 3 capped
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-face/tests/test_commercial_bonafide_ingest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pad_synth_face.commercial_bonafide'`.

- [ ] **Step 3: Implement the ingest module**

Create `pad-synth-face/src/pad_synth_face/commercial_bonafide.py`:

```python
"""Ingest a commercially-licensed bonafide face set into the canonical
224 bonafide root, recording licence provenance.

Input contract (canonical): <src>/<identity>/<sample>.{png,jpg,jpeg},
one directory per subject. Per-vendor layouts are reshaped into this
contract by a thin shim BEFORE calling this — see docs/commercial-bonafide.md.

Real images are never committed (datasets/ is gitignored). Only the licence
string + source URL travel with the data via provenance.jsonl.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from pad_synth_core import IMAGE_SIZE
from pad_synth_core.provenance import BonafideIngested, ProvenanceLedger

_IMG_EXT = {".png", ".jpg", ".jpeg"}


def ingest_commercial_bonafide(
    src: Path,
    out: Path,
    license: str,
    source_url: str,
    vendor: str = "unknown",
    size: int = IMAGE_SIZE,
    max_per_identity: int | None = None,
) -> dict[str, Any]:
    """Resize <src>/<id>/<sample> images to size x size PNGs under <out>/<id>/.

    Idempotent: skips destination files that already exist; records a single
    BonafideIngested provenance event only when at least one new file is
    written. Returns a summary dict.
    """
    src = Path(src)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    n_ids = 0
    n_written = 0
    n_skipped = 0
    for id_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        out_dir = out / id_dir.name
        out_dir.mkdir(exist_ok=True)
        n_ids += 1
        kept = 0
        for sample_path in sorted(id_dir.iterdir()):
            if sample_path.suffix.lower() not in _IMG_EXT:
                continue
            if max_per_identity is not None and kept >= max_per_identity:
                break
            kept += 1
            out_path = out_dir / f"{sample_path.stem}.png"
            if out_path.exists():
                n_skipped += 1
                continue
            with Image.open(sample_path) as im:
                im = im.convert("RGB").resize((size, size), Image.LANCZOS)
                im.save(out_path, format="PNG")
            n_written += 1

    sha_of_index = hashlib.sha256(
        "|".join(sorted(p.name for p in out.iterdir() if p.is_dir())).encode()
    ).hexdigest()

    if n_written > 0:
        with ProvenanceLedger(out / "provenance.jsonl") as led:
            led.record(BonafideIngested(
                name=f"commercial-bonafide:{vendor}",
                license=license,
                source_url=source_url,
                sha256_of_index=sha_of_index,
            ))

    meta = {
        "vendor": vendor,
        "target_size": size,
        "src": str(src),
        "identities": n_ids,
        "samples_written": n_written,
        "samples_skipped_existing": n_skipped,
        "license": license,
        "source_url": source_url,
    }
    (out / "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
.venv/bin/pytest pad-synth-face/tests/test_commercial_bonafide_ingest.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Create the CLI shim**

Create `scripts/prepare_commercial_bonafide.py`:

```python
#!/usr/bin/env python3
"""CLI wrapper: ingest a commercially-licensed bonafide set into the canonical
224 bonafide root. Thin shim over pad_synth_face.commercial_bonafide.

See docs/commercial-bonafide.md for the input contract and the validation flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.commercial_bonafide import ingest_commercial_bonafide  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path,
                    help="Canonical source root: <src>/<identity>/<sample>.{png,jpg,jpeg}")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination bonafide root (use datasets/_real/commercial_224)")
    ap.add_argument("--license", required=True, help="Commercial licence / EULA string")
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--vendor", default="unknown")
    ap.add_argument("--max-per-identity", type=int, default=None)
    args = ap.parse_args()

    summary = ingest_commercial_bonafide(
        src=args.src, out=args.out,
        license=args.license, source_url=args.source_url,
        vendor=args.vendor, max_per_identity=args.max_per_identity,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Lint the shim parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/prepare_commercial_bonafide.py').read()); print('shim parses OK')"
```

Expected: `shim parses OK`.

- [ ] **Step 7: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/commercial_bonafide.py \
        scripts/prepare_commercial_bonafide.py \
        pad-synth-face/tests/test_commercial_bonafide_ingest.py
git commit -m "feat(pad-commercial): bonafide ingest module + CLI shim

Canonical <id>/<sample> tree -> 224 PNG bonafide root, idempotent, records
a BonafideIngested provenance event with the commercial licence + source URL.
Logic in pad_synth_face.commercial_bonafide; scripts/ shim is thin."
```

---

## Task 2: Identity-pinning script

**Files:**
- Create: `scripts/pin_commercial_identities.py`

**Context:** Direct clone of `scripts/pin_dfdc_identities.py` (already in the repo), retargeted to the commercial root + output files. Deterministic seeded shuffle, disjoint 8-(Set A)/16-(Set B) split, errors if `<24` identities. No unit test — it is a near-exact clone of a script already exercised by hand; a parse-check + the end-to-end run in the runbook cover it.

- [ ] **Step 1: Create the script**

Create `scripts/pin_commercial_identities.py`:

```python
#!/usr/bin/env python3
"""Pin disjoint commercial Set A / Set B identity lists from an ingested root.

Run ONCE after scripts/prepare_commercial_bonafide.py has populated
datasets/_real/commercial_224/<identity>/NNN.png. Writes the two identity
files referenced by configs/runs/commercial_set*_d*.yaml:

    configs/commercial_identities_seta.txt   (8 identities)
    configs/commercial_identities_setb.txt   (next 16 identities)

Deterministic seeded shuffle (idempotent for a given ingested set); Set A and
Set B are identity-disjoint, matching the subject-disjoint discipline of the
DigiFace real_set* baselines.
"""

from __future__ import annotations

import argparse
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=REPO / "datasets/_real/commercial_224",
        help="Ingested commercial bonafide root (one dir per subject).",
    )
    ap.add_argument("--seta-count", type=int, default=8, help="Identities in Set A.")
    ap.add_argument("--setb-count", type=int, default=16, help="Identities in Set B.")
    ap.add_argument("--seed", type=int, default=20260528, help="Shuffle seed.")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(
            f"Commercial root not found: {args.root}\n"
            "Run scripts/prepare_commercial_bonafide.py first "
            "(see docs/commercial-bonafide.md)."
        )

    ids = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    needed = args.seta_count + args.setb_count
    if len(ids) < needed:
        raise SystemExit(
            f"Only {len(ids)} ingested identities under {args.root}; "
            f"need >= {needed} (Set A {args.seta_count} + Set B {args.setb_count})."
        )

    random.Random(args.seed).shuffle(ids)
    seta = sorted(ids[: args.seta_count])
    setb = sorted(ids[args.seta_count : args.seta_count + args.setb_count])

    seta_path = REPO / "configs/commercial_identities_seta.txt"
    setb_path = REPO / "configs/commercial_identities_setb.txt"
    seta_path.write_text("\n".join(seta) + "\n")
    setb_path.write_text("\n".join(setb) + "\n")

    print(f"pinned: {len(seta)} Set A -> {seta_path}")
    print(f"        {len(setb)} Set B -> {setb_path}")
    print("next: git add configs/commercial_identities_set*.txt && commit")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Parse-check the script**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/pin_commercial_identities.py').read()); print('pin script parses OK')"
```

Expected: `pin script parses OK`.

- [ ] **Step 3: Verify the no-data error path is clean**

```bash
.venv/bin/python scripts/pin_commercial_identities.py --root /tmp/does_not_exist_xyz; echo "exit=$?"
```

Expected: prints "Commercial root not found: …" and `exit=1`.

- [ ] **Step 4: Commit**

```bash
git add scripts/pin_commercial_identities.py
git commit -m "feat(pad-commercial): pin Set A/B identities from ingested root

Clone of pin_dfdc_identities.py retargeted to datasets/_real/commercial_224
and configs/commercial_identities_set*.txt. Deterministic disjoint 8/16 split."
```

---

## Task 3: Six sweep configs

**Files:**
- Create: `configs/runs/commercial_seta_d1.yaml`, `…seta_d2.yaml`, `…seta_d3.yaml`, `…setb_d1.yaml`, `…setb_d2.yaml`, `…setb_d3.yaml`

**Context:** Each is an exact mirror of the corresponding `configs/runs/real_<set>_d<n>.yaml`, changing only `run.name`, `run.output`, `bonafide.root` (→ `./datasets/_real/commercial_224`), and `bonafide.identities_file` (→ commercial list). Preserve seeds (Set A 20260522, Set B 20260523), `samples_per_bonafide` (Set A 6/32/256, Set B 4/32/256), print+replay attacks, and per-set sensor preset (Set A mobile-front-2024, Set B webcam-1080p). This matches `mix_set*` scale so the commercial sweep is directly comparable to the committed DigiFace `runs_mix_224_L4_A2` baseline.

- [ ] **Step 1: Create `configs/runs/commercial_seta_d1.yaml`**

```yaml
run:
  name: commercial_seta_d1
  output: ./datasets/commercial_seta_d1
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/commercial_224
  samples_per_bonafide: 6
  identities_file: ./configs/commercial_identities_seta.txt
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

- [ ] **Step 2: Create `configs/runs/commercial_seta_d2.yaml`** (identical to d1 except `name`/`output`/`samples_per_bonafide`)

```yaml
run:
  name: commercial_seta_d2
  output: ./datasets/commercial_seta_d2
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/commercial_224
  samples_per_bonafide: 32
  identities_file: ./configs/commercial_identities_seta.txt
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

- [ ] **Step 3: Create `configs/runs/commercial_seta_d3.yaml`**

```yaml
run:
  name: commercial_seta_d3
  output: ./datasets/commercial_seta_d3
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/commercial_224
  samples_per_bonafide: 256
  identities_file: ./configs/commercial_identities_seta.txt
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

- [ ] **Step 4: Create `configs/runs/commercial_setb_d1.yaml`**

```yaml
run:
  name: commercial_setb_d1
  output: ./datasets/commercial_setb_d1
  seed: 20260523
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/commercial_224
  samples_per_bonafide: 4
  identities_file: ./configs/commercial_identities_setb.txt
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

- [ ] **Step 5: Create `configs/runs/commercial_setb_d2.yaml`**

```yaml
run:
  name: commercial_setb_d2
  output: ./datasets/commercial_setb_d2
  seed: 20260523
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/commercial_224
  samples_per_bonafide: 32
  identities_file: ./configs/commercial_identities_setb.txt
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

- [ ] **Step 6: Create `configs/runs/commercial_setb_d3.yaml`**

```yaml
run:
  name: commercial_setb_d3
  output: ./datasets/commercial_setb_d3
  seed: 20260523
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/commercial_224
  samples_per_bonafide: 256
  identities_file: ./configs/commercial_identities_setb.txt
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

- [ ] **Step 7: Validate all six configs parse + match schema**

```bash
cd /Users/stuartwells/test
.venv/bin/python - <<'PY'
import yaml, pathlib
req = {"root","samples_per_bonafide","identities_file","splits"}
ok = True
for f in sorted(pathlib.Path("configs/runs").glob("commercial_set*_d*.yaml")):
    c = yaml.safe_load(f.read_text())
    checks = [
        c["run"]["name"] == f.stem,
        c["run"]["output"] == f"./datasets/{f.stem}",
        c["bonafide"]["root"] == "./datasets/_real/commercial_224",
        c["sensor_preset"] in ("mobile-front-2024","webcam-1080p"),
        set(c["attacks"]) == {"print","replay"},
        not (req - set(c["bonafide"])),
    ]
    if not all(checks): ok = False
    print(f"{f.name:26} preset={c['sensor_preset']:17} seed={c['run']['seed']} "
          f"spb={c['bonafide']['samples_per_bonafide']:>3}  {'OK' if all(checks) else 'FAIL '+str(checks)}")
print("\nALL GOOD" if ok else "\nPROBLEMS")
PY
```

Expected: 6 lines all `OK`, then `ALL GOOD`.

- [ ] **Step 8: Commit**

```bash
git add configs/runs/commercial_set*_d*.yaml
git commit -m "feat(pad-commercial): 6 sweep configs (mirror real_set*, bonafide swapped)

Exact mirrors of real_set*_d* with only the bonafide source changed to
datasets/_real/commercial_224 + commercial identity lists. Matches mix_set*
scale -> directly comparable to the DigiFace runs_mix_224_L4_A2 baseline."
```

---

## Task 4: Verdict script `compare_bonafide_eer.py`

**Files:**
- Create: `scripts/compare_bonafide_eer.py`
- Test: `tests/test_compare_bonafide_eer.py`

**Context:** The one genuinely new piece. Reads per-cell JSON from two sweep output dirs (each cell file lives at `<dir>/runs/<CAP>_<DLEVEL>_<seed>.json` and contains `capacity`, `data_level`, `seed`, `eer_cross_domain`, `n_train`, `n_val_cross_domain`). Aggregates by `(capacity, data_level)` → mean/std `eer_cross_domain` across seeds. Renders a per-cell table baseline-vs-commercial with `Δ`. **PASS** iff every shared cell has `|Δ| ≤ band` AND no commercial cell mean `≤ collapse`. Emits a matched-scale WARNING if `n_train`/`n_val_cross_domain` differ between the two dirs for any shared cell. `main()` exits non-zero on FAIL (CI-able). Core logic in importable functions so the test does not shell out.

- [ ] **Step 1: Write the failing verdict test**

Create `tests/test_compare_bonafide_eer.py`:

```python
"""Verdict logic for compare_bonafide_eer: matched-scale A/B EER delta."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compare_bonafide_eer",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_bonafide_eer.py",
)
cbe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cbe)


def _write_cell(d: Path, cap: str, dlevel: str, seed: int, eer: float,
               n_train: int = 384, n_val: int = 1024) -> None:
    runs = d / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{cap}_{dlevel}_{seed}.json").write_text(json.dumps({
        "capacity": cap, "data_level": dlevel, "seed": seed,
        "eer_cross_domain": eer, "n_train": n_train,
        "n_val_cross_domain": n_val,
    }))


def test_aggregate_groups_by_cell(tmp_path):
    d = tmp_path / "base"
    _write_cell(d, "L4", "D3", 0, 0.05)
    _write_cell(d, "L4", "D3", 1, 0.07)
    agg = cbe.aggregate(d)
    assert ("L4", "D3") in agg
    cell = agg[("L4", "D3")]
    assert abs(cell["mean"] - 0.06) < 1e-9
    assert cell["n_train"] == 384


def test_compare_passes_when_delta_small(tmp_path):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    for seed in (0, 1, 2):
        _write_cell(base, "L4", "D3", seed, 0.06)
        _write_cell(comm, "L4", "D3", seed, 0.07)  # delta 0.01 < band
    result = cbe.compare(cbe.aggregate(base), cbe.aggregate(comm),
                         band=0.03, collapse=0.001)
    assert result["passed"] is True
    assert result["rows"][0]["verdict"] == "ok"
    assert result["warnings"] == []


def test_compare_fails_on_large_delta_and_collapse(tmp_path):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    for seed in (0, 1, 2):
        _write_cell(base, "L4", "D2", seed, 0.06)
        _write_cell(comm, "L4", "D2", seed, 0.15)  # delta 0.09 > band
        _write_cell(base, "L4", "D3", seed, 0.06)
        _write_cell(comm, "L4", "D3", seed, 0.0005)  # collapsed
    result = cbe.compare(cbe.aggregate(base), cbe.aggregate(comm),
                         band=0.03, collapse=0.001)
    assert result["passed"] is False
    verdicts = {(r["capacity"], r["data_level"]): r["verdict"] for r in result["rows"]}
    assert verdicts[("L4", "D2")] == "delta_exceeds_band"
    assert verdicts[("L4", "D3")] == "collapsed"


def test_compare_warns_on_scale_mismatch(tmp_path):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    _write_cell(base, "L4", "D3", 0, 0.06, n_train=384)
    _write_cell(comm, "L4", "D3", 0, 0.06, n_train=48)  # different scale
    result = cbe.compare(cbe.aggregate(base), cbe.aggregate(comm),
                         band=0.03, collapse=0.001)
    assert any("scale" in w.lower() for w in result["warnings"])


def test_main_exits_nonzero_on_fail(tmp_path, capsys):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    _write_cell(base, "L4", "D3", 0, 0.06)
    _write_cell(comm, "L4", "D3", 0, 0.20)
    rc = cbe.main(["--baseline-dir", str(base), "--commercial-dir", str(comm)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest tests/test_compare_bonafide_eer.py -v
```

Expected: FAIL — `FileNotFoundError` / module load error because `scripts/compare_bonafide_eer.py` does not exist yet.

- [ ] **Step 3: Implement the verdict script**

Create `scripts/compare_bonafide_eer.py`:

```python
#!/usr/bin/env python3
"""Matched-scale A/B verdict: does swapping DigiFace bonafide for a
commercially-licensed set preserve cross-domain EER?

Reads two sweep output dirs (each <dir>/runs/<CAP>_<DLEVEL>_<seed>.json with
eer_cross_domain), aggregates by (capacity, data_level), and prints a per-cell
delta table. PASS iff every shared cell has |Δ| <= band AND no commercial cell
mean <= collapse. Exits non-zero on FAIL. See
docs/superpowers/specs/2026-06-02-pad-commercial-bonafide-validation-design.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
_DEFAULT_BASELINE = (
    REPO / "docs/superpowers/reports/2026-05-22-pad-spark-sweep-results"
    / "runs_mix_224_L4_A2"
)


def aggregate(sweep_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Group cell JSONs by (capacity, data_level) -> aggregated stats."""
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for f in sorted((Path(sweep_dir) / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        rows.setdefault((r["capacity"], r["data_level"]), []).append(r)
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rs in rows.items():
        eers = [r["eer_cross_domain"] for r in rs]
        agg[key] = {
            "mean": statistics.mean(eers),
            "std": statistics.pstdev(eers) if len(eers) > 1 else 0.0,
            "n_seeds": len(eers),
            "n_train": rs[0].get("n_train"),
            "n_val_cross_domain": rs[0].get("n_val_cross_domain"),
        }
    return agg


def compare(
    baseline: dict[tuple[str, str], dict[str, Any]],
    commercial: dict[tuple[str, str], dict[str, Any]],
    band: float,
    collapse: float,
) -> dict[str, Any]:
    """Build the per-cell verdict table + overall pass/fail + warnings."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    passed = True
    shared = sorted(set(baseline) & set(commercial))
    for cap, d in shared:
        b = baseline[(cap, d)]
        c = commercial[(cap, d)]
        delta = c["mean"] - b["mean"]
        if c["mean"] <= collapse:
            verdict = "collapsed"
            passed = False
        elif abs(delta) > band:
            verdict = "delta_exceeds_band"
            passed = False
        else:
            verdict = "ok"
        if b.get("n_train") != c.get("n_train") or \
           b.get("n_val_cross_domain") != c.get("n_val_cross_domain"):
            warnings.append(
                f"{cap}/{d}: scale mismatch (baseline n_train={b.get('n_train')}, "
                f"commercial n_train={c.get('n_train')}) — not matched-scale"
            )
        rows.append({
            "capacity": cap, "data_level": d,
            "baseline_mean": b["mean"], "commercial_mean": c["mean"],
            "delta": delta, "verdict": verdict,
        })
    only_base = sorted(set(baseline) - set(commercial))
    only_comm = sorted(set(commercial) - set(baseline))
    for cap, d in only_base:
        warnings.append(f"{cap}/{d}: present in baseline only — not compared")
    for cap, d in only_comm:
        warnings.append(f"{cap}/{d}: present in commercial only — not compared")
    if not shared:
        passed = False
        warnings.append("no shared cells between the two sweeps")
    return {"passed": passed, "rows": rows, "warnings": warnings}


def _render(result: dict[str, Any], band: float) -> str:
    lines = [
        f"Commercial-bonafide vs DigiFace baseline (band ±{band:.3f})",
        "",
        f"{'cell':<8} {'DigiFace':>9} {'Commercial':>11} {'Δ':>8}  verdict",
        "-" * 48,
    ]
    for r in result["rows"]:
        cell = f"{r['capacity']}·{r['data_level']}"
        lines.append(
            f"{cell:<8} {r['baseline_mean']:>9.3f} {r['commercial_mean']:>11.3f} "
            f"{r['delta']:>+8.3f}  {r['verdict']}"
        )
    for w in result["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append("")
    lines.append("PASS — commercial bonafide ships" if result["passed"]
                 else "FAIL — commercial bonafide does NOT preserve EER")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commercial-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=_DEFAULT_BASELINE)
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--collapse", type=float, default=0.001)
    args = ap.parse_args(argv)

    result = compare(
        aggregate(args.baseline_dir), aggregate(args.commercial_dir),
        band=args.band, collapse=args.collapse,
    )
    print(_render(result, args.band))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
.venv/bin/pytest tests/test_compare_bonafide_eer.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_bonafide_eer.py tests/test_compare_bonafide_eer.py
git commit -m "feat(pad-commercial): compare_bonafide_eer matched-scale A/B verdict

Aggregates two sweep dirs by (capacity, data_level), prints per-cell
cross-domain EER delta, PASS iff every cell |Δ|<=band and no collapse.
Warns on scale mismatch. main() exits non-zero on FAIL (CI-able)."
```

---

## Task 5: Operator runbook `docs/commercial-bonafide.md`

**Files:**
- Create: `docs/commercial-bonafide.md`

**Context:** The end-to-end runbook, structured like `docs/dfdc-bonafide.md`: obtain a free vendor sample → reshape into the canonical contract via a shim if needed → ingest with licence → pin identities → generate datasets + sweep on Spark → run the verdict. Calls out the licence-provenance requirement and the matched-scale caveat for small samples.

- [ ] **Step 1: Write the runbook**

Create `docs/commercial-bonafide.md`:

````markdown
# Commercial-bonafide retrain validation

Swaps the (non-commercial, research-only) DigiFace bonafide for a
**commercially-licensed** bonafide face set, runs the existing synthetic
attacks on top, and produces a PASS/FAIL verdict on whether cross-domain EER
holds. Validates a shippable model BEFORE buying data (use free vendor samples
first). Context + licensing rationale: memory `pad-commercial-licensing` and
spec `docs/superpowers/specs/2026-06-02-pad-commercial-bonafide-validation-design.md`.

**Real images are never committed** — `datasets/` is gitignored. Only the
licence string + source URL travel with the data via `provenance.jsonl`.

## 1. Obtain a commercially-licensed sample

Vendors with free, commercially-licensed samples: AxonData (axonlab.ai),
Nexdata, Unidata, Shaip. Request the free sample of a face anti-spoofing /
liveness set and note the exact licence string + the URL you downloaded from —
both are recorded into provenance at ingest.

## 2. Reshape into the canonical contract

The ingest expects `<src>/<identity>/<sample>.{png,jpg,jpeg}` — one directory
per subject. If the vendor ships a different layout (flat folder, parquet,
video), write a thin shim that reshapes it into that contract first. Example
for a flat folder where the filename prefix is the subject id
(`subjA_001.jpg`, `subjA_002.jpg`, `subjB_001.jpg`, …):

```bash
.venv/bin/python - <<'PY'
import pathlib, shutil
src = pathlib.Path("/path/to/vendor_flat")
dst = pathlib.Path("/path/to/vendor_canonical")
for img in src.glob("*.jpg"):
    subj = img.stem.split("_")[0]
    out = dst / subj
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out / img.name)
print("reshaped into", dst)
PY
```

## 3. Ingest → 224 bonafide root (records the licence)

```bash
.venv/bin/python scripts/prepare_commercial_bonafide.py \
  --src /path/to/vendor_canonical \
  --out datasets/_real/commercial_224 \
  --license "<exact vendor commercial licence string>" \
  --source-url "<the URL you downloaded from>" \
  --vendor axondata
```

Writes `datasets/_real/commercial_224/<identity>/NNN.png` + `provenance.jsonl`
+ `_meta.json`. Optional `--max-per-identity N`.

## 4. Pin Set A / Set B identities

```bash
.venv/bin/python scripts/pin_commercial_identities.py
git add configs/commercial_identities_set*.txt
git commit -m "feat(pad-commercial): pin commercial Set A/B identities"
```

Needs >= 24 ingested identities (8 Set A + 16 Set B). Override
`--seta-count`/`--setb-count`/`--seed` for a smaller sample.

## 5. Generate datasets + sweep on Spark

The six `configs/runs/commercial_set*_d*.yaml` are already committed. Generate
the synthetic datasets locally, rsync to the Spark, and run the L4 sweep —
identical to the DFDC/A2 procedure, `commercial_` prefixes:

```bash
for cfg in configs/runs/commercial_set{a,b}_d{1,2,3}.yaml; do
  ds=$(basename "$cfg" .yaml); rm -rf "datasets/$ds"
  .venv/bin/python -m pad_synth_face.cli generate --config "$cfg"
done
# rsync code + the 6 commercial_* datasets to the Spark, then run
# spark_sweep.py over the L4 cells -> runs_commercial_224_L4/ ; pull back.
```

**Matched-scale caveat:** the verdict compares against the DigiFace
`runs_mix_224_L4_A2` baseline, which used `samples_per_bonafide` 6/32/256. If
your sample is too small for D3 (256/bonafide), reduce `samples_per_bonafide`
*symmetrically* in both the commercial configs and a re-run DigiFace sweep, and
point `--baseline-dir` at that matched DigiFace run. The verdict script prints a
scale-mismatch WARNING if the two sweeps don't actually match.

## 6. Verdict

```bash
.venv/bin/python scripts/compare_bonafide_eer.py \
  --commercial-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_commercial_224_L4 \
  --baseline-dir   docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mix_224_L4_A2
```

Prints the per-cell Δ table and `PASS`/`FAIL`. PASS (every cell |Δ| ≤ 0.03, no
collapse) means the commercial bonafide preserves EER → a shippable model is
viable and the ~$10k purchase is de-risked. FAIL means investigate before
buying. Append the table to
`docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.
````

- [ ] **Step 2: Commit**

```bash
git add docs/commercial-bonafide.md
git commit -m "docs(pad-commercial): operator runbook obtain->ingest->pin->sweep->verdict"
```

---

## Task 6: Full-suite regression + branch finish

**Files:**
- Modify (only if something breaks): whatever a failure points to

**Context:** Catch-all check that the new modules/tests integrate cleanly and nothing else regressed.

- [ ] **Step 1: Run the new tests + the full repo suite**

```bash
cd /Users/stuartwells/test
.venv/bin/pytest pad-synth-face/tests/test_commercial_bonafide_ingest.py \
                 tests/test_compare_bonafide_eer.py -v
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -q
```

Expected: the two new test files pass (4 + 5); full suite green.

- [ ] **Step 2: Confirm the harness is inert without data (no accidental coupling)**

```bash
ls datasets/_real/commercial_224 2>/dev/null && echo "UNEXPECTED: data present" || echo "OK: no commercial data staged (harness is pure scaffolding)"
git status --short
```

Expected: no `datasets/_real/commercial_224`; `git status` clean (all work committed).

- [ ] **Step 3: Review the commit history**

```bash
git log --oneline feat/pad-commercial-bonafide-validation ^main
```

Expected: ~6 commits (spec + one per Task 1–5).

- [ ] **Step 4: Finish the branch**

Hand off to `superpowers:finishing-a-development-branch` to merge to local `main` (the user's established pattern) or open a PR, per the user's choice at that time.

---

## Final Verification

From `/Users/stuartwells/test`:

```bash
.venv/bin/pytest pad-synth-face/tests/ pad-synth-core/tests/ tests/ -q
.venv/bin/python - <<'PY'
import yaml, pathlib
n = len(list(pathlib.Path("configs/runs").glob("commercial_set*_d*.yaml")))
print("commercial configs:", n)
assert n == 6
PY
```

Expected: all tests green; 6 commercial configs present. The harness then sits inert until a commercial sample is staged, at which point `docs/commercial-bonafide.md` is the turnkey path to a verdict.
