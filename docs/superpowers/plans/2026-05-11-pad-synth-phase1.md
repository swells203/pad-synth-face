# PAD Synthetic Dataset — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vertical end-to-end slice of the PAD synthetic dataset pipeline: `pad-synth-core` (manifest, RNG, ontology, orchestrator, QC) plus `pad-synth-face` with two physics-based attack modules (print + replay), capable of generating a small reproducible labelled dataset with a determinism CI guarantee.

**Architecture:** Python 3.11 monorepo with two packages (`pad-synth-core`, `pad-synth-face`). Every sample is byte-exact regenerable from `(master_seed, sample_index)`. Attacks are pluggable modules drawing parameters from a literature-cited YAML ontology. Sensor physics is applied after the attack-specific simulation. JSONL manifest + JSONL provenance ledger record every sample's full lineage.

**Tech Stack:**
- Python 3.11+
- `uv` for dependency management (falls back cleanly to `pip` if not installed)
- `pydantic` v2 for schemas
- `numpy`, `Pillow`, `opencv-python` for image processing
- `pyyaml` for ontology
- `pytest` for tests
- `ruff` for lint
- GitHub Actions for CI

**Phase 1 scope deliberately defers:**
- Real DigiFace-1M ingestion (use a small synthetic test fixture; the loader interface is real, the dataset bytes are not)
- Real CelebA-Spoof eval slice access (the cross-domain eval *scaffold* is built; an `--eval-real` flag is wired up but the default fixture is synthetic)
- Mask, deepfake, all voice attacks (Phases 2–3)
- Distributed multi-machine orchestration (Phase 4)

---

## Repository Layout

```
test/                                    # repo root (existing)
├── docs/superpowers/                    # specs + plans (existing)
├── pyproject.toml                       # workspace root
├── ruff.toml
├── pad-synth-core/
│   ├── pyproject.toml
│   ├── src/pad_synth_core/
│   │   ├── __init__.py
│   │   ├── rng.py
│   │   ├── manifest.py
│   │   ├── provenance.py
│   │   ├── ontology.py
│   │   ├── orchestrator.py
│   │   ├── cli.py
│   │   └── qc/
│   │       ├── __init__.py
│   │       ├── per_sample.py
│   │       └── distribution.py
│   └── tests/
├── pad-synth-face/
│   ├── pyproject.toml
│   ├── src/pad_synth_face/
│   │   ├── __init__.py
│   │   ├── bonafide.py
│   │   ├── sensor.py
│   │   ├── pipeline.py
│   │   ├── cli.py
│   │   └── attacks/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── print.py
│   │       └── replay.py
│   └── tests/
├── ontology/
│   └── face/
│       ├── print.yaml
│       └── replay.yaml
├── configs/
│   └── runs/
│       └── phase1_smoke.yaml
├── tests/
│   └── golden/
│       ├── README.md
│       └── golden_manifest.jsonl       # generated, checked in
└── .github/workflows/
    └── ci.yaml
```

---

## Task 1: Repo & Workspace Scaffolding

**Files:**
- Create: `pyproject.toml` (workspace root)
- Create: `ruff.toml`
- Create: `pad-synth-core/pyproject.toml`
- Create: `pad-synth-core/src/pad_synth_core/__init__.py`
- Create: `pad-synth-core/tests/__init__.py`
- Create: `pad-synth-core/tests/test_smoke.py`
- Create: `pad-synth-face/pyproject.toml`
- Create: `pad-synth-face/src/pad_synth_face/__init__.py`
- Create: `pad-synth-face/tests/__init__.py`
- Create: `pad-synth-face/tests/test_smoke.py`
- Create: `.gitignore`

- [ ] **Step 1: Write the failing smoke tests**

`pad-synth-core/tests/test_smoke.py`:
```python
def test_import_core():
    import pad_synth_core
    assert pad_synth_core.__version__ == "0.1.0"
```

`pad-synth-face/tests/test_smoke.py`:
```python
def test_import_face():
    import pad_synth_face
    assert pad_synth_face.__version__ == "0.1.0"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/stuartwells/test
python -m pytest pad-synth-core/tests/test_smoke.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'pad_synth_core'` (packages not installed yet).

- [ ] **Step 3: Create workspace pyproject.toml**

`pyproject.toml`:
```toml
[tool.uv.workspace]
members = ["pad-synth-core", "pad-synth-face"]

[tool.pytest.ini_options]
testpaths = ["pad-synth-core/tests", "pad-synth-face/tests", "tests"]
python_files = ["test_*.py"]
addopts = "-ra"
```

`ruff.toml`:
```toml
line-length = 100
target-version = "py311"

[lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]
```

`.gitignore`:
```
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
*.egg-info/
dist/
build/
datasets/
```

- [ ] **Step 4: Create core package pyproject.toml**

`pad-synth-core/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pad-synth-core"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]
```

`pad-synth-core/src/pad_synth_core/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Create face package pyproject.toml**

`pad-synth-face/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pad-synth-face"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pad-synth-core",
    "pillow>=10.0",
    "opencv-python>=4.9",
    "numpy>=1.26",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[tool.uv.sources]
pad-synth-core = { workspace = true }
```

`pad-synth-face/src/pad_synth_face/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 6: Install workspace and verify tests pass**

```bash
cd /Users/stuartwells/test
uv sync --all-extras 2>&1 | tail -10
# or fallback:
# python -m venv .venv && .venv/bin/pip install -e pad-synth-core[test] -e pad-synth-face[test]
.venv/bin/python -m pytest -v 2>&1 | tail -20
```

Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml ruff.toml .gitignore pad-synth-core pad-synth-face
git commit -m "feat: scaffold pad-synth-core and pad-synth-face packages"
```

---

## Task 2: Seeded RNG (deterministic sample-seed derivation)

The orchestrator owns a master seed; each sample gets a derived seed from `(master_seed, modality, attack_type, sample_index)`. The derivation must be reproducible across machines and Python versions — we use SHA-256, not the standard library's hash().

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/rng.py`
- Create: `pad-synth-core/tests/test_rng.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/tests/test_rng.py`:
```python
import numpy as np
import pytest

from pad_synth_core.rng import derive_sample_seed, sample_rng


def test_derive_sample_seed_is_deterministic():
    a = derive_sample_seed(42, "face", "print", 0)
    b = derive_sample_seed(42, "face", "print", 0)
    assert a == b


def test_derive_sample_seed_varies_with_inputs():
    s1 = derive_sample_seed(42, "face", "print", 0)
    s2 = derive_sample_seed(42, "face", "print", 1)
    s3 = derive_sample_seed(42, "face", "replay", 0)
    s4 = derive_sample_seed(43, "face", "print", 0)
    assert len({s1, s2, s3, s4}) == 4


def test_derive_sample_seed_fits_in_uint32():
    for i in range(100):
        s = derive_sample_seed(42, "face", "print", i)
        assert 0 <= s < 2**32


def test_sample_rng_is_seeded_numpy_generator():
    rng = sample_rng(123)
    val = rng.integers(0, 1000)
    rng2 = sample_rng(123)
    val2 = rng2.integers(0, 1000)
    assert val == val2


def test_sample_rng_rejects_unseeded_use():
    with pytest.raises(TypeError):
        sample_rng()  # type: ignore[call-arg]
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_rng.py -v 2>&1 | tail -20
```

Expected: ImportError / ModuleNotFoundError on `pad_synth_core.rng`.

- [ ] **Step 3: Implement the RNG module**

`pad-synth-core/src/pad_synth_core/rng.py`:
```python
"""Deterministic seed derivation and RNG construction.

The orchestrator owns a single master seed. Every sample's randomness is
derived from (master_seed, modality, attack_type, sample_index) via SHA-256
so the derivation is reproducible across Python versions and machines.
"""

import hashlib

import numpy as np


def derive_sample_seed(
    master_seed: int, modality: str, attack_type: str, sample_index: int
) -> int:
    payload = f"{master_seed}|{modality}|{attack_type}|{sample_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def sample_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_rng.py -v 2>&1 | tail -20
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/rng.py pad-synth-core/tests/test_rng.py
git commit -m "feat(core): deterministic sample-seed derivation via SHA-256"
```

---

## Task 3: Sample Manifest Schema + Writer

The manifest is JSONL — one row per sample, append-only, single writer, fsync per batch. Schema is enforced via pydantic v2.

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/manifest.py`
- Create: `pad-synth-core/tests/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/tests/test_manifest.py`:
```python
import json
from pathlib import Path

import pytest

from pad_synth_core.manifest import BonafideSource, ManifestWriter, SampleRecord


def make_record(sample_id: str = "face-print-test001") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        modality="face",
        label="attack",
        attack_type="print",
        bonafide_source=BonafideSource(
            dataset="digiface_1m_fixture", id="00000001", license="MIT"
        ),
        attack_params={"paper_type": "matte", "print_dpi": 600},
        sensor_preset="mobile-front-2024",
        sensor_params={"iso": 200, "jpeg_qf": 90},
        generators_used=[],
        pipeline_version="pad-synth-face@0.1.0",
        core_version="pad-synth-core@0.1.0",
        ontology_version="ontology@2026-05-11",
        seed=1234,
        output_path="face/print/face-print-test001.jpg",
        output_sha256="0" * 64,
    )


def test_sample_record_serializes_to_json():
    rec = make_record()
    blob = rec.model_dump_json()
    parsed = json.loads(blob)
    assert parsed["sample_id"] == "face-print-test001"
    assert parsed["bonafide_source"]["license"] == "MIT"


def test_sample_record_rejects_bad_label():
    with pytest.raises(ValueError):
        make_record().model_copy(update={"label": "not-a-label"})


def test_manifest_writer_appends_jsonl(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    writer = ManifestWriter(path)
    writer.append(make_record("a"))
    writer.append(make_record("b"))
    writer.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["sample_id"] == "a"
    assert json.loads(lines[1])["sample_id"] == "b"


def test_manifest_writer_is_resumable(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"

    w1 = ManifestWriter(path)
    w1.append(make_record("a"))
    w1.close()

    w2 = ManifestWriter(path)
    assert w2.existing_sample_ids() == {"a"}
    w2.append(make_record("b"))
    w2.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_manifest.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_core.manifest`.

- [ ] **Step 3: Implement the manifest module**

`pad-synth-core/src/pad_synth_core/manifest.py`:
```python
"""Sample manifest schema and append-only JSONL writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BonafideSource(BaseModel):
    dataset: str
    id: str
    license: str
    url: str | None = None


class GeneratorUsage(BaseModel):
    name: str
    version: str
    license: str
    commercial_ok: bool
    model_hash: str | None = None


class SampleRecord(BaseModel):
    sample_id: str
    modality: Literal["face", "voice"]
    label: Literal["bonafide", "attack"]
    attack_type: str | None
    bonafide_source: BonafideSource
    attack_params: dict[str, Any] = Field(default_factory=dict)
    sensor_preset: str | None = None
    sensor_params: dict[str, Any] = Field(default_factory=dict)
    generators_used: list[GeneratorUsage] = Field(default_factory=list)
    pipeline_version: str
    core_version: str
    ontology_version: str
    seed: int
    output_path: str
    output_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ManifestWriter:
    """Append-only JSONL writer for sample manifests.

    Reads existing sample_ids on open so callers can skip already-completed work.
    Caller is responsible for never instantiating two writers on the same path
    simultaneously.
    """

    def __init__(self, path: Path, fsync_every: int = 100) -> None:
        self.path = Path(path)
        self.fsync_every = fsync_every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing: set[str] = self._scan_existing()
        self._fh = self.path.open("a", encoding="utf-8")
        self._written_since_fsync = 0

    def _scan_existing(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["sample_id"])
        return ids

    def existing_sample_ids(self) -> set[str]:
        return set(self._existing)

    def append(self, record: SampleRecord) -> None:
        if record.sample_id in self._existing:
            return
        self._fh.write(record.model_dump_json() + "\n")
        self._existing.add(record.sample_id)
        self._written_since_fsync += 1
        if self._written_since_fsync >= self.fsync_every:
            self._fsync()

    def _fsync(self) -> None:
        self._fh.flush()
        import os
        os.fsync(self._fh.fileno())
        self._written_since_fsync = 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fsync()
            self._fh.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_manifest.py -v 2>&1 | tail -20
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/manifest.py pad-synth-core/tests/test_manifest.py
git commit -m "feat(core): sample manifest schema with resumable JSONL writer"
```

---

## Task 4: Provenance Ledger

Dataset-level audit trail. Separate JSONL from the sample manifest. Tracks bonafide-dataset ingestion, generator registration, and ontology citations.

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/provenance.py`
- Create: `pad-synth-core/tests/test_provenance.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/tests/test_provenance.py`:
```python
import json
from pathlib import Path

from pad_synth_core.provenance import (
    BonafideIngested,
    GeneratorRegistered,
    OntologyCitation,
    ProvenanceLedger,
)


def test_ledger_records_bonafide_ingestion(tmp_path: Path):
    ledger = ProvenanceLedger(tmp_path / "provenance.jsonl")
    ledger.record(
        BonafideIngested(
            name="digiface_1m_fixture",
            license="MIT",
            source_url="local-fixture",
            sha256_of_index="abc123",
        )
    )
    ledger.close()
    lines = (tmp_path / "provenance.jsonl").read_text().strip().split("\n")
    parsed = json.loads(lines[0])
    assert parsed["type"] == "bonafide_dataset_ingested"
    assert parsed["name"] == "digiface_1m_fixture"


def test_ledger_records_multiple_event_types(tmp_path: Path):
    ledger = ProvenanceLedger(tmp_path / "provenance.jsonl")
    ledger.record(
        BonafideIngested(name="x", license="MIT", source_url="u", sha256_of_index="h")
    )
    ledger.record(
        GeneratorRegistered(
            name="g", version="1.0", license="MIT", commercial_ok=True, model_hash="h"
        )
    )
    ledger.record(
        OntologyCitation(
            attack_type="print",
            axis="paper_type",
            paper="Example 2024",
            doi="10.0/test",
        )
    )
    ledger.close()
    lines = (tmp_path / "provenance.jsonl").read_text().strip().split("\n")
    types = [json.loads(line)["type"] for line in lines]
    assert types == [
        "bonafide_dataset_ingested",
        "generator_registered",
        "ontology_citation",
    ]
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_provenance.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_core.provenance`.

- [ ] **Step 3: Implement the provenance module**

`pad-synth-core/src/pad_synth_core/provenance.py`:
```python
"""Append-only provenance ledger for dataset-level audit trail."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BonafideIngested(BaseModel):
    type: Literal["bonafide_dataset_ingested"] = "bonafide_dataset_ingested"
    name: str
    license: str
    source_url: str
    sha256_of_index: str
    ingested_at: datetime = Field(default_factory=_now)


class GeneratorRegistered(BaseModel):
    type: Literal["generator_registered"] = "generator_registered"
    name: str
    version: str
    license: str
    commercial_ok: bool
    model_hash: str
    registered_at: datetime = Field(default_factory=_now)


class OntologyCitation(BaseModel):
    type: Literal["ontology_citation"] = "ontology_citation"
    attack_type: str
    axis: str
    paper: str
    doi: str | None = None
    url: str | None = None


ProvenanceEvent = BonafideIngested | GeneratorRegistered | OntologyCitation


class ProvenanceLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def record(self, event: ProvenanceEvent) -> None:
        self._fh.write(event.model_dump_json() + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> "ProvenanceLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_provenance.py -v 2>&1 | tail -20
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/provenance.py pad-synth-core/tests/test_provenance.py
git commit -m "feat(core): provenance ledger for dataset audit trail"
```

---

## Task 5: Ontology Loader + Linter

YAML files define each attack's parameter axes with literature citations. The linter enforces every parameter range has a `provenance` field.

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/ontology.py`
- Create: `pad-synth-core/tests/test_ontology.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/tests/test_ontology.py`:
```python
from pathlib import Path

import pytest

from pad_synth_core.ontology import (
    Ontology,
    OntologyLintError,
    load_ontology,
)


GOOD_YAML = """
version: "2026-05-11"
attack_type: print
axes:
  paper_type:
    type: categorical
    values: [matte, glossy, photo]
    weights: [0.5, 0.4, 0.1]
    provenance:
      paper: "Galbally 2014"
      doi: "10.0/example"
  print_dpi:
    type: categorical
    values: [150, 300, 600, 1200]
    weights: [0.1, 0.4, 0.4, 0.1]
    provenance:
      paper: "Example Vendor Spec 2023"
      url: "https://example.com/spec"
  tilt_degrees:
    type: uniform
    low: -30.0
    high: 30.0
    provenance:
      paper: "Boulkenafet 2017 OULU-NPU paper"
      doi: "10.0/oulu"
"""


BAD_YAML_NO_PROVENANCE = """
version: "2026-05-11"
attack_type: print
axes:
  paper_type:
    type: categorical
    values: [matte, glossy]
    weights: [0.5, 0.5]
"""


def test_load_ontology_parses_axes(tmp_path: Path):
    p = tmp_path / "print.yaml"
    p.write_text(GOOD_YAML)
    ont = load_ontology(p)
    assert ont.attack_type == "print"
    assert "paper_type" in ont.axes
    assert ont.version == "2026-05-11"


def test_lint_rejects_axis_without_provenance(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(BAD_YAML_NO_PROVENANCE)
    with pytest.raises(OntologyLintError) as exc:
        load_ontology(p)
    assert "paper_type" in str(exc.value)
    assert "provenance" in str(exc.value)


def test_sample_categorical_is_deterministic(tmp_path: Path):
    p = tmp_path / "print.yaml"
    p.write_text(GOOD_YAML)
    ont = load_ontology(p)
    import numpy as np

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    params1 = ont.sample_params(rng1)
    params2 = ont.sample_params(rng2)
    assert params1 == params2
    assert params1["paper_type"] in {"matte", "glossy", "photo"}
    assert -30.0 <= params1["tilt_degrees"] <= 30.0
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_ontology.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_core.ontology`.

- [ ] **Step 3: Implement the ontology module**

`pad-synth-core/src/pad_synth_core/ontology.py`:
```python
"""Attack-parameter ontology with literature-citation enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator


class OntologyLintError(ValueError):
    pass


class Provenance(BaseModel):
    paper: str
    doi: str | None = None
    url: str | None = None


class Axis(BaseModel):
    type: str
    provenance: Provenance
    values: list[Any] | None = None
    weights: list[float] | None = None
    low: float | None = None
    high: float | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "Axis":
        if self.type == "categorical":
            if not self.values or not self.weights:
                raise OntologyLintError("categorical axis needs values and weights")
            if len(self.values) != len(self.weights):
                raise OntologyLintError("values and weights length mismatch")
        elif self.type == "uniform":
            if self.low is None or self.high is None:
                raise OntologyLintError("uniform axis needs low and high")
        else:
            raise OntologyLintError(f"unknown axis type {self.type!r}")
        return self


class Ontology(BaseModel):
    version: str
    attack_type: str
    axes: dict[str, Axis]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, axis in self.axes.items():
            if axis.type == "categorical":
                idx = rng.choice(len(axis.values), p=np.array(axis.weights, dtype=float))
                value = axis.values[int(idx)]
            else:  # uniform
                value = float(rng.uniform(axis.low, axis.high))
            out[name] = value
        return out


def load_ontology(path: Path) -> Ontology:
    raw = yaml.safe_load(Path(path).read_text())
    # Pre-lint: catch missing provenance with a clear error referencing the axis.
    axes = raw.get("axes", {})
    for axis_name, axis_data in axes.items():
        if "provenance" not in axis_data:
            raise OntologyLintError(
                f"axis {axis_name!r} in {path}: missing required 'provenance' field"
            )
    return Ontology.model_validate(raw)
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_ontology.py -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/ontology.py pad-synth-core/tests/test_ontology.py
git commit -m "feat(core): ontology loader with literature-citation lint"
```

---

## Task 6: Ontology YAML files for print and replay

The actual literature-cited ontology data. The DOIs/papers listed are real published works on PAD; the precise distributions used here are reasonable Phase 1 starting values consistent with their findings — they can be refined as the simulator matures.

**Files:**
- Create: `ontology/face/print.yaml`
- Create: `ontology/face/replay.yaml`
- Create: `tests/test_ontology_files.py`

- [ ] **Step 1: Write the failing tests**

`tests/__init__.py` (empty file, just to mark a package):
```python
```

`tests/test_ontology_files.py`:
```python
from pathlib import Path

from pad_synth_core.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_print_ontology_loads():
    ont = load_ontology(REPO_ROOT / "ontology" / "face" / "print.yaml")
    assert ont.attack_type == "print"
    assert "paper_type" in ont.axes
    assert "print_dpi" in ont.axes


def test_replay_ontology_loads():
    ont = load_ontology(REPO_ROOT / "ontology" / "face" / "replay.yaml")
    assert ont.attack_type == "replay"
    assert "device_class" in ont.axes
    assert "refresh_hz" in ont.axes
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_ontology_files.py -v 2>&1 | tail -20
```

Expected: FileNotFoundError on the YAML files.

- [ ] **Step 3: Write the print ontology YAML**

`ontology/face/print.yaml`:
```yaml
version: "2026-05-11"
attack_type: print
axes:
  paper_type:
    type: categorical
    values: [matte, glossy, photo]
    weights: [0.45, 0.35, 0.20]
    provenance:
      paper: "Galbally et al., 'Biometric Antispoofing Methods: A Survey in Face Recognition', IEEE Access 2014"
      doi: "10.1109/ACCESS.2014.2381273"
  print_dpi:
    type: categorical
    values: [150, 300, 600, 1200]
    weights: [0.10, 0.40, 0.40, 0.10]
    provenance:
      paper: "Consumer inkjet/laser printer spec ranges; corroborated by Pereira et al., 'LBP-TOP based countermeasure against face spoofing attacks', ACCV 2012 Workshops"
      doi: "10.1007/978-3-642-37410-4_11"
  tilt_degrees:
    type: uniform
    low: -25.0
    high: 25.0
    provenance:
      paper: "Boulkenafet et al., 'OULU-NPU: A mobile face presentation attack database with real-world variations', FG 2017"
      doi: "10.1109/FG.2017.77"
  holder_present:
    type: categorical
    values: [true, false]
    weights: [0.6, 0.4]
    provenance:
      paper: "Chingovska et al., 'On the Effectiveness of Local Binary Patterns in Face Anti-spoofing', BIOSIG 2012 (Idiap Replay-Attack composition)"
      url: "https://www.idiap.ch/dataset/replayattack"
  cutout:
    type: categorical
    values: [none, eyes, eyes_mouth]
    weights: [0.7, 0.2, 0.1]
    provenance:
      paper: "Zhang et al., 'A Face Antispoofing Database with Diverse Attacks', ICB 2012 (CASIA-FASD attack design)"
      doi: "10.1109/ICB.2012.6199754"
```

- [ ] **Step 4: Write the replay ontology YAML**

`ontology/face/replay.yaml`:
```yaml
version: "2026-05-11"
attack_type: replay
axes:
  device_class:
    type: categorical
    values: [phone_oled, phone_lcd, tablet, laptop, desktop_monitor]
    weights: [0.30, 0.25, 0.20, 0.15, 0.10]
    provenance:
      paper: "Boulkenafet et al., OULU-NPU 2017 (device mix in mobile capture scenarios)"
      doi: "10.1109/FG.2017.77"
  bezel_pct:
    type: uniform
    low: 1.0
    high: 12.0
    provenance:
      paper: "Device teardown spec aggregation; consistent with screen-to-body ratios documented in iFixit teardowns 2020-2024"
      url: "https://www.ifixit.com/Device"
  viewing_angle:
    type: uniform
    low: -35.0
    high: 35.0
    provenance:
      paper: "Patel et al., 'Secure Face Unlock: Spoof Detection on Smartphones', IEEE TIFS 2016 (replay capture geometry)"
      doi: "10.1109/TIFS.2016.2559512"
  refresh_hz:
    type: categorical
    values: [60, 90, 120, 144]
    weights: [0.55, 0.15, 0.20, 0.10]
    provenance:
      paper: "Common consumer display refresh rates; informed by replay-attack moire literature in Galbally 2014"
      doi: "10.1109/ACCESS.2014.2381273"
  ambient_reflection:
    type: uniform
    low: 0.0
    high: 0.45
    provenance:
      paper: "Wen et al., 'Face Spoof Detection With Image Distortion Analysis', IEEE TIFS 2015"
      doi: "10.1109/TIFS.2015.2400395"
```

- [ ] **Step 5: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_ontology_files.py -v 2>&1 | tail -20
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add ontology tests/__init__.py tests/test_ontology_files.py
git commit -m "feat(ontology): print and replay axes with literature citations"
```

---

## Task 7: DigiFace Bonafide Loader (with test fixture)

The interface is real and stable; the data is a tiny generated fixture (16 procedural face-like RGB images) so Phase 1 doesn't depend on the multi-GB DigiFace-1M download. The real loader plugs into the same interface in Phase 2.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/bonafide.py`
- Create: `pad-synth-face/tests/conftest.py`
- Create: `pad-synth-face/tests/test_bonafide.py`

- [ ] **Step 1: Write the failing tests + fixture**

`pad-synth-face/tests/conftest.py`:
```python
"""Shared test fixtures: a tiny on-disk bonafide dataset for fast tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fixture_bonafide_dir(tmp_path: Path) -> Path:
    """Create 16 procedural 'face-like' RGB images on disk, identity 0..7 × 2 each."""
    root = tmp_path / "digiface_fixture"
    root.mkdir()
    rng = np.random.default_rng(0)
    for identity in range(8):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir()
        for sample in range(2):
            # 64x64 RGB blob biased toward a per-identity color (cheap fake "face")
            base = rng.integers(50, 200, size=3)
            arr = np.tile(base, (64, 64, 1)).astype(np.uint8)
            noise = rng.integers(-20, 20, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root
```

`pad-synth-face/tests/test_bonafide.py`:
```python
from pathlib import Path

import numpy as np

from pad_synth_face.bonafide import DigiFaceLoader


def test_loader_lists_identities(fixture_bonafide_dir: Path):
    loader = DigiFaceLoader(fixture_bonafide_dir)
    ids = loader.list_identities()
    assert len(ids) == 8
    assert ids == sorted(ids)


def test_loader_loads_image_as_uint8_rgb(fixture_bonafide_dir: Path):
    loader = DigiFaceLoader(fixture_bonafide_dir)
    identity = loader.list_identities()[0]
    samples = loader.samples_for_identity(identity)
    assert len(samples) == 2
    arr = loader.load(samples[0])
    assert arr.dtype == np.uint8
    assert arr.shape == (64, 64, 3)


def test_identity_disjoint_split_is_deterministic(fixture_bonafide_dir: Path):
    loader = DigiFaceLoader(fixture_bonafide_dir)
    split_a = loader.identity_disjoint_split(seed=42, ratios=(0.5, 0.25, 0.25))
    split_b = loader.identity_disjoint_split(seed=42, ratios=(0.5, 0.25, 0.25))
    assert split_a == split_b
    all_ids = set(loader.list_identities())
    train, dev, test = split_a
    assert set(train) | set(dev) | set(test) == all_ids
    assert not (set(train) & set(dev))
    assert not (set(train) & set(test))
    assert not (set(dev) & set(test))
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_bonafide.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_face.bonafide`.

- [ ] **Step 3: Implement the loader**

`pad-synth-face/src/pad_synth_face/bonafide.py`:
```python
"""Bonafide-face loaders.

The fixture-shaped on-disk layout is `<root>/<identity_id>/<sample_index>.png`.
The DigiFace-1M release follows the same identity-per-directory shape, so the
same loader implementation works for both fixture and real data in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BonafideSample:
    identity: str
    path: Path


class DigiFaceLoader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def list_identities(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def samples_for_identity(self, identity: str) -> list[BonafideSample]:
        identity_dir = self.root / identity
        return [
            BonafideSample(identity=identity, path=p)
            for p in sorted(identity_dir.glob("*.png"))
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

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_bonafide.py -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/bonafide.py pad-synth-face/tests/conftest.py pad-synth-face/tests/test_bonafide.py
git commit -m "feat(face): bonafide loader interface with fixture for fast tests"
```

---

## Task 8: Face Attack Protocol + Print Attack Module (MVP)

The Phase 1 print module implements a deliberately simple physics chain — paper-texture multiply, perspective warp, optional grayscale-cutout — that demonstrates the architecture. Halftoning, full ICC profiling, and specular highlights are tagged for Phase 2 enhancement.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/attacks/__init__.py`
- Create: `pad-synth-face/src/pad_synth_face/attacks/base.py`
- Create: `pad-synth-face/src/pad_synth_face/attacks/print.py`
- Create: `pad-synth-face/tests/test_print_attack.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-face/src/pad_synth_face/attacks/__init__.py`:
```python
```

`pad-synth-face/tests/test_print_attack.py`:
```python
from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.print import PrintAttack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ontology():
    return load_ontology(REPO_ROOT / "ontology" / "face" / "print.yaml")


def test_print_attack_returns_same_shape_uint8():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = PrintAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.dtype == np.uint8
    assert out.shape == bonafide.shape


def test_print_attack_actually_modifies_the_image():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = PrintAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    # The output should not be byte-identical to the input
    assert not np.array_equal(out, bonafide)


def test_print_attack_is_deterministic_under_same_seed():
    bonafide = np.full((64, 64, 3), 128, dtype=np.uint8)
    attack = PrintAttack(_ontology())

    rng1 = sample_rng(99)
    params1 = attack.sample_params(rng1)
    out1 = attack.simulate(bonafide, params1, rng1)

    rng2 = sample_rng(99)
    params2 = attack.sample_params(rng2)
    out2 = attack.simulate(bonafide, params2, rng2)

    assert params1 == params2
    assert np.array_equal(out1, out2)
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_print_attack.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_face.attacks.print`.

- [ ] **Step 3: Implement the attack protocol**

`pad-synth-face/src/pad_synth_face/attacks/base.py`:
```python
"""Protocol every face attack module implements."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from pad_synth_core.ontology import Ontology


class FaceAttackModule(Protocol):
    name: str
    ontology: Ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]: ...

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray: ...
```

- [ ] **Step 4: Implement the print attack**

`pad-synth-face/src/pad_synth_face/attacks/print.py`:
```python
"""Phase 1 print-attack simulator.

Pipeline (MVP):
  1. Paper-color tint (matte/glossy/photo per ontology)
  2. Paper-texture multiply (procedural grain)
  3. Perspective warp simulating a tilted printed page
  4. Optional cutout (eyes / eyes+mouth) by zeroing pixels in the cut regions

The DPI axis is currently informational (recorded in params, not yet used to
band-limit). Halftoning, ICC profiling, and anisotropic specular highlights
are explicitly Phase 2 work.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pad_synth_core.ontology import Ontology


_PAPER_TINTS: dict[str, tuple[float, float, float]] = {
    "matte": (0.96, 0.95, 0.92),
    "glossy": (1.02, 1.02, 1.04),
    "photo": (1.00, 0.99, 0.97),
}


def _paper_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=1.0, scale=0.03, size=(h, w, 1))
    return np.clip(noise, 0.85, 1.10).astype(np.float32)


def _perspective_warp(
    img: np.ndarray, tilt_degrees: float, rng: np.random.Generator
) -> np.ndarray:
    h, w = img.shape[:2]
    shift = int(abs(tilt_degrees) / 25.0 * (w * 0.10))
    sign = 1 if tilt_degrees >= 0 else -1
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array(
        [
            [shift * sign, 0],
            [w - shift * sign, shift // 2],
            [w, h],
            [0, h - shift // 2],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _apply_cutout(img: np.ndarray, cutout: str) -> np.ndarray:
    if cutout == "none":
        return img
    out = img.copy()
    h, w = out.shape[:2]
    # Cheap fixed-position cutouts; consistent with "wearable print" attacks.
    eye_y1, eye_y2 = int(h * 0.30), int(h * 0.45)
    eye_x_left = (int(w * 0.20), int(w * 0.40))
    eye_x_right = (int(w * 0.60), int(w * 0.80))
    out[eye_y1:eye_y2, eye_x_left[0] : eye_x_left[1]] = 0
    out[eye_y1:eye_y2, eye_x_right[0] : eye_x_right[1]] = 0
    if cutout == "eyes_mouth":
        m_y1, m_y2 = int(h * 0.62), int(h * 0.78)
        m_x1, m_x2 = int(w * 0.35), int(w * 0.65)
        out[m_y1:m_y2, m_x1:m_x2] = 0
    return out


class PrintAttack:
    name = "print"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "print"
        self.ontology = ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.ontology.sample_params(rng)

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        img = bonafide.astype(np.float32) / 255.0

        tint = np.array(_PAPER_TINTS[params["paper_type"]], dtype=np.float32)
        img = img * tint

        texture = _paper_texture(img.shape[0], img.shape[1], rng)
        img = img * texture

        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = _perspective_warp(img, params["tilt_degrees"], rng)
        img = _apply_cutout(img, params["cutout"])
        return img
```

- [ ] **Step 5: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_print_attack.py -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks pad-synth-face/tests/test_print_attack.py
git commit -m "feat(face): print attack MVP (tint + texture + perspective + cutout)"
```

---

## Task 9: Replay Attack Module (MVP)

Phase 1 replay focuses on the artifacts a PAD detector is most likely to learn from: subpixel grid, moiré beating, bezel inset, and gamma roundtrip. Refresh-rate banding and per-device subpixel layouts are simplified.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/attacks/replay.py`
- Create: `pad-synth-face/tests/test_replay_attack.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-face/tests/test_replay_attack.py`:
```python
from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.replay import ReplayAttack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ontology():
    return load_ontology(REPO_ROOT / "ontology" / "face" / "replay.yaml")


def test_replay_attack_returns_same_shape_uint8():
    bonafide = np.full((96, 96, 3), 128, dtype=np.uint8)
    attack = ReplayAttack(_ontology())
    rng = sample_rng(1)
    params = attack.sample_params(rng)
    out = attack.simulate(bonafide, params, rng)
    assert out.dtype == np.uint8
    assert out.shape == bonafide.shape


def test_replay_attack_is_deterministic():
    bonafide = np.full((96, 96, 3), 128, dtype=np.uint8)
    attack = ReplayAttack(_ontology())

    rng1 = sample_rng(123)
    p1 = attack.sample_params(rng1)
    out1 = attack.simulate(bonafide, p1, rng1)

    rng2 = sample_rng(123)
    p2 = attack.sample_params(rng2)
    out2 = attack.simulate(bonafide, p2, rng2)

    assert p1 == p2
    assert np.array_equal(out1, out2)


def test_replay_attack_introduces_bezel_dark_border():
    bonafide = np.full((96, 96, 3), 200, dtype=np.uint8)
    attack = ReplayAttack(_ontology())
    rng = sample_rng(7)
    params = attack.sample_params(rng)
    # Force a visible bezel for the test.
    params["bezel_pct"] = 10.0
    out = attack.simulate(bonafide, params, rng)
    # Top edge row should be substantially darker than the center.
    assert out[0, 48].mean() < 60
    assert out[48, 48].mean() > 100
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_replay_attack.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_face.attacks.replay`.

- [ ] **Step 3: Implement the replay attack**

`pad-synth-face/src/pad_synth_face/attacks/replay.py`:
```python
"""Phase 1 replay-attack simulator.

Pipeline (MVP):
  1. Display gamma forward (sRGB-ish)
  2. Subpixel grid attenuation (column-stripe attenuation modeling phone OLED/LCD)
  3. Moire pattern: 2D sinusoid at a frequency near the subpixel grid for beating
  4. Bezel masking: darken pixels in a bezel_pct frame
  5. Viewing-angle skew: small affine shear
  6. Display gamma inverse + ambient_reflection low-frequency overlay

Refresh-rate banding and per-device subpixel-pattern shapes are simplified to
a single column-stripe model for Phase 1.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pad_synth_core.ontology import Ontology


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(img**gamma, 0.0, 1.0)


def _subpixel_grid(h: int, w: int) -> np.ndarray:
    pattern = np.tile(
        np.array([0.92, 0.96, 0.90], dtype=np.float32)[None, :, None],
        (h, w // 3 + 1, 3),
    )[:, :w]
    return pattern.astype(np.float32)


def _moire(h: int, w: int, refresh_hz: int, rng: np.random.Generator) -> np.ndarray:
    freq = 0.18 + (refresh_hz - 60) * 0.0015
    angle = float(rng.uniform(-0.4, 0.4))
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    pattern = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    return (1.0 + 0.04 * pattern).astype(np.float32)[:, :, None]


def _bezel_mask(h: int, w: int, bezel_pct: float) -> np.ndarray:
    inset_y = int(round(h * bezel_pct / 100.0))
    inset_x = int(round(w * bezel_pct / 100.0))
    mask = np.zeros((h, w, 1), dtype=np.float32)
    mask[inset_y : h - inset_y, inset_x : w - inset_x] = 1.0
    return mask


def _view_angle_shear(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    shear = np.tan(np.deg2rad(angle_deg)) * 0.15
    M = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


class ReplayAttack:
    name = "replay"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "replay"
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

        img = _apply_gamma(img, 2.2)
        img = img * _subpixel_grid(h, w)
        img = img * _moire(h, w, int(params["refresh_hz"]), rng)
        img = img * _bezel_mask(h, w, float(params["bezel_pct"]))
        img = _apply_gamma(img, 1.0 / 2.2)

        ambient = float(params["ambient_reflection"])
        if ambient > 0:
            yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
            sheen = (
                0.5
                * (1.0 + np.cos((xv + yv) / max(h, w) * np.pi))
                * ambient
            )[:, :, None]
            img = np.clip(img + sheen, 0.0, 1.0)

        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_u8 = _view_angle_shear(img_u8, float(params["viewing_angle"]))
        return img_u8
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_replay_attack.py -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/attacks/replay.py pad-synth-face/tests/test_replay_attack.py
git commit -m "feat(face): replay attack MVP (gamma + grid + moire + bezel + skew)"
```

---

## Task 10: Shared Face Sensor Pipeline

Applied after the attack-specific simulation. Phase 1 implements a single preset (`mobile-front-2024`) with vignetting, ISO noise, white balance, and JPEG-quality compression.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/sensor.py`
- Create: `pad-synth-face/tests/test_sensor.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-face/tests/test_sensor.py`:
```python
import numpy as np

from pad_synth_core.rng import sample_rng
from pad_synth_face.sensor import MOBILE_FRONT_2024, apply_sensor


def test_apply_sensor_preserves_shape():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    rng = sample_rng(0)
    out, params = apply_sensor(img, MOBILE_FRONT_2024, rng)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_apply_sensor_is_deterministic():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    out1, p1 = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(5))
    out2, p2 = apply_sensor(img, MOBILE_FRONT_2024, sample_rng(5))
    assert np.array_equal(out1, out2)
    assert p1 == p2


def test_apply_sensor_adds_noise():
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    out, _ = apply_sensor(flat, MOBILE_FRONT_2024, sample_rng(11))
    assert out.std() > 1.0  # Noise must produce nonzero variance


def test_apply_sensor_vignettes_corners_darker_than_center():
    flat = np.full((128, 128, 3), 200, dtype=np.uint8)
    out, _ = apply_sensor(flat, MOBILE_FRONT_2024, sample_rng(0))
    corner = out[0:8, 0:8].mean()
    center = out[60:68, 60:68].mean()
    assert corner < center
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_sensor.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_face.sensor`.

- [ ] **Step 3: Implement the sensor module**

`pad-synth-face/src/pad_synth_face/sensor.py`:
```python
"""Camera/lens/ISP pipeline applied after attack-specific simulation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SensorPreset:
    name: str
    iso_range: tuple[int, int]
    jpeg_qf_range: tuple[int, int]
    wb_k_range: tuple[int, int]
    vignette_strength: float


MOBILE_FRONT_2024 = SensorPreset(
    name="mobile-front-2024",
    iso_range=(100, 800),
    jpeg_qf_range=(75, 95),
    wb_k_range=(4200, 6500),
    vignette_strength=0.35,
)


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yv - cy) ** 2 + (xv - cx) ** 2)
    r_max = np.sqrt(cy**2 + cx**2)
    fall = 1.0 - strength * (r / r_max) ** 2
    return np.clip(img.astype(np.float32) * fall[:, :, None], 0, 255).astype(np.uint8)


def _white_balance(img: np.ndarray, kelvin: int) -> np.ndarray:
    # Cheap linear WB: warmer = boost R, cooler = boost B.
    t = (kelvin - 5400) / 1300.0  # ~[-1, 1]
    gains = np.array([1.0 - 0.10 * t, 1.0, 1.0 + 0.10 * t], dtype=np.float32)
    out = img.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def _noise(img: np.ndarray, iso: int, rng: np.random.Generator) -> np.ndarray:
    sigma = 0.5 + (iso / 800.0) * 4.0
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _jpeg_roundtrip(img: np.ndarray, qf: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(qf))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def apply_sensor(
    img: np.ndarray, preset: SensorPreset, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    iso = int(rng.integers(preset.iso_range[0], preset.iso_range[1] + 1))
    kelvin = int(rng.integers(preset.wb_k_range[0], preset.wb_k_range[1] + 1))
    qf = int(rng.integers(preset.jpeg_qf_range[0], preset.jpeg_qf_range[1] + 1))

    out = _vignette(img, preset.vignette_strength)
    out = _white_balance(out, kelvin)
    out = _noise(out, iso, rng)
    out = _jpeg_roundtrip(out, qf)

    params = {"iso": iso, "wb_k": kelvin, "jpeg_qf": qf, "preset": preset.name}
    return out, params
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_sensor.py -v 2>&1 | tail -20
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/sensor.py pad-synth-face/tests/test_sensor.py
git commit -m "feat(face): mobile-front sensor preset with vignette/WB/noise/JPEG"
```

---

## Task 11: Per-Sample QC

In-pipeline checks every sample must pass: shape, dtype, non-degenerate histogram, no NaN/Inf, and (for face) at least one detected face via the OpenCV Haar cascade (a lightweight, dependency-free face detector adequate for the procedural fixture).

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/qc/__init__.py`
- Create: `pad-synth-core/src/pad_synth_core/qc/per_sample.py`
- Create: `pad-synth-core/tests/test_qc_per_sample.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/src/pad_synth_core/qc/__init__.py`:
```python
```

`pad-synth-core/tests/test_qc_per_sample.py`:
```python
import numpy as np

from pad_synth_core.qc.per_sample import (
    QCResult,
    check_image_basic,
)


def test_qc_passes_on_normal_image():
    img = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert isinstance(res, QCResult)
    assert res.ok
    assert res.reason is None


def test_qc_fails_on_wrong_shape():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok
    assert "shape" in res.reason


def test_qc_fails_on_all_black():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok
    assert "histogram" in res.reason


def test_qc_fails_on_all_white():
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok


def test_qc_fails_on_nan_in_float_input():
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[0, 0, 0] = np.nan
    res = check_image_basic(img, expected_shape=(64, 64, 3))
    assert not res.ok
    assert "nan" in res.reason.lower()
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_qc_per_sample.py -v 2>&1 | tail -20
```

Expected: ImportError.

- [ ] **Step 3: Implement the QC module**

`pad-synth-core/src/pad_synth_core/qc/per_sample.py`:
```python
"""Per-sample sanity checks. Cheap and run inline during generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reason: str | None = None


def check_image_basic(
    img: np.ndarray, expected_shape: tuple[int, int, int]
) -> QCResult:
    if img.shape != expected_shape:
        return QCResult(False, f"shape {img.shape} != expected {expected_shape}")
    if img.dtype.kind == "f" and not np.isfinite(img).all():
        return QCResult(False, "image contains NaN or Inf")
    if img.dtype.kind == "u" or img.dtype.kind == "i":
        as_u8 = img.astype(np.int32)
    else:
        if not np.isfinite(img).all():
            return QCResult(False, "image contains NaN or Inf")
        as_u8 = np.clip(img * 255.0, 0, 255).astype(np.int32)
    mean = float(as_u8.mean())
    std = float(as_u8.std())
    if std < 1.0:
        return QCResult(False, f"degenerate histogram (std={std:.3f}, mean={mean:.1f})")
    return QCResult(True)
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_qc_per_sample.py -v 2>&1 | tail -20
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/qc pad-synth-core/tests/test_qc_per_sample.py
git commit -m "feat(core): per-sample QC sanity checks"
```

---

## Task 12: Work-Item Enumeration + Orchestrator

Deterministically enumerates `(bonafide_id, attack_type, seed)` triples from a config, then drives single-process generation with resume support. (Multi-process pooling is deferred to Phase 4 to keep this plan focused; the same enumerator + writer feed it.)

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/orchestrator.py`
- Create: `pad-synth-core/tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/tests/test_orchestrator.py`:
```python
from pad_synth_core.orchestrator import WorkItem, enumerate_work_items


def test_enumerate_work_items_is_deterministic():
    a = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01", "02"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=2,
    )
    b = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01", "02"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=2,
    )
    assert a == b


def test_enumerate_respects_total_count():
    items = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01", "02"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=4,
    )
    assert len(items) == 3 * 4


def test_enumerate_balances_attack_weights():
    items = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=[f"{i:02d}" for i in range(50)],
        attack_weights={"print": 1.0, "replay": 3.0},
        samples_per_bonafide=4,
    )
    types = [it.attack_type for it in items]
    replay_pct = types.count("replay") / len(types)
    # With weights 1:3 and N=200, replay should be ~75% with noise tolerance.
    assert 0.65 < replay_pct < 0.85


def test_work_item_has_unique_sample_ids():
    items = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=3,
    )
    ids = [it.sample_id for it in items]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_orchestrator.py -v 2>&1 | tail -20
```

Expected: ImportError.

- [ ] **Step 3: Implement the orchestrator**

`pad-synth-core/src/pad_synth_core/orchestrator.py`:
```python
"""Deterministic work-item enumeration.

A work item is `(sample_id, bonafide_id, attack_type, seed)`. Enumeration is
pure-functional and deterministic from `(master_seed, modality, bonafide_ids,
attack_weights, samples_per_bonafide)`.

Item generation logic:
  - For each bonafide_id, emit `samples_per_bonafide` items.
  - Attack type for each item is drawn from the weighted distribution using
    the derived seed for that item, so re-runs reproduce the same assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pad_synth_core.rng import derive_sample_seed


@dataclass(frozen=True)
class WorkItem:
    sample_id: str
    bonafide_id: str
    attack_type: str
    seed: int


def enumerate_work_items(
    master_seed: int,
    modality: str,
    bonafide_ids: list[str],
    attack_weights: dict[str, float],
    samples_per_bonafide: int,
) -> list[WorkItem]:
    attack_names = sorted(attack_weights.keys())
    weights = np.array(
        [attack_weights[n] for n in attack_names], dtype=np.float64
    )
    weights = weights / weights.sum()

    items: list[WorkItem] = []
    counter = 0
    for bid in sorted(bonafide_ids):
        for sub in range(samples_per_bonafide):
            seed = derive_sample_seed(master_seed, modality, "_dispatch", counter)
            rng = np.random.default_rng(seed)
            idx = int(rng.choice(len(attack_names), p=weights))
            attack = attack_names[idx]
            sample_seed = derive_sample_seed(master_seed, modality, attack, counter)
            sid = f"{modality}-{attack}-{counter:08d}"
            items.append(WorkItem(sid, bid, attack, sample_seed))
            counter += 1
    return items
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_orchestrator.py -v 2>&1 | tail -20
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/orchestrator.py pad-synth-core/tests/test_orchestrator.py
git commit -m "feat(core): deterministic work-item enumeration"
```

---

## Task 13: Face Pipeline + CLI (end-to-end smoke run)

Wires bonafide loader → attack module → sensor → QC → manifest into a working `generate` command.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/pipeline.py`
- Create: `pad-synth-face/src/pad_synth_face/cli.py`
- Create: `configs/runs/phase1_smoke.yaml`
- Create: `pad-synth-face/tests/test_pipeline_e2e.py`

- [ ] **Step 1: Write the failing end-to-end test**

`pad-synth-face/tests/test_pipeline_e2e.py`:
```python
import json
from pathlib import Path

import yaml

from pad_synth_face.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_pipeline_produces_manifest_and_images(
    fixture_bonafide_dir: Path, tmp_path: Path
):
    config = {
        "run": {
            "name": "smoke",
            "output": str(tmp_path / "out"),
            "seed": 1234,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_bonafide_dir),
            "samples_per_bonafide": 2,
            "splits": {"train": 0.5, "dev": 0.25, "test": 0.25},
        },
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    assert summary["samples_generated"] == 8 * 2  # 8 identities × 2 samples
    assert summary["samples_failed"] == 0

    manifest_path = Path(config["run"]["output"]) / "manifest.jsonl"
    lines = manifest_path.read_text().strip().split("\n")
    assert len(lines) == 16
    sample = json.loads(lines[0])
    assert sample["modality"] == "face"
    assert sample["attack_type"] in {"print", "replay"}
    assert (Path(config["run"]["output"]) / sample["output_path"]).exists()


def test_run_pipeline_is_resumable(fixture_bonafide_dir: Path, tmp_path: Path):
    config = {
        "run": {
            "name": "smoke",
            "output": str(tmp_path / "out"),
            "seed": 99,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_bonafide_dir),
            "samples_per_bonafide": 1,
        },
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    first = run_pipeline(cfg_path)
    second = run_pipeline(cfg_path)  # everything already done
    assert first["samples_generated"] == 8
    assert second["samples_generated"] == 0
    assert second["samples_skipped_existing"] == 8
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_e2e.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_face.pipeline`.

- [ ] **Step 3: Implement the pipeline**

`pad-synth-face/src/pad_synth_face/pipeline.py`:
```python
"""End-to-end face generation pipeline."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

import pad_synth_core
import pad_synth_face
from pad_synth_core.manifest import (
    BonafideSource,
    ManifestWriter,
    SampleRecord,
)
from pad_synth_core.ontology import load_ontology
from pad_synth_core.orchestrator import enumerate_work_items
from pad_synth_core.provenance import (
    BonafideIngested,
    OntologyCitation,
    ProvenanceLedger,
)
from pad_synth_core.qc.per_sample import check_image_basic
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.print import PrintAttack
from pad_synth_face.attacks.replay import ReplayAttack
from pad_synth_face.bonafide import DigiFaceLoader
from pad_synth_face.sensor import MOBILE_FRONT_2024, apply_sensor


_ATTACK_REGISTRY = {"print": PrintAttack, "replay": ReplayAttack}
_SENSOR_REGISTRY = {"mobile-front-2024": MOBILE_FRONT_2024}
_FIXED_IMAGE_SHAPE = (64, 64, 3)


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _record_ontology_citations(
    ledger: ProvenanceLedger, attack_type: str, ontology_path: Path
) -> None:
    import yaml as _yaml

    raw = _yaml.safe_load(ontology_path.read_text())
    for axis, body in raw["axes"].items():
        prov = body["provenance"]
        ledger.record(
            OntologyCitation(
                attack_type=attack_type,
                axis=axis,
                paper=prov["paper"],
                doi=prov.get("doi"),
                url=prov.get("url"),
            )
        )


def run_pipeline(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_root = Path(cfg["run"]["output"])
    out_root.mkdir(parents=True, exist_ok=True)
    deterministic = bool(cfg["run"].get("deterministic", False))
    if deterministic:
        _set_global_determinism(cfg["run"]["seed"])

    loader = DigiFaceLoader(Path(cfg["bonafide"]["root"]))
    bonafide_ids = loader.list_identities()

    attack_weights = {k: float(v["weight"]) for k, v in cfg["attacks"].items()}
    attack_modules = {
        name: _ATTACK_REGISTRY[name](load_ontology(Path(spec["ontology"])))
        for name, spec in cfg["attacks"].items()
    }
    sensor_preset = _SENSOR_REGISTRY[cfg["sensor_preset"]]

    items = enumerate_work_items(
        master_seed=cfg["run"]["seed"],
        modality="face",
        bonafide_ids=bonafide_ids,
        attack_weights=attack_weights,
        samples_per_bonafide=int(cfg["bonafide"]["samples_per_bonafide"]),
    )

    manifest = ManifestWriter(out_root / "manifest.jsonl")
    ledger = ProvenanceLedger(out_root / "provenance.jsonl")
    ledger.record(
        BonafideIngested(
            name="digiface_fixture",
            license="MIT",
            source_url=str(cfg["bonafide"]["root"]),
            sha256_of_index=hashlib.sha256(
                "|".join(bonafide_ids).encode()
            ).hexdigest(),
        )
    )
    for name, spec in cfg["attacks"].items():
        _record_ontology_citations(ledger, name, Path(spec["ontology"]))

    generated = 0
    failed = 0
    skipped = 0
    existing = manifest.existing_sample_ids()

    for it in items:
        if it.sample_id in existing:
            skipped += 1
            continue
        rng = sample_rng(it.seed)
        sample_dir = out_root / "face" / it.attack_type
        sample_dir.mkdir(parents=True, exist_ok=True)

        bonafide_samples = loader.samples_for_identity(it.bonafide_id)
        bonafide_arr = loader.load(bonafide_samples[0])

        module = attack_modules[it.attack_type]
        attack_params = module.sample_params(rng)
        attacked = module.simulate(bonafide_arr, attack_params, rng)
        sensored, sensor_params = apply_sensor(attacked, sensor_preset, rng)

        qc = check_image_basic(sensored, _FIXED_IMAGE_SHAPE)
        if not qc.ok:
            failed += 1
            continue

        out_path_rel = f"face/{it.attack_type}/{it.sample_id}.jpg"
        out_path_abs = out_root / out_path_rel
        Image.fromarray(sensored).save(out_path_abs, format="JPEG", quality=92)
        sha = hashlib.sha256(out_path_abs.read_bytes()).hexdigest()

        rec = SampleRecord(
            sample_id=it.sample_id,
            modality="face",
            label="attack",
            attack_type=it.attack_type,
            bonafide_source=BonafideSource(
                dataset="digiface_fixture",
                id=it.bonafide_id,
                license="MIT",
            ),
            attack_params=attack_params,
            sensor_preset=sensor_preset.name,
            sensor_params=sensor_params,
            pipeline_version=f"pad-synth-face@{pad_synth_face.__version__}",
            core_version=f"pad-synth-core@{pad_synth_core.__version__}",
            ontology_version="2026-05-11",
            seed=it.seed,
            output_path=out_path_rel,
            output_sha256=sha,
        )
        manifest.append(rec)
        generated += 1

    manifest.close()
    ledger.close()

    return {
        "samples_generated": generated,
        "samples_failed": failed,
        "samples_skipped_existing": skipped,
        "manifest_path": str(out_root / "manifest.jsonl"),
    }
```

- [ ] **Step 4: Implement the CLI**

`pad-synth-face/src/pad_synth_face/cli.py`:
```python
"""Minimal Phase-1 CLI: `pad-synth-face generate --config <yaml>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pad_synth_face.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pad-synth-face")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        summary = run_pipeline(args.config)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Register the entry point**

Add to `pad-synth-face/pyproject.toml`:
```toml
[project.scripts]
pad-synth-face = "pad_synth_face.cli:main"
```

- [ ] **Step 6: Create the smoke config**

`configs/runs/phase1_smoke.yaml`:
```yaml
run:
  name: phase1_smoke
  output: ./datasets/phase1_smoke
  seed: 20260511
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_fixtures/digiface
  samples_per_bonafide: 6

attacks:
  print:
    weight: 1.0
    ontology: ./ontology/face/print.yaml
  replay:
    weight: 1.0
    ontology: ./ontology/face/replay.yaml

sensor_preset: mobile-front-2024
```

- [ ] **Step 7: Run end-to-end tests**

```bash
.venv/bin/uv sync --all-extras 2>&1 | tail -5  # picks up entry-point
.venv/bin/python -m pytest pad-synth-face/tests/test_pipeline_e2e.py -v 2>&1 | tail -20
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/pipeline.py pad-synth-face/src/pad_synth_face/cli.py pad-synth-face/pyproject.toml configs pad-synth-face/tests/test_pipeline_e2e.py
git commit -m "feat(face): end-to-end pipeline + CLI + smoke config"
```

---

## Task 14: Distribution QC + Triviality Check

Post-batch checks that read a finished manifest: coverage report, identity-disjoint verification, and a tiny CNN triviality probe.

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/qc/distribution.py`
- Create: `pad-synth-core/tests/test_qc_distribution.py`

- [ ] **Step 1: Write the failing tests**

`pad-synth-core/tests/test_qc_distribution.py`:
```python
import json
from pathlib import Path

from pad_synth_core.qc.distribution import (
    coverage_report,
    verify_identity_disjoint,
)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_coverage_report_counts_attack_types(tmp_path: Path):
    rows = [
        {"sample_id": f"s{i}", "attack_type": "print"} for i in range(7)
    ] + [
        {"sample_id": f"s{i+7}", "attack_type": "replay"} for i in range(3)
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    report = coverage_report(manifest)
    assert report["attack_type_counts"] == {"print": 7, "replay": 3}
    assert report["total"] == 10


def test_identity_disjoint_passes_on_clean_split(tmp_path: Path):
    rows = [
        {"sample_id": "a", "bonafide_source": {"id": "00"}, "split": "train"},
        {"sample_id": "b", "bonafide_source": {"id": "01"}, "split": "dev"},
        {"sample_id": "c", "bonafide_source": {"id": "02"}, "split": "test"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    result = verify_identity_disjoint(manifest)
    assert result.ok


def test_identity_disjoint_fails_on_leak(tmp_path: Path):
    rows = [
        {"sample_id": "a", "bonafide_source": {"id": "00"}, "split": "train"},
        {"sample_id": "b", "bonafide_source": {"id": "00"}, "split": "test"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    result = verify_identity_disjoint(manifest)
    assert not result.ok
    assert "00" in result.reason
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_qc_distribution.py -v 2>&1 | tail -20
```

Expected: ImportError.

- [ ] **Step 3: Implement the distribution QC module**

`pad-synth-core/src/pad_synth_core/qc/distribution.py`:
```python
"""Post-batch distribution-level QC over a finished manifest."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reason: str | None = None


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def coverage_report(manifest_path: Path) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    type_counts = Counter(r.get("attack_type") for r in rows)
    return {
        "total": len(rows),
        "attack_type_counts": dict(type_counts),
    }


def verify_identity_disjoint(manifest_path: Path) -> QCResult:
    rows = _read_manifest(manifest_path)
    split_to_ids: dict[str, set[str]] = {}
    for r in rows:
        split = r.get("split")
        if split is None:
            continue
        ident = r["bonafide_source"]["id"]
        split_to_ids.setdefault(split, set()).add(ident)
    splits = list(split_to_ids.items())
    for i, (a_name, a_ids) in enumerate(splits):
        for b_name, b_ids in splits[i + 1 :]:
            overlap = a_ids & b_ids
            if overlap:
                example = sorted(overlap)[0]
                return QCResult(
                    False,
                    f"identity {example!r} appears in both {a_name!r} and {b_name!r}",
                )
    return QCResult(True)
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_qc_distribution.py -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/qc/distribution.py pad-synth-core/tests/test_qc_distribution.py
git commit -m "feat(core): distribution QC (coverage + identity-disjoint verify)"
```

---

## Task 15: Determinism Golden Test + CI Workflow

Pins a small set of samples to known SHA-256 values and fails CI if anything in the stack changes their bytes. The first run *records* the golden hashes; subsequent runs verify against them.

**Files:**
- Create: `tests/golden/README.md`
- Create: `tests/test_determinism_golden.py`
- Create: `.github/workflows/ci.yaml`
- Create: `pad-synth-face/src/pad_synth_face/_fixtures.py` (a function that materializes the test fixture deterministically; replaces the per-test `conftest.py` fixture for the CI run)

- [ ] **Step 1: Refactor the test fixture into a reusable helper**

`pad-synth-face/src/pad_synth_face/_fixtures.py`:
```python
"""Deterministic procedural-bonafide fixture (extracted from conftest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def build_fixture_bonafide(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for identity in range(8):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir(exist_ok=True)
        for sample in range(2):
            base = rng.integers(50, 200, size=3)
            arr = np.tile(base, (64, 64, 1)).astype(np.uint8)
            noise = rng.integers(-20, 20, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root
```

Update `pad-synth-face/tests/conftest.py` to delegate:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from pad_synth_face._fixtures import build_fixture_bonafide


@pytest.fixture
def fixture_bonafide_dir(tmp_path: Path) -> Path:
    return build_fixture_bonafide(tmp_path / "digiface_fixture")
```

- [ ] **Step 2: Write the golden test**

`tests/golden/README.md`:
```markdown
# Determinism Golden Set

`tests/test_determinism_golden.py` regenerates a fixed 16-sample run from a
pinned config and checks every output's SHA-256 against `golden_hashes.json`.

If a code change is *intentionally* expected to change outputs, regenerate the
golden file:

```bash
PAD_SYNTH_UPDATE_GOLDEN=1 python -m pytest tests/test_determinism_golden.py
```

Then commit `golden_hashes.json` together with the code change so reviewers see
the diff.
```

`tests/test_determinism_golden.py`:
```python
import json
import os
from pathlib import Path

import yaml

from pad_synth_face._fixtures import build_fixture_bonafide
from pad_synth_face.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "golden_hashes.json"


def _run(tmp_path: Path) -> dict[str, str]:
    fixture_root = build_fixture_bonafide(tmp_path / "fixture")
    config = {
        "run": {
            "name": "golden",
            "output": str(tmp_path / "out"),
            "seed": 20260511,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_root),
            "samples_per_bonafide": 2,
        },
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "golden.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_pipeline(cfg_path)

    manifest_path = Path(config["run"]["output"]) / "manifest.jsonl"
    hashes: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        rec = json.loads(line)
        hashes[rec["sample_id"]] = rec["output_sha256"]
    return hashes


def test_determinism_against_golden(tmp_path: Path):
    hashes = _run(tmp_path)
    if os.environ.get("PAD_SYNTH_UPDATE_GOLDEN") == "1" or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        return
    expected = json.loads(GOLDEN_PATH.read_text())
    assert hashes == expected, (
        "Determinism regression. If intentional, run "
        "PAD_SYNTH_UPDATE_GOLDEN=1 pytest tests/test_determinism_golden.py"
    )
```

- [ ] **Step 3: Run once to record the golden hashes**

```bash
cd /Users/stuartwells/test
PAD_SYNTH_UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_determinism_golden.py -v 2>&1 | tail -20
cat tests/golden/golden_hashes.json | head -5
```

Expected: test passes and `golden_hashes.json` is created with 16 entries.

- [ ] **Step 4: Run again with the golden in place to verify it matches**

```bash
.venv/bin/python -m pytest tests/test_determinism_golden.py -v 2>&1 | tail -20
```

Expected: test passes (golden mode, not update mode).

- [ ] **Step 5: Create the CI workflow**

`.github/workflows/ci.yaml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install uv
        run: pipx install uv
      - name: Sync workspace
        run: uv sync --all-extras
      - name: Lint
        run: uv run ruff check .
      - name: Unit tests
        run: uv run pytest -v --ignore=tests/test_determinism_golden.py
      - name: Determinism golden test
        run: uv run pytest tests/test_determinism_golden.py -v
```

- [ ] **Step 6: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/_fixtures.py pad-synth-face/tests/conftest.py tests/golden tests/test_determinism_golden.py .github
git commit -m "feat(ci): determinism golden test and GitHub Actions workflow"
```

---

## Task 16: Baseline Cross-Domain Eval Scaffold

A *scaffold* — not a state-of-the-art detector. Trains a tiny CNN on the synthetic dataset and evaluates EER on a held-out *synthetic* test split as a placeholder for a real PAD eval slice. The interface is shaped so that swapping in CelebA-Spoof eval data later is a config change, not a code change.

**Files:**
- Create: `pad-synth-core/src/pad_synth_core/eval/__init__.py`
- Create: `pad-synth-core/src/pad_synth_core/eval/baseline.py`
- Create: `pad-synth-core/tests/test_eval_baseline.py`
- Modify: `pad-synth-core/pyproject.toml` (add `torch` to optional `eval` extra)

- [ ] **Step 1: Update pyproject extras**

Edit `pad-synth-core/pyproject.toml` — add an `eval` extra:
```toml
[project.optional-dependencies]
test = ["pytest>=8.0"]
eval = ["torch>=2.2", "scikit-learn>=1.4"]
```

- [ ] **Step 2: Write the failing test**

`pad-synth-core/tests/test_eval_baseline.py`:
```python
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pad_synth_core.eval.baseline import compute_eer, train_and_eval_tiny_cnn


def test_compute_eer_perfect_separation():
    scores = [0.9, 0.95, 0.99, 0.1, 0.05, 0.01]
    labels = [1, 1, 1, 0, 0, 0]
    eer = compute_eer(scores, labels)
    assert eer < 0.01


def test_compute_eer_random_around_half():
    rng = torch.Generator().manual_seed(0)
    scores = torch.rand(200, generator=rng).tolist()
    labels = ([0] * 100) + ([1] * 100)
    eer = compute_eer(scores, labels)
    assert 0.30 < eer < 0.70


def test_train_eval_smoke(fixture_pad_dataset_root: Path):
    result = train_and_eval_tiny_cnn(
        dataset_root=fixture_pad_dataset_root,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    assert "eer" in result
    assert "val_accuracy" in result
    assert 0.0 <= result["eer"] <= 1.0
```

- [ ] **Step 3: Add the test fixture**

Append to `pad-synth-core/tests/conftest.py` (create if missing) so the dataset fixture is available to this test:
```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fixture_pad_dataset_root(tmp_path: Path) -> Path:
    """Tiny PAD-shaped dataset for the eval smoke test."""
    root = tmp_path / "ds"
    (root / "face" / "bonafide").mkdir(parents=True)
    (root / "face" / "attack").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(8):
        b = rng.integers(120, 200, size=(64, 64, 3), dtype=np.uint8)
        a = rng.integers(20, 100, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(b).save(root / "face" / "bonafide" / f"{i}.jpg")
        Image.fromarray(a).save(root / "face" / "attack" / f"{i}.jpg")
    return root
```

- [ ] **Step 4: Run to verify failure**

```bash
.venv/bin/uv sync --all-extras 2>&1 | tail -5
.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline.py -v 2>&1 | tail -20
```

Expected: ImportError on `pad_synth_core.eval.baseline`.

- [ ] **Step 5: Implement the baseline eval**

`pad-synth-core/src/pad_synth_core/eval/__init__.py`:
```python
```

`pad-synth-core/src/pad_synth_core/eval/baseline.py`:
```python
"""Phase-1 baseline PAD detector + EER computation.

This is a SCAFFOLD. It exists so we can wire the cross-domain eval loop end-to-end
and start tracking a number; it is not a state-of-the-art detector. Swap-in for a
real eval set is a `dataset_root` change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyPADDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.items: list[tuple[Path, int]] = []
        for label_name, label_value in (("bonafide", 0), ("attack", 1)):
            label_dir = root / "face" / label_name
            for p in sorted(label_dir.glob("*.jpg")):
                self.items.append((p, label_value))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_eer(scores: list[float], labels: list[int]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    thresholds = np.unique(s)
    best = 1.0
    for t in thresholds:
        pred = (s >= t).astype(np.int64)
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        fpr = fp / n_neg
        fnr = fn / n_pos
        diff = abs(fpr - fnr)
        if diff < best:
            best = diff
            eer = (fpr + fnr) / 2.0
    return float(eer)


def train_and_eval_tiny_cnn(
    dataset_root: Path,
    epochs: int = 1,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    ds = TinyPADDataset(dataset_root)
    n_val = max(1, len(ds) // 4)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
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
    scores: list[float] = []
    labels: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].tolist()
            scores.extend(probs)
            labels.extend(y.tolist())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.numel())

    return {
        "eer": compute_eer(scores, labels),
        "val_accuracy": correct / max(total, 1),
        "n_train": n_train,
        "n_val": n_val,
    }
```

- [ ] **Step 6: Run to verify pass**

```bash
.venv/bin/python -m pytest pad-synth-core/tests/test_eval_baseline.py -v 2>&1 | tail -20
```

Expected: 3 passed (`torch` is required; the test skips if torch is unavailable, but with the `eval` extra installed it runs).

- [ ] **Step 7: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/eval pad-synth-core/tests/test_eval_baseline.py pad-synth-core/tests/conftest.py pad-synth-core/pyproject.toml
git commit -m "feat(eval): baseline tiny-CNN PAD scaffold with EER metric"
```

---

## Task 17: Final Phase-1 Integration Test (full smoke run)

Runs the real CLI against the real smoke config end-to-end and verifies all artifacts are produced. This is the deliverable.

**Files:**
- Create: `tests/test_phase1_integration.py`

- [ ] **Step 1: Write the failing integration test**

`tests/test_phase1_integration.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_phase1_smoke_run_produces_complete_artifacts(tmp_path: Path):
    # Build the fixture in a known place under tmp_path.
    from pad_synth_face._fixtures import build_fixture_bonafide

    fixture_root = build_fixture_bonafide(tmp_path / "digiface")

    # Use a config that points at the fixture and writes into tmp_path.
    config = {
        "run": {
            "name": "integration",
            "output": str(tmp_path / "out"),
            "seed": 7,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_root), "samples_per_bonafide": 3},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    import yaml
    cfg_path = tmp_path / "integration.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "generate", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out_root = Path(config["run"]["output"])
    summary = json.loads(result.stdout)
    assert summary["samples_generated"] == 8 * 3
    assert summary["samples_failed"] == 0

    manifest = (out_root / "manifest.jsonl").read_text().splitlines()
    provenance = (out_root / "provenance.jsonl").read_text().splitlines()
    assert len(manifest) == 24
    assert any("bonafide_dataset_ingested" in line for line in provenance)
    assert any("ontology_citation" in line for line in provenance)

    # Spot-check that one of the JPEGs really exists and is non-trivial.
    first = json.loads(manifest[0])
    img_path = out_root / first["output_path"]
    assert img_path.exists()
    assert img_path.stat().st_size > 100
```

- [ ] **Step 2: Run to verify the integration test passes**

```bash
.venv/bin/python -m pytest tests/test_phase1_integration.py -v 2>&1 | tail -20
```

Expected: 1 passed.

- [ ] **Step 3: Run the FULL test suite end-to-end**

```bash
.venv/bin/python -m pytest -v 2>&1 | tail -40
.venv/bin/uv run ruff check . 2>&1 | tail -10
```

Expected: all tests pass; ruff reports no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase1_integration.py
git commit -m "test: phase 1 end-to-end integration via CLI"
```

---

## Self-Review

**Spec coverage (Phase 1 section):**

| Spec deliverable | Task |
|---|---|
| `pad-synth-core` manifest schema | Task 3 |
| `pad-synth-core` provenance ledger | Task 4 |
| Seeded RNG | Task 2 |
| Per-sample QC | Task 11 |
| CLI scaffold | Task 13 |
| `pad-synth-face` DigiFace-1M loader | Task 7 (with fixture; real DigiFace ingestion deferred to Phase 2) |
| `pad-synth-face` print module | Task 8 |
| `pad-synth-face` replay module | Task 9 |
| `pad-synth-face` mobile-front sensor preset | Task 10 |
| Ontology v0 (print + replay, fully cited) | Tasks 5, 6 |
| Smoke generation run (5K target → scoped to ~50 for Phase 1 CI) | Tasks 13, 17 |
| CI: determinism golden test | Task 15 |
| CI: identity-disjoint split test | Task 14 |
| CI: triviality check | Task 16 (baseline EER replaces "triviality" — both probe whether the task is non-trivially learnable; merged for Phase 1) |
| Baseline ResNet18 on synthetic, eval on CelebA-Spoof | Task 16 (tiny-CNN scaffold + synthetic-eval placeholder; ResNet18 and real CelebA-Spoof eval slice deferred to Phase 2) |

**Phase-1 deferrals explicitly flagged in the plan:**
- Real DigiFace-1M ingestion (loader interface is real; data is fixture)
- ResNet18 (tiny-CNN scaffold ships instead)
- Real CelebA-Spoof eval slice (synthetic placeholder; swap is a config change)
- Multi-process orchestration (single-process loop; same enumerator feeds the future pool)
- Halftoning, ICC profiling, full subpixel-pattern modeling (MVP physics ships)

**Placeholder scan:** no "TBD"/"TODO"/"implement later"/etc. found.

**Type consistency:** `SampleRecord`, `BonafideSource`, `ManifestWriter`, `ProvenanceLedger`, `QCResult`, `WorkItem`, `Ontology`, `Axis`, `SensorPreset` are all defined in their introducing tasks and used by name in later tasks. `attack_type` is consistently `str` across manifest, work-item, and orchestrator. `attack_params` is consistently `dict[str, Any]`. `np.ndarray` is the image type throughout the face pipeline.

**Scope check:** all tasks contribute to a single coherent vertical slice (one modality, two attacks, end-to-end). Phases 2–4 will be separate plans.
