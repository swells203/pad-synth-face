# DefinitiveID PoC — Python Data+Models Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the DefinitiveID core mechanic — a model trained on synthetic behavioral sessions distinguishes the enrolled user from an imposter and flags non-human (bot) interaction, validated by an in-domain and a synthetic cross-domain EER, with no real-data or mobile-app dependency.

**Architecture:** A new `defid` uv-workspace package that depends on `pad-synth-core` for the genuinely reusable spine (deterministic RNG, the literature-citation-enforcing ontology loader, the provenance ledger, and `compute_eer`). Behavioral-specific modules — session schema, synthetic session generator, windowed feature extraction, a Mahalanobis one-class continuous-auth scorer, and a numpy logistic-regression bot classifier — are new. Pure-numpy models (no torch/GPU) keep the PoC deterministic, fast on CPU, and dependency-light; this is the right YAGNI choice for a "prove the mechanic" build and matches the spec's "start with a small baseline, escalate only if it plateaus" principle (§6.4).

**Tech Stack:** Python 3.11, numpy, pydantic v2, pyyaml, pytest. Reuses `pad-synth-core`. No torch, no GPU, no network.

---

## Spec & scope references

- Spec: `docs/superpowers/specs/2026-05-15-definitiveid-behavioral-biometrics-design.md`
- This plan implements spec milestones **M0, M1, M2, M4**. Spec milestone **M3** (standalone mobile demo app) is explicitly a separate follow-on plan — different toolchain, not executable in this environment, and a downstream consumer of the feature schema and model artifacts this plan produces.
- Cross-domain validation uses a **synthetic proxy** (a `domain` knob that deterministically perturbs generator parameters), mirroring the PAD Phase 1.5 precedent. Real public-dataset (Touchalytics) validation is documented in the spec roadmap as the follow-on, not built here.

## Reuse map (what comes from `pad-synth-core`, unchanged)

| Reused | Used for |
|---|---|
| `pad_synth_core.rng.derive_sample_seed`, `sample_rng` | Deterministic per-session seeds |
| `pad_synth_core.ontology.load_ontology`, `Ontology` | Behavioral ontology with the mandatory `provenance` lint |
| `pad_synth_core.provenance.ProvenanceLedger`, `OntologyCitation`, `BonafideIngested` | Generation audit trail |
| `pad_synth_core.eval.baseline.compute_eer` | EER metric for the continuous-auth scorer |

## File structure

```
defid/
├── pyproject.toml                       # depends on pad-synth-core (workspace)
├── src/defid/
│   ├── __init__.py
│   ├── session.py                       # BehavioralSession schema + JSONL manifest writer
│   ├── generator.py                     # deterministic genuine/imposter/bot session synthesis
│   ├── qc.py                            # per-session behavioral-plausibility checks
│   ├── features.py                      # windowing + fixed-length feature extraction
│   ├── models.py                        # MahalanobisAuth + LogisticBotClassifier (pure numpy)
│   ├── pipeline.py                      # config-driven generation (manifest + provenance)
│   ├── evaluate.py                      # in-domain + cross-domain proxy eval
│   └── cli.py                           # `defid generate` / `defid eval`
│   └── tests/  -> defid/tests/
├── defid/tests/                         # pytest (importlib mode, no __init__.py)
ontology/behavioral/
├── touch.yaml
├── keystroke.yaml
└── motion.yaml
configs/runs/
├── defid_poc_seta.yaml
└── defid_poc_setb.yaml
```

Workspace root `pyproject.toml` adds `defid` to `[tool.uv.workspace] members`.

---

## Task 1: Scaffold the `defid` package

**Files:**
- Modify: `pyproject.toml` (workspace members)
- Create: `defid/pyproject.toml`
- Create: `defid/src/defid/__init__.py`
- Create: `defid/tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

`defid/tests/test_smoke.py`:
```python
def test_import_defid():
    import defid
    assert defid.__version__ == "0.1.0"
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/stuartwells/test
.venv/bin/python -m pytest defid/tests/test_smoke.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'defid'`.

- [ ] **Step 3: Add the workspace member**

In the root `pyproject.toml`, find:
```toml
[tool.uv.workspace]
members = ["pad-synth-core", "pad-synth-face"]
```
Replace with:
```toml
[tool.uv.workspace]
members = ["pad-synth-core", "pad-synth-face", "defid"]
```

- [ ] **Step 4: Create the package files**

`defid/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "defid"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pad-synth-core",
    "numpy>=1.26",
    "pydantic>=2.5",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[project.scripts]
defid = "defid.cli:main"

[tool.uv.sources]
pad-synth-core = { workspace = true }
```

`defid/src/defid/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Sync and verify pass**

```bash
cd /Users/stuartwells/test
uv sync --all-extras 2>&1 | tail -3
.venv/bin/python -m pytest defid/tests/test_smoke.py -v 2>&1 | tail -10
```

Expected: 1 passed.

- [ ] **Step 6: Verify no regression**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```

Expected: all prior tests still pass + 1 new.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml defid/pyproject.toml defid/src/defid/__init__.py defid/tests/test_smoke.py
git commit -m "feat(defid): scaffold DefinitiveID package in workspace"
```

---

## Task 2: Behavioral ontology YAMLs

Three literature-cited ontology files defining the parameter ranges the generator samples from. They load via the existing `pad_synth_core.ontology.load_ontology`, which enforces a `provenance` field on every axis.

**Files:**
- Create: `ontology/behavioral/touch.yaml`
- Create: `ontology/behavioral/keystroke.yaml`
- Create: `ontology/behavioral/motion.yaml`
- Create: `defid/tests/test_ontology_files.py`

- [ ] **Step 1: Write the failing test**

`defid/tests/test_ontology_files.py`:
```python
from pathlib import Path

from pad_synth_core.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_touch_ontology_loads():
    o = load_ontology(ONT / "touch.yaml")
    assert o.attack_type == "touch"
    assert "touch_speed_mean" in o.axes
    assert "touch_jitter" in o.axes


def test_keystroke_ontology_loads():
    o = load_ontology(ONT / "keystroke.yaml")
    assert o.attack_type == "keystroke"
    assert "key_dwell_mean" in o.axes
    assert "key_flight_mean" in o.axes


def test_motion_ontology_loads():
    o = load_ontology(ONT / "motion.yaml")
    assert o.attack_type == "motion"
    assert "tremor_std" in o.axes
```

(`attack_type` is the ontology schema's discriminator field name from the PAD project; we reuse the loader as-is and treat it as the signal-family name. No loader change.)

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_ontology_files.py -v 2>&1 | tail -10
```

Expected: FileNotFoundError.

- [ ] **Step 3: Write `ontology/behavioral/touch.yaml`**

```yaml
version: "2026-05-15"
attack_type: touch
axes:
  touch_speed_mean:
    type: uniform
    low: 200.0
    high: 1400.0
    provenance:
      paper: "Frank et al., 'Touchalytics: On the Applicability of Touchscreen Input as a Behavioral Biometric for Continuous Authentication', IEEE TIFS 2013"
      doi: "10.1109/TIFS.2012.2225048"
  touch_speed_std:
    type: uniform
    low: 30.0
    high: 400.0
    provenance:
      paper: "Frank et al., Touchalytics, IEEE TIFS 2013 (per-stroke velocity dispersion)"
      doi: "10.1109/TIFS.2012.2225048"
  touch_curvature:
    type: uniform
    low: 0.01
    high: 0.35
    provenance:
      paper: "Serwadda et al., 'Which Behavioral Biometric Modality is Best for Continuous Authentication?', BTAS 2013"
      doi: "10.1109/BTAS.2013.6712747"
  touch_jitter:
    type: uniform
    low: 0.5
    high: 12.0
    provenance:
      paper: "Sitová et al., 'HMOG: New Behavioral Biometric Features for Continuous Authentication of Smartphone Users', IEEE TIFS 2016"
      doi: "10.1109/TIFS.2015.2506542"
  inter_touch_interval_ms:
    type: uniform
    low: 80.0
    high: 900.0
    provenance:
      paper: "Frank et al., Touchalytics, IEEE TIFS 2013 (inter-stroke timing)"
      doi: "10.1109/TIFS.2012.2225048"
```

- [ ] **Step 4: Write `ontology/behavioral/keystroke.yaml`**

```yaml
version: "2026-05-15"
attack_type: keystroke
axes:
  key_dwell_mean:
    type: uniform
    low: 60.0
    high: 180.0
    provenance:
      paper: "Killourhy & Maxion, 'Comparing Anomaly-Detection Algorithms for Keystroke Dynamics', DSN 2009"
      doi: "10.1109/DSN.2009.5270346"
  key_dwell_std:
    type: uniform
    low: 8.0
    high: 60.0
    provenance:
      paper: "Killourhy & Maxion, DSN 2009 (dwell-time dispersion)"
      doi: "10.1109/DSN.2009.5270346"
  key_flight_mean:
    type: uniform
    low: 80.0
    high: 320.0
    provenance:
      paper: "Killourhy & Maxion, DSN 2009 (flight-time distribution)"
      doi: "10.1109/DSN.2009.5270346"
  key_flight_std:
    type: uniform
    low: 15.0
    high: 140.0
    provenance:
      paper: "Killourhy & Maxion, DSN 2009 (flight-time dispersion)"
      doi: "10.1109/DSN.2009.5270346"
```

- [ ] **Step 5: Write `ontology/behavioral/motion.yaml`**

```yaml
version: "2026-05-15"
attack_type: motion
axes:
  accel_mag_mean:
    type: uniform
    low: 9.2
    high: 10.4
    provenance:
      paper: "Sitová et al., HMOG, IEEE TIFS 2016 (resting/holding acceleration magnitude near 1g)"
      doi: "10.1109/TIFS.2015.2506542"
  tremor_std:
    type: uniform
    low: 0.02
    high: 0.45
    provenance:
      paper: "Sitová et al., HMOG, IEEE TIFS 2016 (micro-tremor while holding device)"
      doi: "10.1109/TIFS.2015.2506542"
  motion_touch_coupling:
    type: uniform
    low: 0.15
    high: 0.85
    provenance:
      paper: "Bo et al., 'SilentSense: Silent User Identification via Touch and Movement Behavioral Biometrics', MobiCom 2013"
      doi: "10.1145/2500423.2504572"
```

- [ ] **Step 6: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_ontology_files.py -v 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add ontology/behavioral defid/tests/test_ontology_files.py
git commit -m "feat(defid): literature-cited behavioral ontology (touch/keystroke/motion)"
```

---

## Task 3: BehavioralSession schema + JSONL manifest writer

**Files:**
- Create: `defid/src/defid/session.py`
- Create: `defid/tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_session.py`:
```python
import json
from pathlib import Path

import pytest

from defid.session import BehavioralSession, SessionManifestWriter


def make_session(sid: str = "s1", label: str = "genuine") -> BehavioralSession:
    return BehavioralSession(
        session_id=sid,
        label=label,
        subject_id="subj-0",
        touch=[{"t": 0.0, "x": 1.0, "y": 2.0, "phase": "move"}],
        key=[{"t": 0.0, "phase": "down", "field": "f1"}],
        motion=[{"t": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8}],
        ontology_version="2026-05-15",
        generator_version="defid-gen@0.1.0",
        seed=42,
    )


def test_session_serializes():
    s = make_session()
    blob = json.loads(s.model_dump_json())
    assert blob["label"] == "genuine"
    assert blob["touch"][0]["x"] == 1.0


def test_session_rejects_bad_label():
    with pytest.raises(ValueError):
        BehavioralSession.model_validate(
            {**make_session().model_dump(), "label": "nope"}
        )


def test_manifest_writer_appends_and_resumes(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    w = SessionManifestWriter(path)
    w.append("a", "genuine", "subj-0", "sessions/a.json", "0" * 64)
    w.close()

    w2 = SessionManifestWriter(path)
    assert w2.existing_ids() == {"a"}
    w2.append("b", "bot", "subj-1", "sessions/b.json", "1" * 64)
    w2.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1])["session_id"] == "b"


def test_manifest_writer_tolerates_partial_line(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"session_id": "ok"}\n{"session_id": "partial')
    w = SessionManifestWriter(path)
    assert w.existing_ids() == {"ok"}
    w.close()
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_session.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.session`.

- [ ] **Step 3: Implement `defid/src/defid/session.py`**

```python
"""Behavioral session schema and an append-only JSONL session manifest.

The manifest stores per-session metadata + a pointer to the session payload
file and its sha256 (mirroring the PAD project's manifest/file split). The
full event arrays live in the payload file, not the manifest.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BehavioralSession(BaseModel):
    session_id: str
    label: Literal["genuine", "imposter", "bot"]
    subject_id: str
    touch: list[dict[str, Any]] = Field(default_factory=list)
    key: list[dict[str, Any]] = Field(default_factory=list)
    motion: list[dict[str, Any]] = Field(default_factory=list)
    ontology_version: str
    generator_version: str
    seed: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManifestWriter:
    """Append-only JSONL manifest. One row per session: id, label, subject,
    payload path, payload sha256. Tolerant of a partial trailing line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing = self._scan()
        self._fh = self.path.open("a", encoding="utf-8")

    def _scan(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["session_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a partial trailing line from a crash
        return ids

    def existing_ids(self) -> set[str]:
        return set(self._existing)

    def append(
        self,
        session_id: str,
        label: str,
        subject_id: str,
        payload_path: str,
        payload_sha256: str,
    ) -> None:
        if session_id in self._existing:
            return
        row = {
            "session_id": session_id,
            "label": label,
            "subject_id": subject_id,
            "payload_path": payload_path,
            "payload_sha256": payload_sha256,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._existing.add(session_id)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> "SessionManifestWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_session.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add defid/src/defid/session.py defid/tests/test_session.py
git commit -m "feat(defid): behavioral session schema + resumable JSONL manifest"
```

---

## Task 4: Synthetic session generator

Deterministically synthesizes a `BehavioralSession` for a given `(label, subject, seed, domain)`. Genuine = a stable per-subject motor profile with natural jitter; imposter = different profile; bot = degenerate jitter, machine-regular timing, no motion-touch coupling. `domain="b"` applies a fixed multiplicative shift to sampled parameters — the synthetic cross-domain proxy (Phase 1.5 precedent).

**Files:**
- Create: `defid/src/defid/generator.py`
- Create: `defid/tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_generator.py`:
```python
from pathlib import Path

import numpy as np

from defid.generator import GENERATOR_VERSION, generate_session

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_generate_is_deterministic():
    a = generate_session("genuine", "subj-1", seed=7, ontology_dir=ONT)
    b = generate_session("genuine", "subj-1", seed=7, ontology_dir=ONT)
    assert a.model_dump_json() == b.model_dump_json()


def test_three_labels_supported():
    for label in ("genuine", "imposter", "bot"):
        s = generate_session(label, "subj-1", seed=1, ontology_dir=ONT)
        assert s.label == label
        assert len(s.touch) > 0
        assert len(s.motion) > 0


def test_bot_has_lower_jitter_than_genuine():
    g = generate_session("genuine", "subj-1", seed=3, ontology_dir=ONT)
    b = generate_session("bot", "subj-1", seed=3, ontology_dir=ONT)

    def touch_speed_var(sess):
        xs = np.array([p["x"] for p in sess.touch])
        return float(np.var(np.diff(xs))) if len(xs) > 2 else 0.0

    # Bots move with near-constant velocity → far less speed variance.
    assert touch_speed_var(b) < touch_speed_var(g)


def test_genuine_subject_profile_is_stable_across_seeds():
    # Same subject, different session seeds → similar mean touch speed
    # (a subject has a stable motor profile; sessions vary around it).
    s1 = generate_session("genuine", "subj-9", seed=1, ontology_dir=ONT)
    s2 = generate_session("genuine", "subj-9", seed=2, ontology_dir=ONT)

    def mean_speed(sess):
        pts = np.array([[p["t"], p["x"], p["y"]] for p in sess.touch])
        d = np.linalg.norm(np.diff(pts[:, 1:], axis=0), axis=1)
        dt = np.diff(pts[:, 0])
        return float(np.mean(d / np.maximum(dt, 1e-3)))

    m1, m2 = mean_speed(s1), mean_speed(s2)
    assert abs(m1 - m2) / max(m1, m2) < 0.5  # within 50% — same profile


def test_domain_b_shifts_distribution():
    a = generate_session("genuine", "subj-1", seed=5, ontology_dir=ONT, domain="a")
    b = generate_session("genuine", "subj-1", seed=5, ontology_dir=ONT, domain="b")
    sa = np.array([p["x"] for p in a.touch])
    sb = np.array([p["x"] for p in b.touch])
    assert not np.allclose(sa, sb)
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_generator.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.generator`.

- [ ] **Step 3: Implement `defid/src/defid/generator.py`**

```python
"""Deterministic synthetic behavioral-session generator.

A subject has a stable motor profile derived from a per-subject seed. A
session adds per-session variation around that profile. Label semantics:
  genuine  — the subject's own profile + natural jitter
  imposter — a *different* profile (different subject hash) + jitter
  bot      — machine-regular timing, near-zero jitter, no motion coupling

domain="b" applies a fixed parameter shift — the synthetic cross-domain
proxy (mirrors PAD Phase 1.5's Set B).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import derive_sample_seed

GENERATOR_VERSION = "defid-gen@0.1.0"
_N_TOUCH = 60
_N_KEY = 25
_N_MOTION = 120


def _subject_seed(subject_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(subject_id.encode()).digest()[:4], "big"
    )


def _domain_scale(domain: str) -> float:
    # Deterministic, modest shift so Set B is a genuinely different
    # distribution but still plausible.
    return 1.0 if domain == "a" else 1.35


def generate_session(
    label: str,
    subject_id: str,
    seed: int,
    ontology_dir: Path,
    domain: str = "a",
):
    from defid.session import BehavioralSession

    if label not in ("genuine", "imposter", "bot"):
        raise ValueError(f"bad label {label!r}")

    touch_ont = load_ontology(Path(ontology_dir) / "touch.yaml")
    key_ont = load_ontology(Path(ontology_dir) / "keystroke.yaml")
    motion_ont = load_ontology(Path(ontology_dir) / "motion.yaml")

    # Profile identity: genuine/bot use the real subject; imposter uses a
    # different subject's profile (the attack is "wrong person").
    profile_id = subject_id if label != "imposter" else subject_id + "-imp"
    profile_rng = np.random.default_rng(_subject_seed(profile_id))
    tp = touch_ont.sample_params(profile_rng)
    kp = key_ont.sample_params(profile_rng)
    mp = motion_ont.sample_params(profile_rng)

    scale = _domain_scale(domain)
    sess_seed = derive_sample_seed(seed, "defid", label, _subject_seed(subject_id))
    rng = np.random.default_rng(sess_seed)

    is_bot = label == "bot"
    jitter = 0.02 if is_bot else float(tp["touch_jitter"])
    speed = float(tp["touch_speed_mean"]) * scale
    speed_sd = 1.0 if is_bot else float(tp["touch_speed_std"])
    iti = float(tp["inter_touch_interval_ms"]) / 1000.0

    t = 0.0
    x, y = 100.0, 100.0
    touch = []
    for _ in range(_N_TOUCH):
        step_speed = speed if is_bot else max(rng.normal(speed, speed_sd), 10.0)
        ang = 0.0 if is_bot else rng.normal(0.0, float(tp["touch_curvature"]))
        x += step_speed * 0.01 * np.cos(ang) + rng.normal(0.0, jitter)
        y += step_speed * 0.01 * np.sin(ang) + rng.normal(0.0, jitter)
        t += iti if is_bot else max(rng.normal(iti, iti * 0.3), 0.01)
        touch.append({"t": round(t, 5), "x": round(float(x), 4),
                       "y": round(float(y), 4), "phase": "move"})

    dwell = float(kp["key_dwell_mean"]) / 1000.0
    dwell_sd = 0.001 if is_bot else float(kp["key_dwell_std"]) / 1000.0
    flight = float(kp["key_flight_mean"]) / 1000.0
    flight_sd = 0.001 if is_bot else float(kp["key_flight_std"]) / 1000.0
    tk = 0.0
    key = []
    for _ in range(_N_KEY):
        d = dwell if is_bot else max(rng.normal(dwell, dwell_sd), 0.005)
        key.append({"t": round(tk, 5), "phase": "down", "field": "f1"})
        tk += d
        key.append({"t": round(tk, 5), "phase": "up", "field": "f1"})
        tk += flight if is_bot else max(rng.normal(flight, flight_sd), 0.01)

    amag = float(mp["accel_mag_mean"])
    tremor = 0.0 if is_bot else float(mp["tremor_std"])
    coupling = 0.0 if is_bot else float(mp["motion_touch_coupling"])
    motion = []
    for k in range(_N_MOTION):
        tm = k * (t / _N_MOTION if t > 0 else 0.01)
        base = amag + rng.normal(0.0, tremor)
        # Couple some motion energy to touch activity for non-bots.
        couple = coupling * (0.3 if (k % 5 == 0) else 0.0)
        motion.append({
            "t": round(tm, 5),
            "ax": round(float(rng.normal(0.0, tremor) + couple), 5),
            "ay": round(float(rng.normal(0.0, tremor) + couple), 5),
            "az": round(float(base), 5),
        })

    return BehavioralSession(
        session_id=f"{label}-{subject_id}-{seed}-{domain}",
        label=label,
        subject_id=subject_id,
        touch=touch,
        key=key,
        motion=motion,
        ontology_version=touch_ont.version,
        generator_version=GENERATOR_VERSION,
        seed=sess_seed,
    )
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_generator.py -v 2>&1 | tail -10
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add defid/src/defid/generator.py defid/tests/test_generator.py
git commit -m "feat(defid): deterministic genuine/imposter/bot session generator"
```

---

## Task 5: Per-session behavioral QC

**Files:**
- Create: `defid/src/defid/qc.py`
- Create: `defid/tests/test_qc.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_qc.py`:
```python
from pathlib import Path

from defid.generator import generate_session
from defid.qc import QCResult, check_session

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_qc_passes_on_generated_session():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    r = check_session(s)
    assert isinstance(r, QCResult)
    assert r.ok
    assert r.reason is None


def test_qc_fails_on_empty_touch():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    s.touch = []
    r = check_session(s)
    assert not r.ok
    assert "touch" in r.reason


def test_qc_fails_on_nonfinite():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    s.touch[0]["x"] = float("nan")
    r = check_session(s)
    assert not r.ok
    assert "finite" in r.reason


def test_qc_fails_on_implausible_speed():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    s.touch[5]["x"] = 1e9  # absurd jump
    r = check_session(s)
    assert not r.ok
    assert "speed" in r.reason
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_qc.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.qc`.

- [ ] **Step 3: Implement `defid/src/defid/qc.py`**

```python
"""Per-session behavioral-plausibility QC. Cheap, runs inline during
generation. Mirrors the PAD project's per-sample QC contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from defid.session import BehavioralSession

_MAX_HUMAN_TOUCH_SPEED = 50000.0  # px/s — generous upper plausibility bound


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reason: str | None = None


def check_session(s: BehavioralSession) -> QCResult:
    if len(s.touch) < 3:
        return QCResult(False, "too few touch events (<3)")
    if len(s.motion) < 3:
        return QCResult(False, "too few motion events (<3)")

    for arr_name, arr, keys in (
        ("touch", s.touch, ("t", "x", "y")),
        ("motion", s.motion, ("t", "ax", "ay", "az")),
    ):
        for ev in arr:
            for k in keys:
                v = ev[k]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    return QCResult(False, f"{arr_name}.{k} not finite")

    for i in range(1, len(s.touch)):
        dt = s.touch[i]["t"] - s.touch[i - 1]["t"]
        if dt <= 0:
            return QCResult(False, "non-monotonic touch timestamps")
        dx = s.touch[i]["x"] - s.touch[i - 1]["x"]
        dy = s.touch[i]["y"] - s.touch[i - 1]["y"]
        speed = math.hypot(dx, dy) / dt
        if speed > _MAX_HUMAN_TOUCH_SPEED:
            return QCResult(False, f"implausible touch speed {speed:.0f}")

    return QCResult(True)
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_qc.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add defid/src/defid/qc.py defid/tests/test_qc.py
git commit -m "feat(defid): per-session behavioral-plausibility QC"
```

---

## Task 6: Windowing + feature extraction

**Files:**
- Create: `defid/src/defid/features.py`
- Create: `defid/tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_features.py`:
```python
from pathlib import Path

import numpy as np

from defid.features import FEATURE_NAMES, extract_features
from defid.generator import generate_session

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_feature_vector_fixed_length_and_finite():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    v = extract_features(s)
    assert v.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(v).all()


def test_features_are_deterministic():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    assert np.array_equal(extract_features(s), extract_features(s))


def test_bot_separable_from_genuine_on_jitter_feature():
    g = generate_session("genuine", "subj-1", seed=2, ontology_dir=ONT)
    b = generate_session("bot", "subj-1", seed=2, ontology_dir=ONT)
    gi = FEATURE_NAMES.index("touch_speed_std")
    assert extract_features(b)[gi] < extract_features(g)[gi]


def test_genuine_vs_imposter_differ():
    g = generate_session("genuine", "subj-1", seed=2, ontology_dir=ONT)
    imp = generate_session("imposter", "subj-1", seed=2, ontology_dir=ONT)
    assert not np.allclose(extract_features(g), extract_features(imp))
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_features.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.features`.

- [ ] **Step 3: Implement `defid/src/defid/features.py`**

```python
"""Session -> fixed-length feature vector. Versioned feature schema."""

from __future__ import annotations

import numpy as np

from defid.session import BehavioralSession

FEATURE_SCHEMA_VERSION = "defid-feat@0.1.0"
FEATURE_NAMES = [
    "touch_speed_mean",
    "touch_speed_std",
    "touch_curvature_mean",
    "touch_jitter",
    "inter_touch_interval_mean",
    "key_dwell_mean",
    "key_dwell_std",
    "key_flight_mean",
    "key_flight_std",
    "key_paste_ratio",
    "accel_mag_mean",
    "tremor_std",
    "motion_touch_coupling",
    "touch_without_motion_ratio",
]


def _safe(a: np.ndarray, fn, default: float = 0.0) -> float:
    return float(fn(a)) if a.size else default


def extract_features(s: BehavioralSession) -> np.ndarray:
    tx = np.array([[p["t"], p["x"], p["y"]] for p in s.touch], dtype=np.float64)
    if tx.shape[0] >= 2:
        d = np.linalg.norm(np.diff(tx[:, 1:], axis=0), axis=1)
        dt = np.maximum(np.diff(tx[:, 0]), 1e-3)
        speed = d / dt
        ang = np.arctan2(np.diff(tx[:, 2]), np.diff(tx[:, 1]))
        curv = np.abs(np.diff(ang)) if ang.size >= 2 else np.array([0.0])
        iti = np.diff(tx[:, 0])
    else:
        speed = curv = iti = np.array([0.0])

    downs = [k["t"] for k in s.key if k["phase"] == "down"]
    ups = [k["t"] for k in s.key if k["phase"] == "up"]
    m = min(len(downs), len(ups))
    dwell = np.array([ups[i] - downs[i] for i in range(m)]) if m else np.array([0.0])
    flight = (
        np.array([downs[i + 1] - ups[i] for i in range(m - 1)])
        if m >= 2
        else np.array([0.0])
    )
    paste_ratio = float(np.mean(flight < 0.005)) if flight.size else 0.0

    mo = np.array([[e["ax"], e["ay"], e["az"]] for e in s.motion], dtype=np.float64)
    amag = np.linalg.norm(mo, axis=1) if mo.size else np.array([0.0])
    tremor = _safe(amag, np.std)
    coupling = (
        float(np.corrcoef(amag[: speed.size], speed[: amag.size])[0, 1])
        if min(amag.size, speed.size) >= 2
        else 0.0
    )
    coupling = 0.0 if not np.isfinite(coupling) else coupling
    twm = 1.0 if mo.size == 0 and tx.size > 0 else 0.0

    vec = np.array(
        [
            _safe(speed, np.mean),
            _safe(speed, np.std),
            _safe(curv, np.mean),
            _safe(speed, lambda a: np.std(np.diff(a)) if a.size >= 2 else 0.0),
            _safe(iti, np.mean),
            _safe(dwell, np.mean),
            _safe(dwell, np.std),
            _safe(flight, np.mean),
            _safe(flight, np.std),
            paste_ratio,
            _safe(amag, np.mean),
            tremor,
            coupling,
            twm,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_features.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add defid/src/defid/features.py defid/tests/test_features.py
git commit -m "feat(defid): windowed feature extraction (versioned schema)"
```

---

## Task 7: Continuous-auth model (Mahalanobis one-class)

**Files:**
- Create: `defid/src/defid/models.py`
- Create: `defid/tests/test_models_auth.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_models_auth.py`:
```python
import numpy as np

from defid.models import MahalanobisAuth


def test_auth_scores_imposter_higher_than_genuine():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.0, 1.0, size=(80, 6))
    imposter = rng.normal(4.0, 1.0, size=(40, 6))

    m = MahalanobisAuth().fit(genuine)
    g_scores = m.score(genuine)
    i_scores = m.score(imposter)
    assert i_scores.mean() > g_scores.mean()


def test_auth_eer_low_on_separable_data():
    rng = np.random.default_rng(1)
    genuine = rng.normal(0.0, 1.0, size=(100, 6))
    imposter = rng.normal(5.0, 1.0, size=(100, 6))

    m = MahalanobisAuth().fit(genuine)
    eer = m.eer(genuine, imposter)
    assert eer < 0.1


def test_auth_is_deterministic():
    rng = np.random.default_rng(2)
    g = rng.normal(0.0, 1.0, size=(50, 6))
    s1 = MahalanobisAuth().fit(g).score(g)
    s2 = MahalanobisAuth().fit(g).score(g)
    assert np.array_equal(s1, s2)
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_models_auth.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.models`.

- [ ] **Step 3: Implement `defid/src/defid/models.py`** (auth class; bot class added in Task 8)

```python
"""Pure-numpy PoC models. No torch, deterministic, CPU-instant.

MahalanobisAuth: one-class continuous-auth scorer. Fit on the enrolled
user's feature windows; score = Mahalanobis distance to that profile.
Higher score = more imposter-like.
"""

from __future__ import annotations

import numpy as np

from pad_synth_core.eval.baseline import compute_eer


class MahalanobisAuth:
    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None

    def fit(self, genuine: np.ndarray) -> "MahalanobisAuth":
        x = np.asarray(genuine, dtype=np.float64)
        self._mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * self.reg
        self._inv_cov = np.linalg.pinv(cov)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._inv_cov is not None
        d = np.asarray(x, dtype=np.float64) - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self._inv_cov, d))

    def eer(self, genuine: np.ndarray, imposter: np.ndarray) -> float:
        gs = self.score(genuine)
        is_ = self.score(imposter)
        scores = np.concatenate([gs, is_]).tolist()
        labels = [0] * len(gs) + [1] * len(is_)  # 1 = imposter = positive
        return compute_eer(scores, labels)
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_models_auth.py -v 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add defid/src/defid/models.py defid/tests/test_models_auth.py
git commit -m "feat(defid): Mahalanobis one-class continuous-auth scorer"
```

---

## Task 8: Bot classifier (numpy logistic regression)

**Files:**
- Modify: `defid/src/defid/models.py`
- Create: `defid/tests/test_models_bot.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_models_bot.py`:
```python
import numpy as np

from defid.models import LogisticBotClassifier


def test_bot_classifier_separates_classes():
    rng = np.random.default_rng(0)
    human = rng.normal(0.0, 1.0, size=(100, 5))
    bot = rng.normal(3.0, 0.3, size=(100, 5))
    X = np.vstack([human, bot])
    y = np.array([0] * 100 + [1] * 100)

    clf = LogisticBotClassifier(seed=0).fit(X, y)
    preds = (clf.predict_proba(X) >= 0.5).astype(int)
    acc = (preds == y).mean()
    assert acc > 0.9


def test_bot_classifier_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    p1 = LogisticBotClassifier(seed=3).fit(X, y).predict_proba(X)
    p2 = LogisticBotClassifier(seed=3).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_models_bot.py -v 2>&1 | tail -10
```

Expected: ImportError on `LogisticBotClassifier`.

- [ ] **Step 3: Append `LogisticBotClassifier` to `defid/src/defid/models.py`**

```python
class LogisticBotClassifier:
    """Deterministic logistic regression via fixed-iteration gradient
    descent on standardized features. Label 1 = bot."""

    def __init__(self, seed: int = 0, lr: float = 0.1, iters: int = 2000) -> None:
        self.seed = seed
        self.lr = lr
        self.iters = iters
        self._w: np.ndarray | None = None
        self._b = 0.0
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mu) / self._sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBotClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        Xs = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        self._w = rng.normal(0.0, 0.01, size=Xs.shape[1])
        self._b = 0.0
        n = Xs.shape[0]
        for _ in range(self.iters):
            z = Xs @ self._w + self._b
            p = 1.0 / (1.0 + np.exp(-z))
            gw = Xs.T @ (p - y) / n
            gb = float(np.mean(p - y))
            self._w -= self.lr * gw
            self._b -= self.lr * gb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._w is not None
        Xs = self._standardize(np.asarray(X, dtype=np.float64))
        return 1.0 / (1.0 + np.exp(-(Xs @ self._w + self._b)))
```

- [ ] **Step 4: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_models_bot.py -v 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add defid/src/defid/models.py defid/tests/test_models_bot.py
git commit -m "feat(defid): deterministic logistic-regression bot classifier"
```

---

## Task 9: Generation pipeline + `defid generate` CLI

**Files:**
- Create: `defid/src/defid/pipeline.py`
- Create: `defid/src/defid/cli.py`
- Create: `configs/runs/defid_poc_seta.yaml`
- Create: `configs/runs/defid_poc_setb.yaml`
- Create: `defid/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`defid/tests/test_pipeline.py`:
```python
import json
from pathlib import Path

import yaml

from defid.pipeline import run_generation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, domain: str) -> Path:
    cfg = {
        "run": {"name": "t", "output": str(tmp_path / "out"), "seed": 11},
        "ontology_dir": str(REPO_ROOT / "ontology" / "behavioral"),
        "domain": domain,
        "subjects": 4,
        "sessions_per_subject": 3,
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_generation_produces_manifest_and_payloads(tmp_path: Path):
    summary = run_generation(_cfg(tmp_path, "a"))
    # 4 subjects × 3 sessions × 3 labels (genuine/imposter/bot)
    assert summary["generated"] == 4 * 3 * 3
    assert summary["failed"] == 0
    out = Path(tmp_path / "out")
    manifest = (out / "manifest.jsonl").read_text().strip().split("\n")
    assert len(manifest) == 36
    first = json.loads(manifest[0])
    assert (out / first["payload_path"]).exists()
    prov = (out / "provenance.jsonl").read_text()
    assert "ontology_citation" in prov


def test_generation_is_resumable(tmp_path: Path):
    cfg = _cfg(tmp_path, "a")
    first = run_generation(cfg)
    second = run_generation(cfg)
    assert first["generated"] == 36
    assert second["generated"] == 0
    assert second["skipped"] == 36
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_pipeline.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.pipeline`.

- [ ] **Step 3: Implement `defid/src/defid/pipeline.py`**

```python
"""Config-driven behavioral-session generation with manifest + provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from pad_synth_core.provenance import (
    BonafideIngested,
    OntologyCitation,
    ProvenanceLedger,
)
from defid.generator import generate_session
from defid.qc import check_session
from defid.session import SessionManifestWriter

_LABELS = ("genuine", "imposter", "bot")


def _record_citations(ledger: ProvenanceLedger, ontology_dir: Path) -> None:
    for fname in ("touch.yaml", "keystroke.yaml", "motion.yaml"):
        raw = yaml.safe_load((ontology_dir / fname).read_text())
        for axis, body in raw["axes"].items():
            prov = body["provenance"]
            ledger.record(
                OntologyCitation(
                    attack_type=raw["attack_type"],
                    axis=axis,
                    paper=prov["paper"],
                    doi=prov.get("doi"),
                    url=prov.get("url"),
                )
            )


def run_generation(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["run"]["output"])
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    ontology_dir = Path(cfg["ontology_dir"])
    domain = cfg.get("domain", "a")
    n_subjects = int(cfg["subjects"])
    n_sessions = int(cfg["sessions_per_subject"])
    master_seed = int(cfg["run"]["seed"])

    generated = failed = skipped = 0
    with SessionManifestWriter(out / "manifest.jsonl") as manifest, \
            ProvenanceLedger(out / "provenance.jsonl") as ledger:
        ledger.record(
            BonafideIngested(
                name="defid_synthetic",
                license="OWNED",
                source_url=str(ontology_dir),
                sha256_of_index=hashlib.sha256(
                    str(sorted(p.name for p in ontology_dir.glob("*.yaml"))).encode()
                ).hexdigest(),
            )
        )
        _record_citations(ledger, ontology_dir)
        existing = manifest.existing_ids()

        for subj in range(n_subjects):
            subject_id = f"subj-{subj:03d}"
            for sess in range(n_sessions):
                for label in _LABELS:
                    sid = f"{label}-{subject_id}-{sess}-{domain}"
                    if sid in existing:
                        skipped += 1
                        continue
                    s = generate_session(
                        label, subject_id, seed=master_seed * 1000 + sess,
                        ontology_dir=ontology_dir, domain=domain,
                    )
                    s.session_id = sid
                    qc = check_session(s)
                    if not qc.ok:
                        failed += 1
                        continue
                    rel = f"sessions/{sid}.json"
                    blob = s.model_dump_json()
                    (out / rel).write_text(blob)
                    sha = hashlib.sha256(blob.encode()).hexdigest()
                    manifest.append(sid, label, subject_id, rel, sha)
                    generated += 1

    return {"generated": generated, "failed": failed, "skipped": skipped}
```

- [ ] **Step 4: Implement `defid/src/defid/cli.py`** (generate; eval added in Task 10)

```python
"""DefinitiveID PoC CLI: `defid generate` (eval added in Task 10)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from defid.pipeline import run_generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="defid")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="Generate a synthetic behavioral dataset")
    g.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        summary = run_generation(args.config)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Write the configs**

`configs/runs/defid_poc_seta.yaml`:
```yaml
run:
  name: defid_poc_seta
  output: ./datasets/defid_poc_seta
  seed: 20260515
ontology_dir: ./ontology/behavioral
domain: a
subjects: 12
sessions_per_subject: 8
```

`configs/runs/defid_poc_setb.yaml`:
```yaml
run:
  name: defid_poc_setb
  output: ./datasets/defid_poc_setb
  seed: 20260516
ontology_dir: ./ontology/behavioral
domain: b
subjects: 12
sessions_per_subject: 8
```

- [ ] **Step 6: Re-sync (entry point) and verify**

```bash
cd /Users/stuartwells/test
uv sync --all-extras 2>&1 | tail -2
.venv/bin/python -m pytest defid/tests/test_pipeline.py -v 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add defid/src/defid/pipeline.py defid/src/defid/cli.py configs/runs/defid_poc_seta.yaml configs/runs/defid_poc_setb.yaml defid/tests/test_pipeline.py
git commit -m "feat(defid): generation pipeline + generate CLI + PoC configs"
```

---

## Task 10: Evaluation + cross-domain proxy + `defid eval`

**Files:**
- Create: `defid/src/defid/evaluate.py`
- Modify: `defid/src/defid/cli.py`
- Create: `defid/tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

`defid/tests/test_evaluate.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import yaml

from defid.evaluate import evaluate
from defid.pipeline import run_generation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _gen(tmp_path: Path, name: str, domain: str, seed: int) -> Path:
    cfg = {
        "run": {"name": name, "output": str(tmp_path / name), "seed": seed},
        "ontology_dir": str(REPO_ROOT / "ontology" / "behavioral"),
        "domain": domain,
        "subjects": 8,
        "sessions_per_subject": 6,
    }
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run_generation(p)
    return tmp_path / name


def test_evaluate_in_domain_only(tmp_path: Path):
    a = _gen(tmp_path, "a", "a", 1)
    r = evaluate(a, None)
    assert 0.0 <= r["auth_eer_in_domain"] <= 1.0
    assert 0.0 <= r["bot_accuracy_in_domain"] <= 1.0
    assert r["auth_eer_cross_domain"] is None


def test_evaluate_cross_domain(tmp_path: Path):
    a = _gen(tmp_path, "a", "a", 1)
    b = _gen(tmp_path, "b", "b", 2)
    r = evaluate(a, b)
    assert r["auth_eer_cross_domain"] is not None
    assert 0.0 <= r["auth_eer_cross_domain"] <= 1.0
    assert r["bot_accuracy_cross_domain"] is not None


def test_cli_eval_runs(tmp_path: Path):
    a = _gen(tmp_path, "a", "a", 1)
    b = _gen(tmp_path, "b", "b", 2)
    res = subprocess.run(
        [sys.executable, "-m", "defid.cli", "eval",
         "--train-root", str(a), "--eval-root", str(b)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["auth_eer_cross_domain"] is not None
```

- [ ] **Step 2: Verify failure**

```bash
.venv/bin/python -m pytest defid/tests/test_evaluate.py -v 2>&1 | tail -10
```

Expected: ImportError on `defid.evaluate`.

- [ ] **Step 3: Implement `defid/src/defid/evaluate.py`**

```python
"""Train on a generated session set; evaluate continuous-auth EER and bot
accuracy in-domain, and optionally cross-domain on a second set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from defid.features import extract_features
from defid.models import LogisticBotClassifier, MahalanobisAuth
from defid.session import BehavioralSession


def _load(root: Path) -> dict[str, list[np.ndarray]]:
    by_label: dict[str, list[np.ndarray]] = {
        "genuine": [], "imposter": [], "bot": []
    }
    for line in (root / "manifest.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        s = BehavioralSession.model_validate_json(
            (root / row["payload_path"]).read_text()
        )
        by_label[row["label"]].append(extract_features(s))
    return {k: v for k, v in by_label.items()}


def _auth_eer(train: dict, test: dict) -> float:
    g_train = np.vstack(train["genuine"])
    m = MahalanobisAuth().fit(g_train)
    return m.eer(np.vstack(test["genuine"]), np.vstack(test["imposter"]))


def _bot_acc(train: dict, test: dict) -> float:
    Xtr = np.vstack(train["genuine"] + train["imposter"] + train["bot"])
    ytr = np.array(
        [0] * len(train["genuine"])
        + [0] * len(train["imposter"])
        + [1] * len(train["bot"])
    )
    clf = LogisticBotClassifier(seed=0).fit(Xtr, ytr)
    Xte = np.vstack(test["genuine"] + test["imposter"] + test["bot"])
    yte = np.array(
        [0] * len(test["genuine"])
        + [0] * len(test["imposter"])
        + [1] * len(test["bot"])
    )
    preds = (clf.predict_proba(Xte) >= 0.5).astype(int)
    return float((preds == yte).mean())


def evaluate(train_root: Path, eval_root: Path | None) -> dict[str, Any]:
    train = _load(Path(train_root))
    result: dict[str, Any] = {
        "auth_eer_in_domain": _auth_eer(train, train),
        "bot_accuracy_in_domain": _bot_acc(train, train),
        "auth_eer_cross_domain": None,
        "bot_accuracy_cross_domain": None,
    }
    if eval_root is not None:
        ev = _load(Path(eval_root))
        result["auth_eer_cross_domain"] = _auth_eer(train, ev)
        result["bot_accuracy_cross_domain"] = _bot_acc(train, ev)
    return result
```

- [ ] **Step 4: Add the `eval` subcommand to `defid/src/defid/cli.py`**

Replace the file with:
```python
"""DefinitiveID PoC CLI: `defid generate` / `defid eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from defid.pipeline import run_generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="defid")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a synthetic behavioral dataset")
    g.add_argument("--config", required=True, type=Path)

    e = sub.add_parser("eval", help="Train + evaluate auth EER and bot accuracy")
    e.add_argument("--train-root", required=True, type=Path)
    e.add_argument("--eval-root", required=False, type=Path, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        json.dump(run_generation(args.config), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "eval":
        from defid.evaluate import evaluate

        json.dump(
            evaluate(args.train_root, args.eval_root), sys.stdout, indent=2
        )
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Verify pass**

```bash
.venv/bin/python -m pytest defid/tests/test_evaluate.py -v 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add defid/src/defid/evaluate.py defid/src/defid/cli.py defid/tests/test_evaluate.py
git commit -m "feat(defid): in-domain + cross-domain eval and eval CLI"
```

---

## Task 11: End-to-end PoC integration test + decision readout

**Files:**
- Create: `defid/tests/test_poc_integration.py`

- [ ] **Step 1: Write the integration test**

`defid/tests/test_poc_integration.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_poc_end_to_end(tmp_path: Path):
    import yaml

    def gen(name, domain, seed):
        cfg = {
            "run": {"name": name, "output": str(tmp_path / name), "seed": seed},
            "ontology_dir": str(REPO_ROOT / "ontology" / "behavioral"),
            "domain": domain,
            "subjects": 10,
            "sessions_per_subject": 8,
        }
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        r = subprocess.run(
            [sys.executable, "-m", "defid.cli", "generate", "--config", str(p)],
            capture_output=True, text=True, check=False,
        )
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    sa = gen("seta", "a", 20260515)
    sb = gen("setb", "b", 20260516)
    assert sa["generated"] == 10 * 8 * 3
    assert sb["generated"] == 10 * 8 * 3
    assert sa["failed"] == 0

    r = subprocess.run(
        [sys.executable, "-m", "defid.cli", "eval",
         "--train-root", str(tmp_path / "seta"),
         "--eval-root", str(tmp_path / "setb")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    # The PoC mechanic must work in-domain: auth must beat chance and the
    # bot classifier must be clearly better than random.
    assert out["auth_eer_in_domain"] < 0.45
    assert out["bot_accuracy_in_domain"] > 0.75
    # Cross-domain numbers must be produced (their value is the deliverable,
    # interpreted via the same decision-matrix discipline as the PAD project).
    assert out["auth_eer_cross_domain"] is not None
    assert out["bot_accuracy_cross_domain"] is not None
```

- [ ] **Step 2: Run the integration test**

```bash
cd /Users/stuartwells/test
.venv/bin/python -m pytest defid/tests/test_poc_integration.py -v 2>&1 | tail -15
```

Expected: 1 passed.

- [ ] **Step 3: Run the full suite**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -5
```

Expected: all tests pass (existing PAD suite + all new `defid` tests).

- [ ] **Step 4: Commit**

```bash
git add defid/tests/test_poc_integration.py
git commit -m "test(defid): end-to-end PoC integration (generate + cross-domain eval)"
```

---

## Final verification (manual — produces the headline numbers)

- [ ] **Step 1: Full suite green**

```bash
cd /Users/stuartwells/test
.venv/bin/python -m pytest -q 2>&1 | tail -5
```

- [ ] **Step 2: Generate both sets and run the cross-domain eval**

```bash
rm -rf datasets/defid_poc_seta datasets/defid_poc_setb
.venv/bin/python -m defid.cli generate --config configs/runs/defid_poc_seta.yaml | tail -5
.venv/bin/python -m defid.cli generate --config configs/runs/defid_poc_setb.yaml | tail -5
.venv/bin/python -m defid.cli eval \
    --train-root datasets/defid_poc_seta \
    --eval-root  datasets/defid_poc_setb | tail -10
```

Record `auth_eer_in_domain`, `auth_eer_cross_domain`, `bot_accuracy_in_domain`, `bot_accuracy_cross_domain`. Interpret with the same regime logic the PAD project used: in-domain numbers prove the mechanic; the cross-domain gap indicates whether the synthetic behavioral distribution generalizes or the generator needs richer modeling before any real-data step.

---

## Self-Review

**Spec coverage (this plan = spec M0/M1/M2/M4; M3 explicitly deferred to a separate plan):**

| Spec element | Task |
|---|---|
| M0 — synthetic behavioral generator, deterministic, ontology-driven, manifest+provenance | Tasks 2, 3, 4, 9 |
| M0 — behavioral QC | Task 5 |
| M1 — windowed feature extraction | Task 6 |
| M1 — continuous-auth model + EER | Task 7 |
| M2 — bot/automation classifier | Task 8 |
| M4 — cross-domain (synthetic proxy) eval + report | Tasks 10, 11, Final Verification |
| Reuse of pad-synth-core spine (rng, ontology, provenance, compute_eer) | Tasks 2, 4, 7, 9 |
| Determinism discipline (seeded, reproducible) | Tasks 4, 7, 8 (explicit determinism tests) |
| Privacy-by-construction (no content, only behavioral metadata) | Generator emits only behavioral features by construction; no content field exists in the schema (Task 3) |
| Spec §7.3 PoC out-of-scope (server, ATO/RAT/scam, wearable, real data, mobile app) | Not implemented by design; M3 split called out at top of plan |

**Placeholder scan:** No "TBD/TODO/implement later/similar to Task N". Every code step contains complete code.

**Type consistency:** `BehavioralSession` fields (`session_id, label, subject_id, touch, key, motion, ontology_version, generator_version, seed`) are defined in Task 3 and used identically in Tasks 4, 5, 6, 9, 10. `extract_features` returns a vector of length `len(FEATURE_NAMES)` (Task 6), consumed by `MahalanobisAuth`/`LogisticBotClassifier` (Tasks 7, 8) and `evaluate` (Task 10). `compute_eer(scores, labels)` signature matches its PAD-project definition (positive label = 1 = imposter). `generate_session(label, subject_id, seed, ontology_dir, domain)` signature is consistent across Tasks 4, 9, 10, 11. CLI subcommands `generate`/`eval` consistent across Tasks 9, 10.

No issues found.
