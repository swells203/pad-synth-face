# DefinitiveID Live Partner Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live in-browser enroll-and-impostor behavioral-biometrics demo: a phone captures touch+keystroke timing, a laptop-local FastAPI service runs the validated `defid` scorer live-enrolled on the subject, and a Dashboard visibly accepts the enrollee and rejects impostors.

**Architecture:** New uv-workspace package `defid-demo-pkg` (import `defid_demo`). A pure, fully-tested `DemoService` orchestrates an adapter (browser JSON → `defid` session dicts), a windowed feature extractor that calls the *actual* `defid.features.extract_features` and slices the 9 touch+keystroke features, and `DemoAuth` (a `MahalanobisAuth` subclass adding shrinkage + constant-column drop + empirical threshold). A thin FastAPI `app.py` exposes it over LAN; static web clients capture and visualize. The existing `defid` package is never modified.

**Tech Stack:** Python 3.11+, numpy, pydantic, FastAPI + uvicorn, pytest (`--import-mode=importlib`, no `tests/__init__.py`, no `conftest.py`). Reuses `defid.features.extract_features`, `defid.models.MahalanobisAuth`, `defid.session.BehavioralSession`, `defid.qc.QCResult`.

---

## Reference: facts the engineer needs

**Repo layout & conventions** (verified): uv workspace. Root `pyproject.toml` has `[tool.uv.workspace] members = ["pad-synth-core", "pad-synth-face", "defid-pkg"]` and `[tool.pytest.ini_options] testpaths = ["pad-synth-core/tests", "pad-synth-face/tests", "defid-pkg/tests", "tests"]` plus `addopts` using `--import-mode=importlib`. Packages are `<name>-pkg/` directories whose import name differs (e.g. `defid-pkg/src/defid/`). Tests have **no** `__init__.py` and **no** `conftest.py`. The directory must be `defid-demo-pkg` (a hyphen makes it a non-importable dir name so it cannot shadow the installed `defid_demo` package under importlib mode — this exact bug bit the `defid` package and was fixed by the hyphen rename; do not repeat it).

**`defid.features.FEATURE_NAMES`** order (indices): `0 touch_speed_mean, 1 touch_speed_std, 2 touch_curvature_mean, 3 touch_jitter, 4 inter_touch_interval_mean, 5 key_dwell_mean, 6 key_dwell_std, 7 key_flight_mean, 8 key_flight_std, 9 key_paste_ratio, 10 accel_mag_mean, 11 tremor_std, 12 motion_touch_coupling, 13 touch_without_motion_ratio`. The demo subset is indices **0..8** (`FEATURE_NAMES[:9]`).

**`defid.features.extract_features(s: BehavioralSession) -> np.ndarray(14,)`**: reads `s.touch` items `{"t","x","y"}`, `s.key` items `{"t","phase"}` where `phase ∈ {"down","up"}` (pairs the i-th down with the i-th up), `s.motion` items `{"t","ax","ay","az"}`. With `motion=[]` it returns a finite 14-vector (indices 10–13 become 0/constant); slicing `[:9]` yields exactly the touch+keystroke subset using the identical validated formulas.

**`defid.session.BehavioralSession`** required fields: `session_id:str, label:Literal["genuine","imposter","bot"], subject_id:str, ontology_version:str, generator_version:str, seed:int`. `touch/key/motion` default to `[]`. `created_at` has a default.

**`defid.models.MahalanobisAuth`**: `__init__(self, reg=1e-3)`; `fit(self, genuine: np.ndarray) -> self` sets `self._mean`, `self._inv_cov` (`cov = np.cov(x, rowvar=False); cov = np.atleast_2d(cov); cov += np.eye(n)*reg; self._inv_cov = np.linalg.pinv(cov)`); `score(self, x: np.ndarray) -> np.ndarray` returns `sqrt(einsum("ij,jk,ik->i", d, inv_cov, d))` where `d = x - self._mean`.

**`defid.qc.QCResult`**: `@dataclass(frozen=True) class QCResult: ok: bool; reason: str | None = None`. Import and reuse it; do not redefine.

## File Structure

```
defid-demo-pkg/
  pyproject.toml                         # package "defid-demo", import defid_demo
  src/defid_demo/__init__.py
  src/defid_demo/adapter.py              # browser JSON -> (touch, key) defid dicts
  src/defid_demo/windows.py              # FEATURE_SUBSET + windowed extraction via defid.features
  src/defid_demo/qc.py                   # check_rep() -> defid.qc.QCResult
  src/defid_demo/demo_auth.py            # DemoAuth(MahalanobisAuth): shrinkage + drop + threshold + verdict
  src/defid_demo/service.py              # DemoService: enroll/calibrate/attempt/reset (pure, in-memory)
  src/defid_demo/app.py                  # thin FastAPI shell over DemoService + static mount
  src/defid_demo/web/index.html          # mobile capture client + Dashboard
  src/defid_demo/web/spectator.html      # read-only spectator view
  tests/test_smoke.py
  tests/test_adapter.py
  tests/test_windows.py
  tests/test_qc.py
  tests/test_demo_auth.py
  tests/test_service_flow.py             # headless enroll->calibrate->genuine->impostor
  tests/test_app_smoke.py                # FastAPI TestClient
```

Modify (workspace registration only): root `pyproject.toml` (`members`, `testpaths`).

---

## Task 1: Scaffold `defid-demo-pkg`

**Files:**
- Create: `defid-demo-pkg/pyproject.toml`
- Create: `defid-demo-pkg/src/defid_demo/__init__.py`
- Create: `defid-demo-pkg/tests/test_smoke.py`
- Modify: `pyproject.toml` (root: add workspace member + testpath)

- [ ] **Step 1: Write the failing smoke test**

`defid-demo-pkg/tests/test_smoke.py`:
```python
def test_import_defid_demo():
    import defid_demo

    assert defid_demo.__name__ == "defid_demo"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/stuartwells/test && .venv/bin/python -m pytest defid-demo-pkg/tests/test_smoke.py -q 2>&1 | tail -5`
Expected: collection error / `ModuleNotFoundError: No module named 'defid_demo'`.

- [ ] **Step 3: Create the package files**

`defid-demo-pkg/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "defid-demo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "defid",
    "pad-synth-core",
    "numpy>=1.26",
    "pydantic>=2.5",
    "fastapi>=0.110",
    "uvicorn>=0.27",
]

[project.optional-dependencies]
test = ["pytest>=8.0", "httpx>=0.27"]

[tool.uv.sources]
defid = { workspace = true }
pad-synth-core = { workspace = true }
```

`defid-demo-pkg/src/defid_demo/__init__.py`:
```python
"""DefinitiveID live partner demo: in-browser enroll-and-impostor."""
```

- [ ] **Step 4: Register the package in the root workspace**

In `/Users/stuartwells/test/pyproject.toml`, change the members line to include `defid-demo-pkg` and the testpaths to include its tests. Set:

```toml
members = ["pad-synth-core", "pad-synth-face", "defid-pkg", "defid-demo-pkg"]
```
and
```toml
testpaths = ["pad-synth-core/tests", "pad-synth-face/tests", "defid-pkg/tests", "defid-demo-pkg/tests", "tests"]
```

- [ ] **Step 5: Sync the environment**

Run: `cd /Users/stuartwells/test && uv sync 2>&1 | tail -5`
Expected: resolves and installs `fastapi`, `uvicorn`, `httpx`, and the new editable `defid-demo` package. If `uv sync` cannot reach the network, stop and report BLOCKED (fastapi/uvicorn/httpx are required external deps).

- [ ] **Step 6: Run the smoke test to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_smoke.py -q 2>&1 | tail -5`
Expected: 1 passed.

- [ ] **Step 7: Full suite still green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: 103 passed (102 prior + 1 new), 4 warnings.

- [ ] **Step 8: Commit**

```bash
git add defid-demo-pkg/pyproject.toml defid-demo-pkg/src/defid_demo/__init__.py defid-demo-pkg/tests/test_smoke.py pyproject.toml uv.lock
git commit -m "feat(defid-demo): scaffold live-demo package in workspace"
```

---

## Task 2: `adapter.py` — browser events → `defid` session dicts

Browser sends one rep as JSON: a list of pointer samples and a list of key events. The adapter converts to `defid`-shaped `touch` (`{"t","x","y"}`, seconds) and `key` (`{"t","phase"}`) lists. Key *content* is never sent — only `code` (physical key id, e.g. `"KeyA"`) for down/up pairing, which the adapter consumes and discards. The i-th emitted `down` must pair with the i-th emitted `up` (that is how `defid.features` reads them): emit one `(down,up)` pair per completed keystroke in keyup-completion order.

**Files:**
- Create: `defid-demo-pkg/src/defid_demo/adapter.py`
- Create: `defid-demo-pkg/tests/test_adapter.py`

- [ ] **Step 1: Write the failing test**

`defid-demo-pkg/tests/test_adapter.py`:
```python
import pytest

from defid_demo.adapter import RepPayload, payload_to_session


def test_pointer_samples_become_seconds_and_xy():
    p = RepPayload(
        pointer=[
            {"x": 10.0, "y": 20.0, "ts": 1000.0},
            {"x": 12.0, "y": 25.0, "ts": 1016.0},
            {"x": 15.0, "y": 30.0, "ts": 1033.0},
        ],
        keys=[],
    )
    touch, key = payload_to_session(p)
    assert touch[0] == {"t": 0.0, "x": 10.0, "y": 20.0}
    assert touch[1]["t"] == pytest.approx(0.016)
    assert touch[2]["t"] == pytest.approx(0.033)
    assert key == []


def test_keys_pair_down_up_by_code_in_completion_order():
    p = RepPayload(
        pointer=[],
        keys=[
            {"code": "KeyA", "phase": "down", "ts": 2000.0},
            {"code": "KeyB", "phase": "down", "ts": 2050.0},
            {"code": "KeyA", "phase": "up", "ts": 2090.0},
            {"code": "KeyB", "phase": "up", "ts": 2160.0},
        ],
    )
    touch, key = payload_to_session(p)
    # Completion order = by keyup ts: KeyA completes first, then KeyB.
    assert [(k["phase"], round(k["t"], 3)) for k in key] == [
        ("down", 0.0), ("up", 0.09), ("down", 0.05), ("up", 0.16),
    ]
    # i-th down pairs with i-th up: dwellA = 0.09-0.0, dwellB = 0.16-0.05
    downs = [k["t"] for k in key if k["phase"] == "down"]
    ups = [k["t"] for k in key if k["phase"] == "up"]
    assert round(ups[0] - downs[0], 3) == 0.09
    assert round(ups[1] - downs[1], 3) == 0.11


def test_unmatched_key_events_are_dropped():
    p = RepPayload(
        pointer=[],
        keys=[
            {"code": "KeyA", "phase": "up", "ts": 10.0},      # up with no down
            {"code": "KeyB", "phase": "down", "ts": 20.0},    # down with no up
        ],
    )
    _, key = payload_to_session(p)
    assert key == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_adapter.py -q 2>&1 | tail -5`
Expected: FAIL `ModuleNotFoundError: No module named 'defid_demo.adapter'`.

- [ ] **Step 3: Implement the adapter**

`defid-demo-pkg/src/defid_demo/adapter.py`:
```python
"""Browser rep payload -> defid-shaped (touch, key) event lists.

Privacy: only physical key `code` is used, solely to pair down/up; it is
consumed here and never stored or forwarded. No character content exists
in the payload by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RepPayload:
    pointer: list[dict[str, Any]]  # [{"x","y","ts"} ...]  ts in ms
    keys: list[dict[str, Any]]     # [{"code","phase","ts"} ...] phase down|up


def payload_to_session(p: RepPayload) -> tuple[list[dict], list[dict]]:
    touch: list[dict] = []
    if p.pointer:
        t0 = p.pointer[0]["ts"]
        for s in p.pointer:
            touch.append(
                {
                    "t": (s["ts"] - t0) / 1000.0,
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                }
            )

    # Pair down/up per physical key. A keystroke completes at its keyup;
    # order completed keystrokes by keyup ts so the i-th down aligns with
    # the i-th up after splitting by phase.
    open_down: dict[str, float] = {}
    completed: list[tuple[float, float]] = []  # (down_ts, up_ts)
    for ev in p.keys:
        code = ev["code"]
        if ev["phase"] == "down":
            open_down[code] = ev["ts"]
        elif ev["phase"] == "up" and code in open_down:
            completed.append((open_down.pop(code), ev["ts"]))
    completed.sort(key=lambda du: du[1])

    key: list[dict] = []
    if completed:
        k0 = completed[0][0]
        for down_ts, up_ts in completed:
            key.append({"t": (down_ts - k0) / 1000.0, "phase": "down"})
            key.append({"t": (up_ts - k0) / 1000.0, "phase": "up"})

    return touch, key
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_adapter.py -q 2>&1 | tail -5`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add defid-demo-pkg/src/defid_demo/adapter.py defid-demo-pkg/tests/test_adapter.py
git commit -m "feat(defid-demo): browser-event -> session adapter (timing only)"
```

---

## Task 3: `qc.py` — degenerate-rep rejection

Reuse `defid.qc.QCResult`. A demo rep has no motion, so we cannot reuse `defid.qc.check_session` (it requires motion ≥3). `check_rep` validates a captured rep is a usable swipe + typed passphrase.

**Files:**
- Create: `defid-demo-pkg/src/defid_demo/qc.py`
- Create: `defid-demo-pkg/tests/test_qc.py`

- [ ] **Step 1: Write the failing test**

`defid-demo-pkg/tests/test_qc.py`:
```python
from defid_demo.qc import check_rep


def _swipe(n):
    return [{"t": i * 0.02, "x": float(i), "y": float(i)} for i in range(n)]


def _keys(n):
    out = []
    for i in range(n):
        out.append({"t": i * 0.2, "phase": "down"})
        out.append({"t": i * 0.2 + 0.08, "phase": "up"})
    return out


def test_good_rep_passes():
    r = check_rep(_swipe(20), _keys(6))
    assert r.ok and r.reason is None


def test_too_few_touch_points_is_a_tap_not_a_swipe():
    r = check_rep(_swipe(2), _keys(6))
    assert not r.ok and "touch" in r.reason


def test_no_typing_rejected():
    r = check_rep(_swipe(20), [])
    assert not r.ok and "key" in r.reason


def test_non_monotonic_touch_rejected():
    bad = _swipe(10)
    bad[5]["t"] = bad[4]["t"] - 0.01
    r = check_rep(bad, _keys(6))
    assert not r.ok and "monotonic" in r.reason
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_qc.py -q 2>&1 | tail -5`
Expected: FAIL `ModuleNotFoundError: No module named 'defid_demo.qc'`.

- [ ] **Step 3: Implement QC**

`defid-demo-pkg/src/defid_demo/qc.py`:
```python
"""Per-rep capture QC. Reuses defid.qc.QCResult (do not redefine)."""

from __future__ import annotations

import math

from defid.qc import QCResult

_MIN_TOUCH = 8   # a real swipe; below this it's a tap/accident
_MIN_KEYS = 2    # at least two completed keystrokes for dwell+flight


def check_rep(touch: list[dict], key: list[dict]) -> QCResult:
    if len(touch) < _MIN_TOUCH:
        return QCResult(False, f"too few touch points (<{_MIN_TOUCH}) — swipe, don't tap")
    downs = [k for k in key if k["phase"] == "down"]
    if len(downs) < _MIN_KEYS:
        return QCResult(False, f"too few keystrokes (<{_MIN_KEYS}) — type the passphrase")
    for i in range(1, len(touch)):
        if touch[i]["t"] - touch[i - 1]["t"] <= 0:
            return QCResult(False, "non-monotonic touch timestamps")
    for ev in touch:
        for k in ("t", "x", "y"):
            if not isinstance(ev[k], (int, float)) or not math.isfinite(ev[k]):
                return QCResult(False, f"touch.{k} not finite")
    return QCResult(True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_qc.py -q 2>&1 | tail -5`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add defid-demo-pkg/src/defid_demo/qc.py defid-demo-pkg/tests/test_qc.py
git commit -m "feat(defid-demo): per-rep capture QC (reuses defid QCResult)"
```

---

## Task 4: `windows.py` — windowed extraction via the real `defid.features`

A rep's touch stream is sliced into `K=5` overlapping sub-windows (50% overlap). For each sub-window we build a `BehavioralSession(touch=subwindow, key=rep_key, motion=[])` and call the **actual** `defid.features.extract_features`, then keep indices 0..8 (the touch+keystroke subset). This is the literal validated extractor — central to the design's honesty claim.

**Files:**
- Create: `defid-demo-pkg/src/defid_demo/windows.py`
- Create: `defid-demo-pkg/tests/test_windows.py`

- [ ] **Step 1: Write the failing test**

`defid-demo-pkg/tests/test_windows.py`:
```python
import numpy as np

from defid.features import FEATURE_NAMES, extract_features
from defid.session import BehavioralSession
from defid_demo.windows import FEATURE_SUBSET, SUBSET_IDX, extract_windows


def _swipe(n):
    return [{"t": i * 0.02, "x": float(i) * 1.3, "y": float(i) * 0.7}
            for i in range(n)]


def _keys(n):
    out = []
    for i in range(n):
        out.append({"t": i * 0.25, "phase": "down"})
        out.append({"t": i * 0.25 + 0.07, "phase": "up"})
    return out


def test_subset_is_first_nine_touch_keystroke_features():
    assert SUBSET_IDX == list(range(9))
    assert FEATURE_SUBSET == FEATURE_NAMES[:9]
    assert "key_paste_ratio" not in FEATURE_SUBSET
    assert "tremor_std" not in FEATURE_SUBSET


def test_window_matches_sliced_extract_features():
    touch, key = _swipe(40), _keys(8)
    W = extract_windows(touch, key, k=5, overlap=0.5)
    assert W.shape == (5, 9)
    assert np.all(np.isfinite(W))

    # First window must equal extract_features([:firstslice]) sliced to 0..8.
    step = int(len(touch) * (1 - 0.5) / (5 - 1 + 1e-9))  # see impl
    win_len = len(touch) - step * (5 - 1)
    s = BehavioralSession(
        session_id="d", label="genuine", subject_id="d",
        touch=touch[0:win_len], key=key, motion=[],
        ontology_version="d", generator_version="d", seed=0,
    )
    expected = extract_features(s)[:9]
    assert np.allclose(W[0], expected)


def test_two_distinct_motor_profiles_separate_in_feature_space():
    rng = np.random.default_rng(0)

    def rep(speed, dwell):
        t = 0.0
        touch = []
        for i in range(40):
            t += 0.02
            touch.append({"t": t, "x": i * speed + rng.normal(0, 0.3),
                          "y": i * 0.5 + rng.normal(0, 0.3)})
        key = []
        tk = 0.0
        for _ in range(8):
            key.append({"t": tk, "phase": "down"})
            tk += dwell
            key.append({"t": tk, "phase": "up"})
            tk += 0.15
        return touch, key

    a = np.vstack([extract_windows(*rep(1.0, 0.08)) for _ in range(4)])
    b = np.vstack([extract_windows(*rep(3.0, 0.20)) for _ in range(4)])
    assert np.linalg.norm(a.mean(0) - b.mean(0)) > 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_windows.py -q 2>&1 | tail -5`
Expected: FAIL `ModuleNotFoundError: No module named 'defid_demo.windows'`.

- [ ] **Step 3: Implement windowing**

`defid-demo-pkg/src/defid_demo/windows.py`:
```python
"""Windowed feature extraction. Calls the real defid.features.extract_features
and keeps the touch+keystroke subset (indices 0..8). The demo therefore runs
the identical validated extractor formulas."""

from __future__ import annotations

import numpy as np

from defid.features import FEATURE_NAMES, extract_features
from defid.session import BehavioralSession

SUBSET_IDX = list(range(9))
FEATURE_SUBSET = FEATURE_NAMES[:9]


def _session(touch: list[dict], key: list[dict]) -> BehavioralSession:
    return BehavioralSession(
        session_id="demo", label="genuine", subject_id="demo",
        touch=touch, key=key, motion=[],
        ontology_version="demo", generator_version="demo", seed=0,
    )


def extract_windows(
    touch: list[dict], key: list[dict], k: int = 5, overlap: float = 0.5
) -> np.ndarray:
    """k overlapping touch sub-windows; each row = subset features for that
    touch slice paired with the rep's full keystroke stream."""
    n = len(touch)
    if n < k:
        # Too short to window: one row from the whole rep, repeated k times
        # so downstream shapes are stable.
        v = extract_features(_session(touch, key))[SUBSET_IDX]
        return np.tile(v, (k, 1))
    step = int(n * (1 - overlap) / (k - 1 + 1e-9))
    step = max(step, 1)
    win_len = n - step * (k - 1)
    rows = []
    for i in range(k):
        lo = i * step
        sub = touch[lo : lo + win_len]
        rows.append(extract_features(_session(sub, key))[SUBSET_IDX])
    return np.asarray(rows, dtype=np.float64)
```

Note: the test's `step`/`win_len` mirror this exactly so `W[0]` equals the sliced `extract_features` of `touch[0:win_len]`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_windows.py -q 2>&1 | tail -5`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add defid-demo-pkg/src/defid_demo/windows.py defid-demo-pkg/tests/test_windows.py
git commit -m "feat(defid-demo): windowed extraction via real defid.features (subset 0..8)"
```

---

## Task 5: `demo_auth.py` — `DemoAuth(MahalanobisAuth)`

Subclass `MahalanobisAuth`: override `fit` to drop constant columns then apply shrinkage `cov_shrunk = (1-α)·cov + α·diag(diag(cov))` before the inherited `reg` diagonal loading; override `score` to project to the kept columns; add `calibrate` (threshold from held-out genuine distances) and `classify` (per-attempt verdict).

**Files:**
- Create: `defid-demo-pkg/src/defid_demo/demo_auth.py`
- Create: `defid-demo-pkg/tests/test_demo_auth.py`

- [ ] **Step 1: Write the failing test**

`defid-demo-pkg/tests/test_demo_auth.py`:
```python
import numpy as np

from defid.models import MahalanobisAuth
from defid_demo.demo_auth import DemoAuth


def test_is_a_mahalanobis_auth_subclass():
    assert issubclass(DemoAuth, MahalanobisAuth)


def test_constant_columns_are_dropped_and_recorded():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(30, 5))
    X[:, 2] = 7.0  # constant column
    a = DemoAuth(alpha=0.1).fit(X)
    assert a.kept_idx == [0, 1, 3, 4]
    assert a.dropped_names == []  # names set via fit_named; see below
    # score still works and is finite
    d = a.score(rng.normal(0, 1, size=(4, 5)))
    assert d.shape == (4,) and np.all(np.isfinite(d))


def test_fit_named_records_dropped_feature_names():
    X = np.ones((10, 3))
    X[:, 0] = np.arange(10)
    X[:, 1] = np.linspace(0, 1, 10)
    a = DemoAuth(alpha=0.1).fit_named(X, ["fa", "fb", "fc"])
    assert a.dropped_names == ["fc"]


def test_determinism_same_input_same_scores():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(40, 6))
    q = rng.normal(0, 1, size=(8, 6))
    s1 = DemoAuth(alpha=0.15).fit(X).score(q)
    s2 = DemoAuth(alpha=0.15).fit(X).score(q)
    assert np.array_equal(s1, s2)


def test_calibrate_then_classify_separates_genuine_from_outlier():
    rng = np.random.default_rng(3)
    enroll = rng.normal(0.0, 1.0, size=(40, 6))
    holdout = rng.normal(0.0, 1.0, size=(10, 6))
    a = DemoAuth(alpha=0.1).fit(enroll)
    a.calibrate(holdout)
    assert a.threshold is not None and a.threshold > 0

    genuine_attempt = rng.normal(0.0, 1.0, size=(5, 6))
    impostor_attempt = rng.normal(8.0, 1.0, size=(5, 6))
    g = a.classify(genuine_attempt)
    im = a.classify(impostor_attempt)
    assert g["verdict"] == "ACCEPT"
    assert im["verdict"] == "REJECT"
    assert im["frac_above"] >= 0.5
    assert len(g["distances"]) == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_demo_auth.py -q 2>&1 | tail -5`
Expected: FAIL `ModuleNotFoundError: No module named 'defid_demo.demo_auth'`.

- [ ] **Step 3: Implement DemoAuth**

`defid-demo-pkg/src/defid_demo/demo_auth.py`:
```python
"""DemoAuth: a MahalanobisAuth subclass adding constant-column drop,
covariance shrinkage, empirical threshold calibration, and per-attempt
verdict aggregation. defid.models is not modified."""

from __future__ import annotations

import numpy as np

from defid.models import MahalanobisAuth


class DemoAuth(MahalanobisAuth):
    def __init__(self, reg: float = 1e-3, alpha: float = 0.10) -> None:
        super().__init__(reg=reg)
        self.alpha = alpha
        self.kept_idx: list[int] = []
        self.dropped_names: list[str] = []
        self.threshold: float | None = None

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x[:, self.kept_idx]

    def fit(self, genuine: np.ndarray) -> "DemoAuth":
        x = np.asarray(genuine, dtype=np.float64)
        var = x.var(axis=0)
        self.kept_idx = [i for i in range(x.shape[1]) if var[i] > 1e-12]
        xk = x[:, self.kept_idx]
        self._mean = xk.mean(axis=0)
        cov = np.atleast_2d(np.cov(xk, rowvar=False))
        cov = (1.0 - self.alpha) * cov + self.alpha * np.diag(np.diag(cov))
        cov = cov + np.eye(cov.shape[0]) * self.reg
        self._inv_cov = np.linalg.pinv(cov)
        return self

    def fit_named(self, genuine: np.ndarray, names: list[str]) -> "DemoAuth":
        self.fit(genuine)
        kept = set(self.kept_idx)
        self.dropped_names = [n for i, n in enumerate(names) if i not in kept]
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        d = self._project(x) - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self._inv_cov, d))

    def calibrate(self, holdout_genuine: np.ndarray) -> float:
        d = self.score(holdout_genuine)
        if d.size >= 4:
            self.threshold = float(d.mean() + 3.0 * d.std())
        else:
            self.threshold = float(d.max() * 1.10)
        return self.threshold

    def classify(self, attempt_windows: np.ndarray) -> dict:
        assert self.threshold is not None, "calibrate() first"
        d = self.score(attempt_windows)
        frac = float(np.mean(d > self.threshold))
        return {
            "verdict": "REJECT" if frac >= 0.5 else "ACCEPT",
            "frac_above": frac,
            "distances": [float(v) for v in d],
            "threshold": self.threshold,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_demo_auth.py -q 2>&1 | tail -5`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add defid-demo-pkg/src/defid_demo/demo_auth.py defid-demo-pkg/tests/test_demo_auth.py
git commit -m "feat(defid-demo): DemoAuth (shrinkage + drop + threshold + verdict)"
```

---

## Task 6: `service.py` — `DemoService` orchestrator + headless flow

Pure, in-memory orchestrator. No FastAPI here so it is fully unit-testable. The headless flow test is the integration gate: enroll → calibrate → genuine confirm ACCEPTs → impostor REJECTs.

**Files:**
- Create: `defid-demo-pkg/src/defid_demo/service.py`
- Create: `defid-demo-pkg/tests/test_service_flow.py`

- [ ] **Step 1: Write the failing test**

`defid-demo-pkg/tests/test_service_flow.py`:
```python
import numpy as np

from defid_demo.adapter import RepPayload
from defid_demo.service import DemoService


def _rep(rng, speed, dwell, jitter):
    """Synthesize a browser RepPayload for a motor profile."""
    pointer, ts = [], 1000.0
    x = y = 50.0
    for i in range(45):
        x += speed + rng.normal(0, jitter)
        y += speed * 0.4 + rng.normal(0, jitter)
        ts += 16.0 + rng.normal(0, 2.0)
        pointer.append({"x": x, "y": y, "ts": ts})
    keys, kt = [], 3000.0
    for i in range(8):
        code = f"K{i}"
        keys.append({"code": code, "phase": "down", "ts": kt})
        kt += dwell * 1000.0
        keys.append({"code": code, "phase": "up", "ts": kt})
        kt += 180.0 + rng.normal(0, 20.0)
    return RepPayload(pointer=pointer, keys=keys)


def test_full_flow_accepts_enrollee_rejects_impostor():
    svc = DemoService()
    rng = np.random.default_rng(7)

    # Enrollee profile.
    for _ in range(8):
        r = svc.enroll(_rep(rng, speed=2.0, dwell=0.09, jitter=0.4))
        assert r["ok"], r

    cal = svc.calibrate()
    assert cal["threshold"] > 0
    assert cal["kept"] >= 1

    # Genuine confirm (same profile) must ACCEPT.
    g = svc.attempt(_rep(rng, speed=2.0, dwell=0.09, jitter=0.4))
    assert g["verdict"] == "ACCEPT", g

    # The attempt result carries the live feature vector (spec §11).
    assert set(g["feature_values"]) == set(svc.state()["features"])
    assert all(isinstance(v, float) for v in g["feature_values"].values())

    # Three distinct impostor profiles must all REJECT.
    rejects = 0
    for sp, dw in [(6.0, 0.20), (0.6, 0.04), (4.0, 0.30)]:
        a = svc.attempt(_rep(rng, speed=sp, dwell=dw, jitter=1.5))
        if a["verdict"] == "REJECT":
            rejects += 1
    assert rejects == 3


def test_bad_rep_is_rejected_with_reason_and_not_enrolled():
    svc = DemoService()
    bad = RepPayload(pointer=[{"x": 1, "y": 1, "ts": 0.0}], keys=[])
    r = svc.enroll(bad)
    assert not r["ok"] and r["reason"]
    assert svc.state()["enroll_reps"] == 0


def test_calibrate_requires_minimum_reps():
    svc = DemoService()
    rng = np.random.default_rng(1)
    for _ in range(2):
        svc.enroll(_rep(rng, 2.0, 0.09, 0.4))
    out = svc.calibrate()
    assert out["ok"] is False and "more" in out["reason"]


def test_reset_clears_state():
    svc = DemoService()
    rng = np.random.default_rng(1)
    svc.enroll(_rep(rng, 2.0, 0.09, 0.4))
    svc.reset()
    assert svc.state()["enroll_reps"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_service_flow.py -q 2>&1 | tail -5`
Expected: FAIL `ModuleNotFoundError: No module named 'defid_demo.service'`.

- [ ] **Step 3: Implement the service**

`defid-demo-pkg/src/defid_demo/service.py`:
```python
"""In-memory demo orchestrator. Pure (no web layer) so it is fully
unit-testable. Holds out the last HOLDOUT_REPS enrollment reps for
threshold calibration."""

from __future__ import annotations

import numpy as np

from defid_demo.adapter import RepPayload, payload_to_session
from defid_demo.demo_auth import DemoAuth
from defid_demo.qc import check_rep
from defid_demo.windows import FEATURE_SUBSET, extract_windows

MIN_FIT_REPS = 4      # reps used to fit (after holdout)
HOLDOUT_REPS = 2      # last enrollment reps reserved for calibration


class DemoService:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._reps: list[np.ndarray] = []  # one (k,9) array per enroll rep
        self._auth: DemoAuth | None = None
        self._attempts: list[dict] = []

    def _windows(self, payload: RepPayload):
        touch, key = payload_to_session(payload)
        qc = check_rep(touch, key)
        if not qc.ok:
            return None, qc.reason
        return extract_windows(touch, key), None

    def enroll(self, payload: RepPayload) -> dict:
        W, reason = self._windows(payload)
        if W is None:
            return {"ok": False, "reason": reason}
        self._reps.append(W)
        return {"ok": True, "enroll_reps": len(self._reps)}

    def calibrate(self) -> dict:
        if len(self._reps) < MIN_FIT_REPS + HOLDOUT_REPS:
            need = MIN_FIT_REPS + HOLDOUT_REPS - len(self._reps)
            return {"ok": False,
                    "reason": f"need {need} more enrollment reps"}
        fit_reps = self._reps[:-HOLDOUT_REPS]
        hold_reps = self._reps[-HOLDOUT_REPS:]
        Xfit = np.vstack(fit_reps)
        Xhold = np.vstack(hold_reps)
        self._auth = DemoAuth().fit_named(Xfit, list(FEATURE_SUBSET))
        thr = self._auth.calibrate(Xhold)
        return {
            "ok": True,
            "threshold": thr,
            "kept": len(self._auth.kept_idx),
            "dropped": self._auth.dropped_names,
        }

    def attempt(self, payload: RepPayload) -> dict:
        if self._auth is None or self._auth.threshold is None:
            return {"ok": False, "reason": "not calibrated"}
        W, reason = self._windows(payload)
        if W is None:
            return {"ok": False, "reason": reason}
        result = self._auth.classify(W)
        result["ok"] = True
        feat_mean = W.mean(axis=0)
        result["feature_values"] = {
            n: float(v) for n, v in zip(FEATURE_SUBSET, feat_mean)
        }
        self._attempts.append(result)
        return result

    def state(self) -> dict:
        return {
            "enroll_reps": len(self._reps),
            "calibrated": self._auth is not None
            and self._auth.threshold is not None,
            "threshold": None if self._auth is None else self._auth.threshold,
            "attempts": self._attempts,
            "features": list(FEATURE_SUBSET),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_service_flow.py -q 2>&1 | tail -8`
Expected: 4 passed. (If the separation assertion is flaky, the profiles are deliberately far apart; investigate rather than loosen — a real failure here means the pipeline is broken.)

- [ ] **Step 5: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: all pass (102 prior + Tasks 1–6 new tests), 4 warnings.

- [ ] **Step 6: Commit**

```bash
git add defid-demo-pkg/src/defid_demo/service.py defid-demo-pkg/tests/test_service_flow.py
git commit -m "feat(defid-demo): DemoService orchestrator + headless flow gate"
```

---

## Task 7: `app.py` FastAPI shell + static web clients

Thin HTTP shell over `DemoService` (one global instance) + static capture/Dashboard client and a spectator view. The web client is intentionally minimal but complete.

**Files:**
- Create: `defid-demo-pkg/src/defid_demo/app.py`
- Create: `defid-demo-pkg/src/defid_demo/web/index.html`
- Create: `defid-demo-pkg/src/defid_demo/web/spectator.html`
- Create: `defid-demo-pkg/tests/test_app_smoke.py`

- [ ] **Step 1: Write the failing test**

`defid-demo-pkg/tests/test_app_smoke.py`:
```python
from fastapi.testclient import TestClient

from defid_demo.app import app


def _rep(speed, dwell):
    ptr, ts, x, y = [], 1000.0, 50.0, 50.0
    for i in range(45):
        x += speed
        y += speed * 0.4
        ts += 16.0
        ptr.append({"x": x, "y": y, "ts": ts})
    keys, kt = [], 3000.0
    for i in range(8):
        keys.append({"code": f"K{i}", "phase": "down", "ts": kt})
        kt += dwell * 1000.0
        keys.append({"code": f"K{i}", "phase": "up", "ts": kt})
        kt += 180.0
    return {"pointer": ptr, "keys": keys}


def test_index_and_spectator_served():
    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert "DefinitiveID" in c.get("/").text
    assert c.get("/spectator").status_code == 200


def test_reset_enroll_calibrate_attempt_roundtrip():
    c = TestClient(app)
    assert c.post("/api/reset").status_code == 200
    for _ in range(8):
        r = c.post("/api/enroll", json=_rep(2.0, 0.09))
        assert r.status_code == 200 and r.json()["ok"]
    cal = c.post("/api/calibrate").json()
    assert cal["ok"] and cal["threshold"] > 0
    a = c.post("/api/attempt", json=_rep(2.0, 0.09)).json()
    assert a["verdict"] in ("ACCEPT", "REJECT")
    assert "distances" in a
    assert len(a["feature_values"]) == 9
    st = c.get("/api/state").json()
    assert st["enroll_reps"] == 8 and st["calibrated"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_app_smoke.py -q 2>&1 | tail -5`
Expected: FAIL `ModuleNotFoundError: No module named 'defid_demo.app'`.

- [ ] **Step 3: Implement the FastAPI app**

`defid-demo-pkg/src/defid_demo/app.py`:
```python
"""Thin FastAPI shell over DemoService. One in-memory instance; trusted
LAN demo only — no auth, no persistence (by design, see spec §10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from defid_demo.adapter import RepPayload
from defid_demo.service import DemoService

app = FastAPI(title="DefinitiveID Live Demo")
_svc = DemoService()
_WEB = Path(__file__).parent / "web"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.get("/spectator")
def spectator() -> FileResponse:
    return FileResponse(_WEB / "spectator.html")


@app.post("/api/reset")
def reset() -> dict:
    _svc.reset()
    return {"ok": True}


@app.post("/api/enroll")
def enroll(payload: dict) -> dict:
    return _svc.enroll(RepPayload(pointer=payload.get("pointer", []),
                                  keys=payload.get("keys", [])))


@app.post("/api/calibrate")
def calibrate() -> dict:
    return _svc.calibrate()


@app.post("/api/attempt")
def attempt(payload: dict) -> dict:
    return _svc.attempt(RepPayload(pointer=payload.get("pointer", []),
                                   keys=payload.get("keys", [])))


@app.get("/api/state")
def state() -> dict:
    return _svc.state()
```

- [ ] **Step 4: Write the capture client**

`defid-demo-pkg/src/defid_demo/web/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>DefinitiveID — Live Demo</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#0f1419;color:#e6e6e6}
  header{padding:12px 16px;background:#161b22;font-weight:700}
  .wrap{display:flex;flex-wrap:wrap}
  .left,.right{flex:1;min-width:300px;padding:16px}
  #pad{width:100%;height:240px;background:#1c2330;border:1px solid #30363d;border-radius:8px;touch-action:none}
  input,button{font-size:16px;padding:10px;margin:6px 0;width:100%;box-sizing:border-box}
  button{background:#2f81f7;color:#fff;border:0;border-radius:6px;font-weight:700}
  button.alt{background:#30363d}
  #verdict{font-size:34px;font-weight:800;text-align:center;margin:12px 0}
  .ACCEPT{color:#3fb950}.REJECT{color:#f85149}
  #gauge{height:18px;background:#30363d;border-radius:9px;overflow:hidden}
  #gfill{height:100%;width:0;background:#f85149}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td{border-bottom:1px solid #30363d;padding:3px 6px}
  #scatter{height:160px;border:1px solid #30363d;border-radius:8px;position:relative;background:#1c2330}
  .pt{position:absolute;width:8px;height:8px;border-radius:50%;transform:translate(-50%,-50%)}
  .g{background:#3fb950}.i{background:#f85149}
  .thr{position:absolute;left:0;right:0;border-top:2px dashed #d29922}
  small{color:#8b949e}
</style>
</head>
<body>
<header>DefinitiveID — Live Behavioral Demo</header>
<div class="wrap">
  <div class="left">
    <p><small id="phase">Phase: enroll. Swipe on the pad, then type the passphrase, then tap the action.</small></p>
    <canvas id="pad"></canvas>
    <input id="pass" placeholder="type: definitive identity" autocomplete="off" autocapitalize="off" spellcheck="false">
    <button id="go">Submit enrollment rep (<span id="cnt">0</span>/8)</button>
    <button id="cal" class="alt">Calibrate</button>
    <button id="att" class="alt">Submit attempt</button>
    <button id="rst" class="alt">Reset demo</button>
    <div id="verdict"></div>
    <div id="gauge"><div id="gfill"></div></div>
    <small id="dist"></small>
  </div>
  <div class="right">
    <p><small>Attempt history — genuine below the line, impostor above.</small></p>
    <div id="scatter"><div class="thr" id="thrline" style="top:50%"></div></div>
    <p><small>Live feature vector (most recent attempt)</small></p>
    <table id="ftab"></table>
    <small id="dropped"></small>
  </div>
</div>
<script>
const pad=document.getElementById('pad'),ctx=pad.getContext('2d');
function fit(){pad.width=pad.clientWidth;pad.height=pad.clientHeight;}
window.addEventListener('resize',fit);fit();
let pointer=[],keys=[],drawing=false;
function clearPad(){pointer=[];ctx.clearRect(0,0,pad.width,pad.height);}
pad.addEventListener('pointerdown',e=>{drawing=true;ctx.beginPath();ctx.moveTo(e.offsetX,e.offsetY);});
pad.addEventListener('pointermove',e=>{
  if(!drawing)return;
  pointer.push({x:e.clientX,y:e.clientY,ts:e.timeStamp});
  ctx.lineTo(e.offsetX,e.offsetY);ctx.strokeStyle='#2f81f7';ctx.lineWidth=3;ctx.stroke();
});
window.addEventListener('pointerup',()=>{drawing=false;});
const pf=document.getElementById('pass');
pf.addEventListener('keydown',e=>keys.push({code:e.code,phase:'down',ts:e.timeStamp}));
pf.addEventListener('keyup',e=>keys.push({code:e.code,phase:'up',ts:e.timeStamp}));
function rep(){const r={pointer:pointer.slice(),keys:keys.slice()};clearPad();keys=[];pf.value='';return r;}
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):null});return r.json();}
let cnt=0;
document.getElementById('go').onclick=async()=>{
  const o=await post('/api/enroll',rep());
  if(!o.ok){alert(o.reason);return;}
  cnt=o.enroll_reps;document.getElementById('cnt').textContent=cnt;
};
document.getElementById('cal').onclick=async()=>{
  const o=await post('/api/calibrate');
  if(!o.ok){alert(o.reason);return;}
  document.getElementById('phase').textContent='Phase: attempt. Threshold set ('+o.threshold.toFixed(2)+').';
  document.getElementById('dropped').textContent=o.dropped.length?('dropped constant features: '+o.dropped.join(', ')):'';
};
function render(o){
  const v=document.getElementById('verdict');
  v.textContent=o.verdict;v.className=o.verdict;
  const mx=o.threshold*2.2,avg=o.distances.reduce((a,b)=>a+b,0)/o.distances.length;
  document.getElementById('gfill').style.width=Math.min(100,avg/mx*100)+'%';
  document.getElementById('dist').textContent='mean dist '+avg.toFixed(2)+'  vs  thr '+o.threshold.toFixed(2);
  const sc=document.getElementById('scatter');
  document.getElementById('thrline').style.top='50%';
  o.distances.forEach(d=>{
    const p=document.createElement('div');
    p.className='pt '+(o.verdict==='ACCEPT'?'g':'i');
    p.style.left=(10+Math.random()*80)+'%';
    p.style.top=Math.max(2,Math.min(98,50-(d-o.threshold)/(o.threshold||1)*25))+'%';
    sc.appendChild(p);
  });
  const t=document.getElementById('ftab');t.innerHTML='';
  Object.entries(o.feature_values).forEach(([n,val])=>{
    t.innerHTML+='<tr><td>'+n+'</td><td>'+val.toFixed(4)+'</td></tr>';
  });
}
document.getElementById('att').onclick=async()=>{
  const o=await post('/api/attempt',rep());
  if(!o.ok){alert(o.reason);return;}
  render(o);
};
document.getElementById('rst').onclick=async()=>{await post('/api/reset');cnt=0;document.getElementById('cnt').textContent='0';location.reload();};
</script>
</body>
</html>
```

- [ ] **Step 5: Write the spectator view**

`defid-demo-pkg/src/defid_demo/web/spectator.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>DefinitiveID — Spectator</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0f1419;color:#e6e6e6;margin:0;padding:24px}
  h1{font-size:20px}#v{font-size:64px;font-weight:800;text-align:center;margin:24px}
  .ACCEPT{color:#3fb950}.REJECT{color:#f85149}
  table{width:100%;border-collapse:collapse}td{border-bottom:1px solid #30363d;padding:6px}
  small{color:#8b949e}
</style>
</head>
<body>
<h1>DefinitiveID — live (read-only)</h1>
<div id="v">—</div>
<p style="text-align:center"><small id="meta"></small></p>
<table id="t"></table>
<script>
async function tick(){
  const s=await (await fetch('/api/state')).json();
  const last=s.attempts[s.attempts.length-1];
  const v=document.getElementById('v');
  if(last){v.textContent=last.verdict;v.className=last.verdict;
    document.getElementById('meta').textContent='threshold '+(s.threshold||0).toFixed(2)+' · attempts '+s.attempts.length;}
  const t=document.getElementById('t');t.innerHTML='';
  s.attempts.slice(-10).forEach((a,i)=>{t.innerHTML+='<tr><td>#'+(s.attempts.length-10+i+1)+'</td><td class="'+a.verdict+'">'+a.verdict+'</td><td>frac_above '+a.frac_above.toFixed(2)+'</td></tr>';});
}
setInterval(tick,1000);tick();
</script>
</body>
</html>
```

- [ ] **Step 6: Run the smoke test to verify it passes**

Run: `.venv/bin/python -m pytest defid-demo-pkg/tests/test_app_smoke.py -q 2>&1 | tail -5`
Expected: 2 passed.

- [ ] **Step 7: Full suite green**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: all pass, 4 warnings (the known pre-existing numpy corrcoef warnings from `defid`).

- [ ] **Step 8: Commit**

```bash
git add defid-demo-pkg/src/defid_demo/app.py defid-demo-pkg/src/defid_demo/web/index.html defid-demo-pkg/src/defid_demo/web/spectator.html defid-demo-pkg/tests/test_app_smoke.py
git commit -m "feat(defid-demo): FastAPI shell + capture/Dashboard + spectator views"
```

---

## Final verification (manual — the on-device dry run)

- [ ] **Step 1: Full suite green + defid untouched**

```bash
cd /Users/stuartwells/test
.venv/bin/python -m pytest -q 2>&1 | tail -3
git diff --stat 2c8bb8d -- defid-pkg | cat   # expect: NO output (defid unmodified)
```

- [ ] **Step 2: Run the service and dry-run the protocol**

```bash
.venv/bin/python -m uvicorn defid_demo.app:app --host 0.0.0.0 --port 8000
```

On a phone on the same Wi-Fi, open `http://<laptop-LAN-IP>:8000/`. Follow the spec §7 protocol: 8 enrollment reps (swipe + type the passphrase) → Calibrate → one genuine attempt (expect **ACCEPT**) → at least 3 different people each do one attempt (expect **REJECT**, distances visibly above the threshold line). Open `http://<laptop-LAN-IP>:8000/spectator` on the projected screen.

Record: enrollment time, genuine-confirm verdict, the 3+ impostor verdicts, and any dropped constant features shown after Calibrate. Interpret per the design's section-2 honesty framing: this proves the signal is real and live-discriminative; the synthetic→real gap remains the separately-quantified offline cross-domain number.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose A+B (visceral + methodology) | Tasks 6, 7 (verdict + scatter/feature panel); Final dry run |
| §2 honesty claim; `defid` unmodified | Task 4 (real `extract_features`), Task 5 (`MahalanobisAuth` subclass); Final Step 1 diff check |
| §3 architecture (mobile web + local FastAPI reusing defid) | Tasks 6, 7 |
| §4 components (adapter/windows/demo_auth/qc/service/app/web) | Tasks 2,3,4,5,6,7 |
| §4.1 feature subset (9; exclusions) | Task 4 (`SUBSET_IDX`, test asserts exclusions) |
| §4.2 windowing (K=5, 50% overlap) | Task 4 |
| §4.3 shrinkage α=0.10, threshold mean+3σ / max×1.1, verdict frac≥0.5 | Task 5 |
| §5 data flow | Task 6 (`DemoService`) |
| §6 Dashboard layout | Task 7 (`index.html`) |
| §7 choreography (pre-flight/enroll/calibrate/genuine/impostor) | Final dry run; Task 6 enforces holdout |
| §8 error handling (degenerate, conditioning, LAN, PointerEvents) | Task 3 (QC), Task 6 (calibrate guard), Task 7 (pointer client) |
| §9 testing (unit + headless integration + manual) | Tasks 2–7 unit; Task 6 flow; Final manual |
| §10 non-goals (no DB/auth/motion/native) | Honored: in-memory `DemoService`, no motion path, web client |
| §11 success criteria | Final verification |

**Placeholder scan:** No "TBD/TODO/similar to Task N". Every code step contains complete code. The web client is complete (minimal but functional).

**Type consistency:** `RepPayload(pointer,keys)` consistent across Tasks 2/6/7. `payload_to_session -> (touch,key)` used by `windows`/`service`. `extract_windows(touch,key,k=5,overlap=0.5)` signature consistent Tasks 4/6. `DemoAuth.fit/fit_named/score/calibrate/classify` consistent Tasks 5/6. `DemoService.enroll/calibrate/attempt/reset/state` consistent Tasks 6/7. `QCResult` imported from `defid.qc` (not redefined). `FEATURE_SUBSET`/`SUBSET_IDX` consistent Tasks 4/6.
