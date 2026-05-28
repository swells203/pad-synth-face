# PAD DFDC-Grounded Bonafide Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a DFDC face-frame ingester that produces a DigiFace-shaped identity-per-directory layout the existing pipeline reads unchanged — harness now, the actual DFDC ingest waits on user obtaining the dataset.

**Architecture:** A pure-function `extract_dfdc_bonafide(src, out, license, source_url, res, frames_per_video, crop_margin, max_videos, detector)` in `pad-synth-face/src/pad_synth_face/dfdc.py`. It walks DFDC source trees (videos + per-chunk `metadata.json`), filters to `label == "REAL"`, extracts `frames_per_video` evenly-spaced frames per video via ffmpeg subprocess, calls the injected `detector` (default: lazy-loaded MediaPipe Face Detection) to find a face bbox, square-crops with margin, resizes to `res×res`, and writes `<out>/<video_stem>/<NNN>.png` + a `manifest.jsonl` (per-frame `SampleRecord`) + a `provenance.jsonl` recording a new `DFDCBonafideIngested` event with license capture. The CLI `scripts/prepare_dfdc.py` is a thin shim. Test fixture uses `ffmpeg lavfi` to synthesize tiny mp4s and a stub detector — no PII, no MediaPipe import in the test suite.

**Tech Stack:** Python 3.12+, NumPy, Pillow, ffmpeg (subprocess), MediaPipe Face Detection (new optional `dfdc` extra; lazy import only when no detector is injected). Reuses `ManifestWriter`, `SampleRecord`, `BonafideSource`, `ProvenanceLedger`, `check_image_basic`, and `DigiFaceLoader`.

**Spec:** [`../specs/2026-05-28-pad-dfdc-bonafide-design.md`](../specs/2026-05-28-pad-dfdc-bonafide-design.md)

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `pad-synth-core/src/pad_synth_core/provenance.py` | add `DFDCBonafideIngested` event | Modify |
| `pad-synth-core/tests/test_provenance_dfdc.py` | event serialises | Create |
| `pad-synth-face/src/pad_synth_face/_fixtures.py` | `build_fixture_dfdc(root)` — synthesised mp4s + metadata.json | Modify |
| `pad-synth-face/tests/test_fixture_dfdc.py` | fixture layout valid | Create |
| `pad-synth-face/src/pad_synth_face/dfdc.py` | `extract_dfdc_bonafide` + helpers + lazy MediaPipe wrapper | Create |
| `pad-synth-face/tests/test_dfdc_ingest.py` | layout, manifest, provenance, idempotency, detection-skip, DigiFaceLoader integration | Create |
| `pad-synth-face/tests/test_dfdc_pipeline_integration.py` | ingested fixture → `run_pipeline` smoke run | Create |
| `scripts/prepare_dfdc.py` | thin CLI wrapper | Create |
| `pad-synth-face/pyproject.toml` | new `dfdc` optional extra (`mediapipe`) | Modify |
| `docs/dfdc-bonafide.md` | source convention, prepare cmd, sweep-swap recipe, no-commit policy, EULA pointer | Create |

---

## Task 1: `DFDCBonafideIngested` provenance event

**Files:**
- Modify: `pad-synth-core/src/pad_synth_core/provenance.py`
- Create (test): `pad-synth-core/tests/test_provenance_dfdc.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-core/tests/test_provenance_dfdc.py`:

```python
import json

from pad_synth_core.provenance import DFDCBonafideIngested, ProvenanceLedger


def test_dfdc_bonafide_ingested_serialises(tmp_path):
    ev = DFDCBonafideIngested(
        license="DFDC research licence (Meta AI)",
        source_url="https://example.org/dfdc",
        n_chunks=2,
        n_videos=10,
        n_frames_written=58,
        detection_rate=0.967,
        real_filenames_sha256="abc123",
    )
    assert ev.type == "dfdc_bonafide_dataset_ingested"

    led_path = tmp_path / "provenance.jsonl"
    with ProvenanceLedger(led_path) as led:
        led.record(ev)
    rec = json.loads(led_path.read_text().splitlines()[0])
    assert rec["type"] == "dfdc_bonafide_dataset_ingested"
    assert rec["license"] == "DFDC research licence (Meta AI)"
    assert rec["n_videos"] == 10
    assert rec["detection_rate"] == 0.967
    assert "ingested_at" in rec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_provenance_dfdc.py -v`
Expected: FAIL — `ImportError: cannot import name 'DFDCBonafideIngested'`.

- [ ] **Step 3: Add the event**

In `pad-synth-core/src/pad_synth_core/provenance.py`, add this class after the existing `RealAttackIngested` class (reuses `_now`):

```python
class DFDCBonafideIngested(BaseModel):
    type: Literal["dfdc_bonafide_dataset_ingested"] = "dfdc_bonafide_dataset_ingested"
    license: str
    source_url: str
    n_chunks: int
    n_videos: int
    n_frames_written: int
    detection_rate: float
    real_filenames_sha256: str
    ingested_at: datetime = Field(default_factory=_now)
```

Then add it to the `ProvenanceEvent` union (extend the existing one):

```python
ProvenanceEvent = (
    BonafideIngested
    | GeneratorRegistered
    | OntologyCitation
    | RealAttackIngested
    | DFDCBonafideIngested
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-core/tests/test_provenance_dfdc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pad-synth-core/src/pad_synth_core/provenance.py pad-synth-core/tests/test_provenance_dfdc.py
git commit -m "feat(pad-core): DFDCBonafideIngested provenance event"
```

---

## Task 2: `build_fixture_dfdc` fixture

Synthesises a tiny DFDC-shaped source tree using ffmpeg `lavfi` — no real video, no PII. Skipped cleanly when ffmpeg isn't on PATH.

**Files:**
- Modify: `pad-synth-face/src/pad_synth_face/_fixtures.py`
- Create (test): `pad-synth-face/tests/test_fixture_dfdc.py`

- [ ] **Step 1: Write the failing test**

Create `pad-synth-face/tests/test_fixture_dfdc.py`:

```python
import json
import shutil

import pytest

from pad_synth_face._fixtures import build_fixture_dfdc


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def test_fixture_dfdc_layout(tmp_path):
    root = build_fixture_dfdc(tmp_path / "src")
    chunk = root / "chunk_00"
    assert (chunk / "metadata.json").is_file()
    metadata = json.loads((chunk / "metadata.json").read_text())
    # Two REALs and one FAKE.
    real = [k for k, v in metadata.items() if v["label"] == "REAL"]
    fake = [k for k, v in metadata.items() if v["label"] == "FAKE"]
    assert len(real) == 2
    assert len(fake) == 1
    # FAKE references one of the REALs.
    assert metadata[fake[0]]["original"] in real
    # Every referenced file exists.
    for name in metadata:
        assert (chunk / name).is_file()
        assert (chunk / name).stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_fixture_dfdc.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_fixture_dfdc'`.

- [ ] **Step 3: Implement the fixture**

Append to `pad-synth-face/src/pad_synth_face/_fixtures.py` (add `import json` and `import subprocess` near the existing imports if not already present):

```python
def build_fixture_dfdc(root: Path) -> Path:
    """Procedural DFDC-shaped source for tests: one chunk with 2 REAL +
    1 FAKE tiny mp4s (synthesised via ffmpeg lavfi -- no PII) and a
    matching metadata.json. Requires ffmpeg on PATH; tests should
    pytest.skip if it isn't.
    """
    import json
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    chunk = root / "chunk_00"
    chunk.mkdir(exist_ok=True)
    spec = [
        ("video_a.mp4", "REAL", None),
        ("video_b.mp4", "REAL", None),
        ("video_c.mp4", "FAKE", "video_a.mp4"),
    ]
    for name, _label, _orig in spec:
        out_path = chunk / name
        # 2-second 128x96 test pattern, h264 in mp4 (broadly compatible).
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc2=size=128x96:rate=10:d=2",
             "-pix_fmt", "yuv420p", "-c:v", "libx264",
             str(out_path)],
            check=True,
        )
    metadata = {
        name: ({"label": label, "original": orig} if orig else {"label": label})
        for name, label, orig in spec
    }
    (chunk / "metadata.json").write_text(json.dumps(metadata))
    return root
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_fixture_dfdc.py -v`
Expected: PASS (2 REAL + 1 FAKE mp4s + metadata.json).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/_fixtures.py pad-synth-face/tests/test_fixture_dfdc.py
git commit -m "feat(pad-dfdc): synthesised mp4-based DFDC fixture (no PII)"
```

---

## Task 3: `extract_dfdc_bonafide` core + tests

The substantive task. Implements the full ingester with a dependency-injected detector so tests use a deterministic stub.

**Files:**
- Create: `pad-synth-face/src/pad_synth_face/dfdc.py`
- Create (test): `pad-synth-face/tests/test_dfdc_ingest.py`

- [ ] **Step 1: Write the failing tests**

Create `pad-synth-face/tests/test_dfdc_ingest.py`:

```python
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pad_synth_face._fixtures import build_fixture_dfdc


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _stub_detector_center(frame: np.ndarray):
    """Deterministic: return a centered square bbox (half the smaller frame dim)."""
    h, w = frame.shape[:2]
    side = min(h, w) // 2
    return (w // 2 - side // 2, h // 2 - side // 2, side, side)


def _stub_detector_none(_frame: np.ndarray):
    return None


def test_extract_writes_one_dir_per_real_video(tmp_path):
    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    summary = extract_dfdc_bonafide(
        src=src, out=out,
        license="test-only", source_url="https://example.org/fixture",
        res=64, frames_per_video=3,
        detector=_stub_detector_center,
    )

    # Only the two REAL videos got directories; FAKE excluded.
    id_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert id_dirs == ["video_a", "video_b"]
    # Each REAL video produced 3 frames at 64x64.
    for name in id_dirs:
        pngs = sorted((out / name).glob("*.png"))
        assert len(pngs) == 3, f"{name}: {pngs}"
        arr = np.asarray(Image.open(pngs[0]).convert("RGB"))
        assert arr.shape == (64, 64, 3)
    assert summary["n_videos"] == 2
    assert summary["n_frames_written"] == 6
    assert 0.0 < summary["detection_rate"] <= 1.0


def test_manifest_records_dfdc_attribution(tmp_path):
    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    extract_dfdc_bonafide(
        src=src, out=out, license="LIC", source_url="URL",
        res=64, frames_per_video=2, detector=_stub_detector_center,
    )
    recs = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(recs) == 4  # 2 REAL videos x 2 frames
    for r in recs:
        assert r["label"] == "bonafide"
        assert r["attack_type"] is None
        assert r["bonafide_source"]["dataset"] == "DFDC"
        assert r["bonafide_source"]["license"] == "LIC"
        assert r["bonafide_source"]["id"] in {"video_a", "video_b"}


def test_provenance_event_recorded(tmp_path):
    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    extract_dfdc_bonafide(
        src=src, out=out, license="LIC", source_url="URL",
        res=64, frames_per_video=2, detector=_stub_detector_center,
    )
    prov = [json.loads(l) for l in (out / "provenance.jsonl").read_text().splitlines()]
    dfdc = [e for e in prov if e["type"] == "dfdc_bonafide_dataset_ingested"]
    assert len(dfdc) == 1
    assert dfdc[0]["license"] == "LIC"
    assert dfdc[0]["n_videos"] == 2
    assert dfdc[0]["n_frames_written"] == 4
    assert dfdc[0]["n_chunks"] == 1
    assert "real_filenames_sha256" in dfdc[0]


def test_idempotent_skips_existing_identities(tmp_path):
    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    common = dict(src=src, out=out, license="L", source_url="U",
                  res=64, frames_per_video=2, detector=_stub_detector_center)
    s1 = extract_dfdc_bonafide(**common)
    import hashlib
    def digest():
        h = hashlib.sha256()
        for p in sorted(out.rglob("*.png")):
            h.update(p.read_bytes())
        return h.hexdigest()
    d1 = digest()
    s2 = extract_dfdc_bonafide(**common)
    assert s2["n_videos"] == 0
    assert s2["n_frames_written"] == 0
    assert digest() == d1
    assert s1["n_videos"] == 2


def test_detection_failure_produces_empty_identity_and_zero_rate(tmp_path):
    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    summary = extract_dfdc_bonafide(
        src=src, out=out, license="L", source_url="U",
        res=64, frames_per_video=3, detector=_stub_detector_none,
    )
    assert summary["n_videos"] == 0
    assert summary["n_frames_written"] == 0
    assert summary["detection_rate"] == 0.0


def test_integration_with_digiface_loader(tmp_path):
    from pad_synth_face.bonafide import DigiFaceLoader
    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    extract_dfdc_bonafide(
        src=src, out=out, license="L", source_url="U",
        res=64, frames_per_video=2, detector=_stub_detector_center,
    )
    loader = DigiFaceLoader(out)
    identities = loader.list_identities()
    assert set(identities) == {"video_a", "video_b"}
    samples = loader.samples_for_identity("video_a")
    assert len(samples) == 2
    arr = loader.load(samples[0])
    assert arr.shape == (64, 64, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_dfdc_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: pad_synth_face.dfdc`.

- [ ] **Step 3: Implement the ingester**

Create `pad-synth-face/src/pad_synth_face/dfdc.py`:

```python
"""DFDC face-frame ingester.

Reads a DFDC source tree (videos + per-chunk metadata.json), extracts face
crops from REAL videos via an injectable detector (default: MediaPipe Face
Detection), and writes a DigiFace-shaped identity-per-directory layout that
`pad_synth_face.bonafide.DigiFaceLoader` consumes unchanged.

DFDC is licence-gated (research EULA). The default MediaPipe wrapper is
constructed lazily so the package imports cleanly even when the optional
`dfdc` extra (which pulls in `mediapipe`) is not installed -- tests inject
a stub detector and the import path stays cold.

Install MediaPipe for production use:
    pip install -e '.[dfdc]'
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import pad_synth_core
import pad_synth_face
from pad_synth_core.manifest import BonafideSource, ManifestWriter, SampleRecord
from pad_synth_core.provenance import DFDCBonafideIngested, ProvenanceLedger
from pad_synth_core.qc.per_sample import check_image_basic

# Detector signature: take an RGB uint8 frame, return (x, y, w, h) of the
# highest-confidence face bbox or None when no face is detected.
DetectorFn = Callable[[np.ndarray], tuple[int, int, int, int] | None]


def _default_mediapipe_detector() -> DetectorFn:
    """Construct a MediaPipe Face Detection wrapper. Lazy import: only called
    when the caller doesn't pass a detector explicitly."""
    import mediapipe as mp  # type: ignore[import-untyped]

    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    def detect(frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = frame_rgb.shape[:2]
        res = fd.process(frame_rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.score[0])
        b = best.location_data.relative_bounding_box
        return (
            int(b.xmin * w), int(b.ymin * h),
            int(b.width * w), int(b.height * h),
        )

    return detect


def _video_duration_seconds(video_path: Path) -> float:
    """Best-effort duration via ffprobe; falls back to 1.0 on any error."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return float(r.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 1.0


def _extract_frames(video_path: Path, n: int) -> list[np.ndarray]:
    """Pull n evenly-spaced frames from a video via ffmpeg. Missing frames
    (decode errors) are silently dropped -- the caller treats absence as a
    no-detection skip."""
    duration = _video_duration_seconds(video_path)
    timestamps = [duration * (i + 0.5) / n for i in range(n)]
    frames: list[np.ndarray] = []
    for t in timestamps:
        try:
            r = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}",
                 "-i", str(video_path), "-frames:v", "1",
                 "-f", "image2pipe", "-vcodec", "png", "-"],
                capture_output=True, check=True, timeout=15,
            )
            img = Image.open(io.BytesIO(r.stdout)).convert("RGB")
            frames.append(np.asarray(img, dtype=np.uint8))
        except (subprocess.SubprocessError, OSError):
            continue
    return frames


def _square_crop_resize(
    frame: np.ndarray, bbox: tuple[int, int, int, int],
    margin: float, res: int,
) -> np.ndarray:
    """Square-crop around bbox centre scaled by `margin`, clip to frame, resize."""
    fh, fw = frame.shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    half = max(w, h) * margin / 2.0
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(fw, int(round(cx + half)))
    y1 = min(fh, int(round(cy + half)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((res, res, 3), dtype=np.uint8)
    img = Image.fromarray(crop).resize((res, res), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def extract_dfdc_bonafide(
    src: Path,
    out: Path,
    license: str,
    source_url: str,
    res: int = 64,
    frames_per_video: int = 6,
    crop_margin: float = 1.3,
    max_videos: int | None = None,
    detector: DetectorFn | None = None,
) -> dict[str, Any]:
    """Extract face crops from DFDC REAL videos into a DigiFace-shaped layout.

    `detector` is injectable: pass a callable (frame_rgb) -> bbox|None to
    avoid importing MediaPipe (tests do this). When None, the default
    MediaPipe wrapper is constructed lazily on first use.
    """
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if detector is None:
        detector = _default_mediapipe_detector()

    # Discover (video_path, stem) for every REAL entry across all chunks.
    real_videos: list[tuple[Path, str]] = []
    chunks_seen: list[Path] = []
    for meta_path in sorted(src.rglob("metadata.json")):
        chunks_seen.append(meta_path.parent)
        records = json.loads(meta_path.read_text())
        for filename, info in sorted(records.items()):
            if info.get("label") != "REAL":
                continue
            video_path = meta_path.parent / filename
            if not video_path.exists():
                continue
            real_videos.append((video_path, Path(filename).stem))
    if max_videos is not None:
        real_videos = real_videos[:max_videos]

    n_videos = 0
    n_frames_written = 0
    n_frames_attempted = 0

    with ManifestWriter(out / "manifest.jsonl") as manifest:
        existing_ids = manifest.existing_sample_ids()
        for video_path, stem in real_videos:
            id_dir = out / stem
            # Per-identity idempotency: skip if any PNG already exists for this
            # identity (covers the common "re-run after success" case).
            if id_dir.exists() and any(id_dir.glob("*.png")):
                continue
            id_dir.mkdir(exist_ok=True)
            frames = _extract_frames(video_path, frames_per_video)
            n_frames_attempted += frames_per_video
            written_here = 0
            for i, frame in enumerate(frames):
                sid = f"dfdc-{stem}-{i:03d}"
                if sid in existing_ids:
                    continue
                bbox = detector(frame)
                if bbox is None:
                    continue
                arr = _square_crop_resize(frame, bbox, crop_margin, res)
                if not check_image_basic(arr, (res, res, 3)).ok:
                    continue
                out_rel = f"{stem}/{i:03d}.png"
                Image.fromarray(arr).save(out / out_rel, format="PNG")
                sha = hashlib.sha256((out / out_rel).read_bytes()).hexdigest()
                manifest.append(SampleRecord(
                    sample_id=sid,
                    modality="face",
                    label="bonafide",
                    attack_type=None,
                    bonafide_source=BonafideSource(
                        dataset="DFDC", id=stem, license=license,
                    ),
                    pipeline_version=f"pad-synth-face@{pad_synth_face.__version__}",
                    core_version=f"pad-synth-core@{pad_synth_core.__version__}",
                    ontology_version="dfdc-bonafide-ingest",
                    seed=0,
                    output_path=out_rel,
                    output_sha256=sha,
                ))
                written_here += 1
            if written_here > 0:
                n_videos += 1
                n_frames_written += written_here

    detection_rate = (
        n_frames_written / n_frames_attempted if n_frames_attempted else 0.0
    )
    sha_of_index = hashlib.sha256(
        "|".join(sorted(stem for _, stem in real_videos)).encode()
    ).hexdigest()

    # Provenance: only record an event when at least one frame was written
    # (matches the real-attack-capture idempotency: a no-op re-run logs nothing).
    if n_frames_written > 0:
        with ProvenanceLedger(out / "provenance.jsonl") as led:
            led.record(DFDCBonafideIngested(
                license=license,
                source_url=source_url,
                n_chunks=len(chunks_seen),
                n_videos=n_videos,
                n_frames_written=n_frames_written,
                detection_rate=detection_rate,
                real_filenames_sha256=sha_of_index,
            ))

    return {
        "out": str(out),
        "n_videos": n_videos,
        "n_frames_written": n_frames_written,
        "detection_rate": detection_rate,
        "n_real_filenames": len(real_videos),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_dfdc_ingest.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add pad-synth-face/src/pad_synth_face/dfdc.py pad-synth-face/tests/test_dfdc_ingest.py
git commit -m "feat(pad-dfdc): folder-walking ingester (injectable detector, DigiFace-shaped output)"
```

---

## Task 4: `prepare_dfdc.py` CLI wrapper

**Files:**
- Create: `scripts/prepare_dfdc.py`

- [ ] **Step 1: Write the script**

Create `scripts/prepare_dfdc.py`:

```python
#!/usr/bin/env python3
"""CLI wrapper: ingest a DFDC source tree into a DigiFace-shaped bonafide root.

Thin shim over `pad_synth_face.dfdc.extract_dfdc_bonafide`. Default detector
is MediaPipe Face Detection -- requires `pip install -e '.[dfdc]'` first.
See docs/dfdc-bonafide.md for the source-folder convention and the
sweep-swap recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.dfdc import extract_dfdc_bonafide  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="DFDC source tree (each chunk_dir has metadata.json + videos)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination DigiFace-shaped bonafide root (datasets/_real/dfdc_<res>/)")
    ap.add_argument("--license", required=True, help="Dataset licence / EULA string")
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--crop-margin", type=float, default=1.3)
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    summary = extract_dfdc_bonafide(
        src=args.src, out=args.out,
        license=args.license, source_url=args.source_url,
        res=args.res, frames_per_video=args.frames_per_video,
        crop_margin=args.crop_margin, max_videos=args.max_videos,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI against the fixture (with an injected-stub workaround)**

The CLI defaults to MediaPipe, which may not be installed. Smoke-test via the package function directly with a stub:

```bash
.venv/bin/python - <<'PY'
import shutil, sys, tempfile, pathlib
if shutil.which("ffmpeg") is None:
    print("ffmpeg not on PATH; skipping CLI smoke."); sys.exit(0)
sys.path.insert(0, "pad-synth-face/src")
sys.path.insert(0, "pad-synth-core/src")
from pad_synth_face._fixtures import build_fixture_dfdc
from pad_synth_face.dfdc import extract_dfdc_bonafide
import numpy as np
def stub(frame):
    h, w = frame.shape[:2]
    s = min(h, w) // 2
    return (w//2 - s//2, h//2 - s//2, s, s)
d = pathlib.Path(tempfile.mkdtemp())
build_fixture_dfdc(d/"src")
out = extract_dfdc_bonafide(
    src=d/"src", out=d/"out", license="x", source_url="y",
    res=64, frames_per_video=2, detector=stub,
)
assert out["n_videos"] == 2
assert (d/"out"/"video_a").is_dir()
print("CLI/function path OK:", out)
PY
```

Expected: prints `CLI/function path OK: {...}` and exits 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/prepare_dfdc.py
git commit -m "feat(pad-dfdc): prepare_dfdc.py CLI wrapper"
```

---

## Task 5: Pipeline-integration test

Proves an ingested DFDC fixture is usable as a bonafide source by `run_pipeline`.

**Files:**
- Create (test): `pad-synth-face/tests/test_dfdc_pipeline_integration.py`

- [ ] **Step 1: Write the test**

Create `pad-synth-face/tests/test_dfdc_pipeline_integration.py`:

```python
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from pad_synth_face._fixtures import build_fixture_dfdc
from pad_synth_face.dfdc import extract_dfdc_bonafide
from pad_synth_face.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_center(frame: np.ndarray):
    h, w = frame.shape[:2]
    s = min(h, w) // 2
    return (w // 2 - s // 2, h // 2 - s // 2, s, s)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_run_pipeline_consumes_dfdc_bonafide(tmp_path):
    # Ingest the fixture to a DigiFace-shaped bonafide root.
    src = build_fixture_dfdc(tmp_path / "src")
    bonafide_root = tmp_path / "bona"
    extract_dfdc_bonafide(
        src=src, out=bonafide_root, license="L", source_url="U",
        res=64, frames_per_video=2, detector=_stub_center,
    )

    cfg = {
        "run": {"name": "dfdc", "output": str(tmp_path / "out"),
                "seed": 1, "deterministic": True},
        "modality": "face",
        "bonafide": {"root": str(bonafide_root), "samples_per_bonafide": 1},
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "dfdc.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    summary = run_pipeline(cfg_path)
    # 2 REAL videos -> 2 identities, samples_per_bonafide=1 -> 2 attack + 2 bonafide.
    assert summary["samples_generated"] == 2
    assert summary["bonafide_emitted"] == 2
    assert summary["samples_failed"] == 0
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest pad-synth-face/tests/test_dfdc_pipeline_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pad-synth-face/tests/test_dfdc_pipeline_integration.py
git commit -m "test(pad-dfdc): pipeline consumes ingested DFDC fixture end-to-end"
```

---

## Task 6: `pyproject.toml` `dfdc` extra

**Files:**
- Modify: `pad-synth-face/pyproject.toml`

- [ ] **Step 1: Add the optional extra**

In `pad-synth-face/pyproject.toml`, extend the `[project.optional-dependencies]` block (which today has only `test`):

```toml
[project.optional-dependencies]
test = ["pytest>=8.0"]
dfdc = ["mediapipe>=0.10"]
```

- [ ] **Step 2: Verify the extra parses and the package still installs**

Run: `.venv/bin/python -m pip install -e pad-synth-face --no-deps --quiet 2>&1 | tail -2`
Expected: `Successfully installed pad-synth-face-0.1.0` (or already-installed equivalent), no parse error. (We do not install the `dfdc` extra here — only confirming the pyproject is valid.)

- [ ] **Step 3: Commit**

```bash
git add pad-synth-face/pyproject.toml
git commit -m "feat(pad-dfdc): optional dfdc extra (mediapipe)"
```

---

## Task 7: Doc + full-suite/lint checkpoint

**Files:**
- Create: `docs/dfdc-bonafide.md`

- [ ] **Step 1: Write the doc**

Create `docs/dfdc-bonafide.md`:

````markdown
# DFDC-grounded bonafide: ingesting Meta's DFDC for the existing pipeline

Replaces (or augments) the DigiFace bonafide source with real face frames
extracted from Meta's Deepfake Detection Challenge dataset. The existing
synthetic attack physics (print/replay/mask) rides on the new bonafide
distribution unchanged.

## 1. Obtain DFDC

DFDC is licence-gated (research EULA). Download from Meta / the original
Kaggle competition page; Preview (~5 GB) is the easiest first step, full
release is ~470 GB. Accept the licence terms before running anything below.

Unzip chunks anywhere on the laptop — each chunk extracts to a directory
containing videos plus a `metadata.json` mapping `filename -> {label:
REAL|FAKE, original: <REAL filename if FAKE>}`.

## 2. Install the optional `dfdc` extra (MediaPipe)

```bash
pip install -e 'pad-synth-face/[dfdc]'
```

Tests use a stub detector and don't require this; production ingest does.

## 3. Ingest → DigiFace-shaped bonafide root

```bash
.venv/bin/python scripts/prepare_dfdc.py \
  --src /path/to/dfdc/chunks \
  --out datasets/_real/dfdc_64 \
  --license "DFDC research licence (Meta AI)" \
  --source-url "<the URL you downloaded from>" \
  --res 64 \
  --frames-per-video 6
```

Writes `datasets/_real/dfdc_64/<video_stem>/NNN.png` (one directory per
REAL video, frames inside), plus `manifest.jsonl` and `provenance.jsonl`
recording the licence. Optional flags: `--max-videos N` for a quick smoke,
`--crop-margin 1.3` to widen the face crop. The default `--res 64` is a
drop-in replacement for DigiFace; bump it later when A1+A2 ships.

**Real frames are never committed.** `datasets/` is gitignored; keep
ingested DFDC roots under `datasets/_real/dfdc_<res>/`. Only the script,
fixture, tests, doc, and provenance/manifest schemas are committed.

## 4. Pin Set A / Set B identities

After the first ingest, pick disjoint identity lists from the ingested
videos and commit them:

```bash
.venv/bin/python - <<'PY'
import pathlib, random
ids = sorted(p.name for p in pathlib.Path("datasets/_real/dfdc_64").iterdir() if p.is_dir())
random.Random(20260528).shuffle(ids)
seta, setb = ids[:8], ids[8:24]
pathlib.Path("configs/dfdc_identities_seta.txt").write_text("\n".join(seta) + "\n")
pathlib.Path("configs/dfdc_identities_setb.txt").write_text("\n".join(setb) + "\n")
print("pinned:", len(seta), "Set A,", len(setb), "Set B")
PY
git add configs/dfdc_identities_set*.txt
git commit -m "feat(pad-dfdc): pin DFDC Set A/B identities"
```

## 5. Create dfdc_set*_d* sweep configs (paste once after ingest)

For each `(set, d)` ∈ {(a, 1), (a, 2), (a, 3), (b, 1), (b, 2), (b, 3)},
write `configs/runs/dfdc_<set>_d<n>.yaml` as a clone of
`real_<set>_d<n>.yaml` with two changes — point `bonafide.root` at the
DFDC dir and `bonafide.identities_file` at the DFDC list, e.g.
`configs/runs/dfdc_seta_d3.yaml`:

```yaml
run:
  name: dfdc_seta_d3
  output: ./datasets/dfdc_seta_d3
  seed: 20260522
  deterministic: true

modality: face

bonafide:
  root: ./datasets/_real/dfdc_64
  samples_per_bonafide: 256
  identities_file: ./configs/dfdc_identities_seta.txt
  splits: {train: 0.0, dev: 0.0, test: 1.0}

attacks:
  mask:
    weight: 1.0
    ontology: ./ontology/face/mask.yaml

sensor_preset: mobile-front-2024
```

(Mirror the corresponding `real_set*` files for non-D3 / Set B; preserve
the seeds and `samples_per_bonafide` numbers.)

## 6. Run the sweep

Generate the synthetic datasets locally, rsync to the Spark, sweep on the
GB10 — same procedure as `docs/real-attack-capture.md` §4, swapping
`mix_seta_d*` for `dfdc_seta_d*`. The headline question is whether real
DFDC bonafide + v2.1 synthetic attacks beats the DigiFace-bonafide baseline
(mask-only L3·D3 ≈ 0.089, integrated L2·D3 ≈ 0.094) at 64×64. Append the
result table to `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`.

If yes, the next cycle is the A1+A2 resolution bump which compounds with
this.
````

- [ ] **Step 2: Full-suite + lint checkpoint**

Run:
```bash
.venv/bin/python -m pytest -q
uvx ruff check --select E,F,B,UP --line-length 100 --ignore E501 \
  pad-synth-face/src/pad_synth_face/dfdc.py \
  pad-synth-face/src/pad_synth_face/_fixtures.py \
  pad-synth-core/src/pad_synth_core/provenance.py \
  scripts/prepare_dfdc.py \
  pad-synth-face/tests/test_dfdc_ingest.py \
  pad-synth-face/tests/test_dfdc_pipeline_integration.py \
  pad-synth-face/tests/test_fixture_dfdc.py \
  pad-synth-core/tests/test_provenance_dfdc.py
```
Expected: suite green (prior baseline was 207 passed / 1 skipped; this adds the new dfdc tests — some skipped if ffmpeg isn't on PATH). Ruff: `All checks passed!` on the new files.

Note on ruff: run via `uvx ruff` with `--select E,F,B,UP`. Do NOT use the `I`/isort rule from the repo root — `uvx ruff` misclassifies the `src`-layout packages as third-party and will spuriously rewrite import blocks across the whole codebase. Match the existing files' import style (blank line before first-party imports) by hand.

- [ ] **Step 3: Commit**

```bash
git add docs/dfdc-bonafide.md
git commit -m "docs(pad-dfdc): folder convention + prepare/sweep guide"
```

---

## Self-review notes

- **Spec coverage:** §3 input contract → Tasks 2 + 3 (fixture + walker); §4 canonical output → Task 3 (writes `<out>/<stem>/NNN.png`); §5 components → Tasks 1-7; §6 ingest pipeline → Task 3 `extract_dfdc_bonafide`; §7 detector injection → Task 3 (DetectorFn typedef + lazy MediaPipe import + tests using stub); §8 data handling → doc + provenance recording the licence; §9 identity = filename stem → Task 3 sample_id format; §10 configs deferred → doc §4-5; §11 testing → Tasks 2, 3, 5; §12 user workflow → doc.
- **Scope boundary upheld:** no DFDC download, no real-data ingest in tests, no sweep number — strictly the harness.
- **Test suite stays runnable without MediaPipe:** all tests inject a stub detector; the only consumer of `_default_mediapipe_detector()` is the production CLI path.
- **No commits of DFDC frames** — `datasets/` is gitignored; the doc reinforces the policy.
- **Backward compat:** `DigiFaceLoader` is unchanged; new provenance event is additive to the `ProvenanceEvent` union; existing tests/sweeps untouched.
