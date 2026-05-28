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
    """Best-effort duration via ffprobe (or ffmpeg -i fallback); returns 1.0 on error."""
    # Try ffprobe first; fall through to ffmpeg -i on any failure (including
    # FileNotFoundError when ffprobe is not installed).
    for args, parser in [
        (
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            lambda out, _err: float(out.strip()),
        ),
        (
            ["ffmpeg", "-loglevel", "error", "-i", str(video_path),
             "-f", "null", "-"],
            # ffmpeg -i prints "Duration: HH:MM:SS.ss" to stderr even in error mode
            lambda _out, err: _parse_ffmpeg_duration(err),
        ),
    ]:
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, timeout=15,
            )
            return parser(r.stdout, r.stderr)
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return 1.0


def _parse_ffmpeg_duration(stderr: str) -> float:
    """Extract duration seconds from ffmpeg -i stderr output."""
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
    if not m:
        raise ValueError("no Duration line in ffmpeg stderr")
    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mn * 60 + s


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
