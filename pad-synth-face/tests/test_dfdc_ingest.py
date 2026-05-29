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
    recs = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
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
    prov = [json.loads(line) for line in (out / "provenance.jsonl").read_text().splitlines()]
    dfdc = [e for e in prov if e["type"] == "dfdc_bonafide_dataset_ingested"]
    assert len(dfdc) == 1
    assert dfdc[0]["license"] == "LIC"
    assert dfdc[0]["n_videos"] == 2
    assert dfdc[0]["n_frames_written"] == 4
    assert dfdc[0]["n_chunks"] == 1
    assert "real_filenames_sha256" in dfdc[0]


def test_idempotent_skips_existing_identities(tmp_path):
    import hashlib

    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = build_fixture_dfdc(tmp_path / "src")
    out = tmp_path / "out"
    common = dict(src=src, out=out, license="L", source_url="U",
                  res=64, frames_per_video=2, detector=_stub_detector_center)
    s1 = extract_dfdc_bonafide(**common)
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
    # Provenance is not recorded on a zero-write run (matches the idempotency
    # contract: a no-op run logs nothing).
    assert not (out / "provenance.jsonl").exists()
    # And no empty identity directories are left behind (mkdir is lazy).
    assert not (out / "video_a").exists()
    assert not (out / "video_b").exists()


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


def test_multi_chunk_discovery(tmp_path):
    """Two chunk_dirs side by side -> n_chunks=2, both chunks' REAL videos ingested."""
    import json as _json
    import subprocess

    from pad_synth_face.dfdc import extract_dfdc_bonafide

    src = tmp_path / "src"
    for chunk_i in range(2):
        chunk = src / f"chunk_{chunk_i:02d}"
        chunk.mkdir(parents=True)
        for v_i, label in enumerate(("REAL", "REAL")):
            name = f"chunk{chunk_i}_video{v_i}.mp4"
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "testsrc2=size=128x96:rate=10:d=2",
                 "-pix_fmt", "yuv420p", "-c:v", "libx264",
                 str(chunk / name)],
                check=True,
            )
        meta = {f"chunk{chunk_i}_video{v_i}.mp4": {"label": "REAL"}
                for v_i in range(2)}
        (chunk / "metadata.json").write_text(_json.dumps(meta))

    out = tmp_path / "out"
    summary = extract_dfdc_bonafide(
        src=src, out=out, license="L", source_url="U",
        res=64, frames_per_video=2, detector=_stub_detector_center,
    )
    assert summary["n_videos"] == 4  # 2 chunks x 2 REALs each

    prov = [json.loads(line) for line in (out / "provenance.jsonl").read_text().splitlines()]
    dfdc = [e for e in prov if e["type"] == "dfdc_bonafide_dataset_ingested"]
    assert dfdc[0]["n_chunks"] == 2
