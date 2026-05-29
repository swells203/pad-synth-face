import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from pad_synth_core import IMAGE_SIZE
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
        res=IMAGE_SIZE, frames_per_video=2, detector=_stub_center,
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
