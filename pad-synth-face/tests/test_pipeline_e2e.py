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
