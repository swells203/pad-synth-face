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
            "mask": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "mask.yaml"),
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
