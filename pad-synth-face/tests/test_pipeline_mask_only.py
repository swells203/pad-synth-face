import json
from pathlib import Path

import yaml

from pad_synth_face.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mask_only_pipeline_runs(fixture_bonafide_dir: Path, tmp_path: Path):
    config = {
        "run": {
            "name": "mask_only",
            "output": str(tmp_path / "out"),
            "seed": 7,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 1},
        "attacks": {
            "mask": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "mask.yaml"),
            }
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "mask_only.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    assert summary["samples_generated"] == 8
    assert summary["bonafide_emitted"] == 8

    manifest = (Path(config["run"]["output"]) / "manifest.jsonl").read_text()
    recs = [json.loads(line) for line in manifest.splitlines()]
    attack_types = {r["attack_type"] for r in recs if r["label"] == "attack"}
    assert attack_types == {"mask"}
    # Canonical version came from the mask ontology (no print present).
    assert any(r["ontology_version"] == "2026-05-22" for r in recs)
    # Mask images landed under face/mask/.
    assert list((Path(config["run"]["output"]) / "face" / "mask").glob("*.jpg"))
