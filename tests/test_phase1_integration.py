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
    assert summary["samples_generated"] == 8 * 3  # 24 attacks
    assert summary["bonafide_emitted"] == 8 * 3  # 24 bonafide (per-identity × samples_per_bonafide)
    assert summary["samples_failed"] == 0

    manifest = (out_root / "manifest.jsonl").read_text().splitlines()
    provenance = (out_root / "provenance.jsonl").read_text().splitlines()
    assert len(manifest) == 48  # 24 attacks + 24 bonafide
    assert any("bonafide_dataset_ingested" in line for line in provenance)
    assert any("ontology_citation" in line for line in provenance)

    # Spot-check that one of the JPEGs really exists and is non-trivial.
    first = json.loads(manifest[0])
    img_path = out_root / first["output_path"]
    assert img_path.exists()
    assert img_path.stat().st_size > 100
