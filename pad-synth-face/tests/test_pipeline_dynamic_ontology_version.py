import json
from pathlib import Path

import yaml

from pad_synth_face._fixtures import build_fixture_bonafide
from pad_synth_face.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]


def test_manifest_records_dynamic_ontology_version(tmp_path: Path):
    """The manifest's ontology_version must match the loaded print ontology's
    version (currently 2026-05-23 post v2.1), not a hardcoded string."""
    fixture_root = build_fixture_bonafide(tmp_path / "fixture")
    config = {
        "run": {
            "name": "dyn_ver_test",
            "output": str(tmp_path / "out"),
            "seed": 20260522,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_root), "samples_per_bonafide": 1},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO / "ontology" / "face" / "replay.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_pipeline(cfg_path)

    print_ont_yaml = yaml.safe_load((REPO / "ontology" / "face" / "print.yaml").read_text())
    expected_version = print_ont_yaml["version"]
    assert expected_version != "2026-05-11", (
        "test guard: print ontology has been bumped past v1; "
        "if you see this, the v2/v2.1 bumps were reverted somehow"
    )

    manifest = (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    assert manifest, "manifest is empty"
    for line in manifest:
        rec = json.loads(line)
        assert rec["ontology_version"] == expected_version, (
            f"sample {rec['sample_id']} has ontology_version={rec['ontology_version']!r}, "
            f"expected {expected_version!r}"
        )


def test_pipeline_honors_bonafide_identities_file(tmp_path: Path):
    """When bonafide.identities_file is set, the pipeline restricts iteration
    to those identities."""
    fixture_root = build_fixture_bonafide(tmp_path / "fixture")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("00000002\n00000005\n")
    config = {
        "run": {
            "name": "restrict_test",
            "output": str(tmp_path / "out"),
            "seed": 20260522,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_root),
            "samples_per_bonafide": 2,
            "identities_file": str(ids_file),
        },
        "attacks": {
            "print": {"weight": 1.0, "ontology": str(REPO / "ontology" / "face" / "print.yaml")},
            "replay": {"weight": 1.0, "ontology": str(REPO / "ontology" / "face" / "replay.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_pipeline(cfg_path)

    manifest = (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    sources = {json.loads(line)["bonafide_source"]["id"] for line in manifest}
    assert sources == {"00000002", "00000005"}, f"expected only those 2 IDs, got {sources}"
