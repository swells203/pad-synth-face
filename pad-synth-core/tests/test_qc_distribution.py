import json
from pathlib import Path

from pad_synth_core.qc.distribution import (
    coverage_report,
    verify_identity_disjoint,
)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_coverage_report_counts_attack_types(tmp_path: Path):
    rows = [
        {"sample_id": f"s{i}", "attack_type": "print"} for i in range(7)
    ] + [
        {"sample_id": f"s{i+7}", "attack_type": "replay"} for i in range(3)
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    report = coverage_report(manifest)
    assert report["attack_type_counts"] == {"print": 7, "replay": 3}
    assert report["total"] == 10


def test_identity_disjoint_passes_on_clean_split(tmp_path: Path):
    rows = [
        {"sample_id": "a", "bonafide_source": {"id": "00"}, "split": "train"},
        {"sample_id": "b", "bonafide_source": {"id": "01"}, "split": "dev"},
        {"sample_id": "c", "bonafide_source": {"id": "02"}, "split": "test"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    result = verify_identity_disjoint(manifest)
    assert result.ok


def test_identity_disjoint_fails_on_leak(tmp_path: Path):
    rows = [
        {"sample_id": "a", "bonafide_source": {"id": "00"}, "split": "train"},
        {"sample_id": "b", "bonafide_source": {"id": "00"}, "split": "test"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    result = verify_identity_disjoint(manifest)
    assert not result.ok
    assert "00" in result.reason


def test_identity_disjoint_check_works_on_real_pipeline_output(tmp_path):
    """Smoke test: after a real pipeline run, verify_identity_disjoint should
    correctly read the split field from each SampleRecord and confirm no leak."""
    import yaml
    from pathlib import Path
    from pad_synth_face._fixtures import build_fixture_bonafide
    from pad_synth_face.pipeline import run_pipeline

    repo_root = Path(__file__).resolve().parents[2]
    fixture = build_fixture_bonafide(tmp_path / "digiface")
    config = {
        "run": {"name": "t", "output": str(tmp_path / "out"), "seed": 11, "deterministic": True},
        "modality": "face",
        "bonafide": {
            "root": str(fixture),
            "samples_per_bonafide": 1,
            "splits": {"train": 0.5, "dev": 0.25, "test": 0.25},
        },
        "attacks": {
            "print": {"weight": 1.0, "ontology": str(repo_root / "ontology" / "face" / "print.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_pipeline(cfg_path)
    manifest = Path(config["run"]["output"]) / "manifest.jsonl"

    # Every record should now have a split field populated.
    import json
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert all(r["split"] in {"train", "dev", "test"} for r in rows)

    # verify_identity_disjoint should pass on real pipeline output.
    result = verify_identity_disjoint(manifest)
    assert result.ok, result.reason
