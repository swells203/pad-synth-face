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
