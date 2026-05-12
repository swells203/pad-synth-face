import json
from pathlib import Path

import pytest

from pad_synth_core.manifest import BonafideSource, ManifestWriter, SampleRecord


def make_record(sample_id: str = "face-print-test001") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        modality="face",
        label="attack",
        attack_type="print",
        bonafide_source=BonafideSource(
            dataset="digiface_1m_fixture", id="00000001", license="MIT"
        ),
        attack_params={"paper_type": "matte", "print_dpi": 600},
        sensor_preset="mobile-front-2024",
        sensor_params={"iso": 200, "jpeg_qf": 90},
        generators_used=[],
        pipeline_version="pad-synth-face@0.1.0",
        core_version="pad-synth-core@0.1.0",
        ontology_version="ontology@2026-05-11",
        seed=1234,
        output_path="face/print/face-print-test001.jpg",
        output_sha256="0" * 64,
    )


def test_sample_record_serializes_to_json():
    rec = make_record()
    blob = rec.model_dump_json()
    parsed = json.loads(blob)
    assert parsed["sample_id"] == "face-print-test001"
    assert parsed["bonafide_source"]["license"] == "MIT"


def test_sample_record_rejects_bad_label():
    with pytest.raises(ValueError):
        SampleRecord.model_validate({**make_record().model_dump(), "label": "not-a-label"})


def test_manifest_writer_appends_jsonl(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    writer = ManifestWriter(path)
    writer.append(make_record("a"))
    writer.append(make_record("b"))
    writer.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["sample_id"] == "a"
    assert json.loads(lines[1])["sample_id"] == "b"


def test_manifest_writer_is_resumable(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"

    w1 = ManifestWriter(path)
    w1.append(make_record("a"))
    w1.close()

    w2 = ManifestWriter(path)
    assert w2.existing_sample_ids() == {"a"}
    w2.append(make_record("b"))
    w2.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_manifest_writer_skips_same_instance_duplicates(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    writer = ManifestWriter(path)
    writer.append(make_record("dup"))
    writer.append(make_record("dup"))
    writer.append(make_record("dup"))
    writer.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1


def test_manifest_writer_tolerates_partial_last_line(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    # Simulate a prior crash: one complete record + one truncated record (no
    # trailing newline, missing closing brace).
    path.write_text('{"sample_id": "ok", "label": "attack"}\n{"sample_id": "partial')

    writer = ManifestWriter(path)
    assert writer.existing_sample_ids() == {"ok"}
    writer.close()
