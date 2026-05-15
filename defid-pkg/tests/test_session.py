import json
from pathlib import Path

import pytest

from defid.session import BehavioralSession, SessionManifestWriter


def make_session(sid: str = "s1", label: str = "genuine") -> BehavioralSession:
    return BehavioralSession(
        session_id=sid,
        label=label,
        subject_id="subj-0",
        touch=[{"t": 0.0, "x": 1.0, "y": 2.0, "phase": "move"}],
        key=[{"t": 0.0, "phase": "down", "field": "f1"}],
        motion=[{"t": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8}],
        ontology_version="2026-05-15",
        generator_version="defid-gen@0.1.0",
        seed=42,
    )


def test_session_serializes():
    s = make_session()
    blob = json.loads(s.model_dump_json())
    assert blob["label"] == "genuine"
    assert blob["touch"][0]["x"] == 1.0


def test_session_rejects_bad_label():
    with pytest.raises(ValueError):
        BehavioralSession.model_validate(
            {**make_session().model_dump(), "label": "nope"}
        )


def test_manifest_writer_appends_and_resumes(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    w = SessionManifestWriter(path)
    w.append("a", "genuine", "subj-0", "sessions/a.json", "0" * 64)
    w.close()

    w2 = SessionManifestWriter(path)
    assert w2.existing_ids() == {"a"}
    w2.append("b", "bot", "subj-1", "sessions/b.json", "1" * 64)
    w2.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1])["session_id"] == "b"


def test_manifest_writer_tolerates_partial_line(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"session_id": "ok"}\n{"session_id": "partial')
    w = SessionManifestWriter(path)
    assert w.existing_ids() == {"ok"}
    w.close()
