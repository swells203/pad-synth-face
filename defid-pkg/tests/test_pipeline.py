import json
from pathlib import Path

import yaml

from defid.pipeline import run_generation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, domain: str) -> Path:
    cfg = {
        "run": {"name": "t", "output": str(tmp_path / "out"), "seed": 11},
        "ontology_dir": str(REPO_ROOT / "ontology" / "behavioral"),
        "domain": domain,
        "subjects": 4,
        "sessions_per_subject": 3,
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_generation_produces_manifest_and_payloads(tmp_path: Path):
    summary = run_generation(_cfg(tmp_path, "a"))
    assert summary["generated"] == 4 * 3 * 3
    assert summary["failed"] == 0
    out = Path(tmp_path / "out")
    manifest = (out / "manifest.jsonl").read_text().strip().split("\n")
    assert len(manifest) == 36
    first = json.loads(manifest[0])
    assert (out / first["payload_path"]).exists()
    prov = (out / "provenance.jsonl").read_text()
    assert "ontology_citation" in prov
    assert "bonafide_dataset_ingested" in prov


def test_generation_is_resumable(tmp_path: Path):
    cfg = _cfg(tmp_path, "a")
    first = run_generation(cfg)
    second = run_generation(cfg)
    assert first["generated"] == 36
    assert second["generated"] == 0
    assert second["skipped"] == 36
