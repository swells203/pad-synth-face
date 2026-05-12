import json
from pathlib import Path

from pad_synth_core.provenance import (
    BonafideIngested,
    GeneratorRegistered,
    OntologyCitation,
    ProvenanceLedger,
)


def test_ledger_records_bonafide_ingestion(tmp_path: Path):
    ledger = ProvenanceLedger(tmp_path / "provenance.jsonl")
    ledger.record(
        BonafideIngested(
            name="digiface_1m_fixture",
            license="MIT",
            source_url="local-fixture",
            sha256_of_index="abc123",
        )
    )
    ledger.close()
    lines = (tmp_path / "provenance.jsonl").read_text().strip().split("\n")
    parsed = json.loads(lines[0])
    assert parsed["type"] == "bonafide_dataset_ingested"
    assert parsed["name"] == "digiface_1m_fixture"


def test_ledger_records_multiple_event_types(tmp_path: Path):
    ledger = ProvenanceLedger(tmp_path / "provenance.jsonl")
    ledger.record(
        BonafideIngested(name="x", license="MIT", source_url="u", sha256_of_index="h")
    )
    ledger.record(
        GeneratorRegistered(
            name="g", version="1.0", license="MIT", commercial_ok=True, model_hash="h"
        )
    )
    ledger.record(
        OntologyCitation(
            attack_type="print",
            axis="paper_type",
            paper="Example 2024",
            doi="10.0/test",
        )
    )
    ledger.close()
    lines = (tmp_path / "provenance.jsonl").read_text().strip().split("\n")
    types = [json.loads(line)["type"] for line in lines]
    assert types == [
        "bonafide_dataset_ingested",
        "generator_registered",
        "ontology_citation",
    ]
