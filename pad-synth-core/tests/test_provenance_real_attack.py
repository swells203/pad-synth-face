import json

from pad_synth_core.provenance import ProvenanceLedger, RealAttackIngested


def test_real_attack_ingested_serialises(tmp_path):
    ev = RealAttackIngested(
        name="MSU-MFSD",
        license="MSU research EULA",
        source_url="https://example.org/msu-mfsd",
        sha256_of_index="abc123",
        attack_types=["print", "replay"],
    )
    assert ev.type == "real_attack_dataset_ingested"

    led_path = tmp_path / "provenance.jsonl"
    with ProvenanceLedger(led_path) as led:
        led.record(ev)
    rec = json.loads(led_path.read_text().splitlines()[0])
    assert rec["type"] == "real_attack_dataset_ingested"
    assert rec["name"] == "MSU-MFSD"
    assert rec["license"] == "MSU research EULA"
    assert rec["attack_types"] == ["print", "replay"]
    assert "ingested_at" in rec
