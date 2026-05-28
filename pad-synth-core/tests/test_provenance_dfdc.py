import json

from pad_synth_core.provenance import DFDCBonafideIngested, ProvenanceLedger


def test_dfdc_bonafide_ingested_serialises(tmp_path):
    ev = DFDCBonafideIngested(
        license="DFDC research licence (Meta AI)",
        source_url="https://example.org/dfdc",
        n_chunks=2,
        n_videos=10,
        n_frames_written=58,
        detection_rate=0.967,
        real_filenames_sha256="abc123",
    )
    assert ev.type == "dfdc_bonafide_dataset_ingested"

    led_path = tmp_path / "provenance.jsonl"
    with ProvenanceLedger(led_path) as led:
        led.record(ev)
    rec = json.loads(led_path.read_text().splitlines()[0])
    assert rec["type"] == "dfdc_bonafide_dataset_ingested"
    assert rec["license"] == "DFDC research licence (Meta AI)"
    assert rec["n_videos"] == 10
    assert rec["detection_rate"] == 0.967
    assert "ingested_at" in rec
