from pathlib import Path

from pad_synth_core.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_print_ontology_loads():
    ont = load_ontology(REPO_ROOT / "ontology" / "face" / "print.yaml")
    assert ont.attack_type == "print"
    assert "paper_type" in ont.axes
    assert "print_dpi" in ont.axes


def test_replay_ontology_loads():
    ont = load_ontology(REPO_ROOT / "ontology" / "face" / "replay.yaml")
    assert ont.attack_type == "replay"
    assert "device_class" in ont.axes
    assert "refresh_hz" in ont.axes
