from pathlib import Path

from pad_synth_core.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_touch_ontology_loads():
    o = load_ontology(ONT / "touch.yaml")
    assert o.attack_type == "touch"
    assert "touch_speed_mean" in o.axes
    assert "touch_jitter" in o.axes


def test_keystroke_ontology_loads():
    o = load_ontology(ONT / "keystroke.yaml")
    assert o.attack_type == "keystroke"
    assert "key_dwell_mean" in o.axes
    assert "key_flight_mean" in o.axes


def test_motion_ontology_loads():
    o = load_ontology(ONT / "motion.yaml")
    assert o.attack_type == "motion"
    assert "tremor_std" in o.axes
