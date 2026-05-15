from pathlib import Path

from defid.generator import generate_session
from defid.qc import QCResult, check_session

REPO_ROOT = Path(__file__).resolve().parents[2]
ONT = REPO_ROOT / "ontology" / "behavioral"


def test_qc_passes_on_generated_session():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    r = check_session(s)
    assert isinstance(r, QCResult)
    assert r.ok
    assert r.reason is None


def test_qc_fails_on_empty_touch():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    s.touch = []
    r = check_session(s)
    assert not r.ok
    assert "touch" in r.reason


def test_qc_fails_on_nonfinite():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    s.touch[0]["x"] = float("nan")
    r = check_session(s)
    assert not r.ok
    assert "finite" in r.reason


def test_qc_fails_on_implausible_speed():
    s = generate_session("genuine", "subj-1", seed=1, ontology_dir=ONT)
    s.touch[5]["x"] = 1e9  # absurd jump
    r = check_session(s)
    assert not r.ok
    assert "speed" in r.reason
