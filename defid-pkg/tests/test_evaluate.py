import json
import subprocess
import sys
from pathlib import Path

import yaml

from defid.evaluate import evaluate
from defid.pipeline import run_generation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _gen(tmp_path: Path, name: str, domain: str, seed: int) -> Path:
    cfg = {
        "run": {"name": name, "output": str(tmp_path / name), "seed": seed},
        "ontology_dir": str(REPO_ROOT / "ontology" / "behavioral"),
        "domain": domain,
        "subjects": 8,
        "sessions_per_subject": 6,
    }
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run_generation(p)
    return tmp_path / name


def test_evaluate_in_domain_only(tmp_path: Path):
    a = _gen(tmp_path, "a", "a", 1)
    r = evaluate(a, None)
    assert 0.0 <= r["auth_eer_in_domain"] <= 1.0
    assert 0.0 <= r["bot_accuracy_in_domain"] <= 1.0
    assert r["auth_eer_cross_domain"] is None


def test_evaluate_cross_domain(tmp_path: Path):
    a = _gen(tmp_path, "a", "a", 1)
    b = _gen(tmp_path, "b", "b", 2)
    r = evaluate(a, b)
    assert r["auth_eer_cross_domain"] is not None
    assert 0.0 <= r["auth_eer_cross_domain"] <= 1.0
    assert r["bot_accuracy_cross_domain"] is not None


def test_cli_eval_runs(tmp_path: Path):
    a = _gen(tmp_path, "a", "a", 1)
    b = _gen(tmp_path, "b", "b", 2)
    res = subprocess.run(
        [sys.executable, "-m", "defid.cli", "eval",
         "--train-root", str(a), "--eval-root", str(b)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["auth_eer_cross_domain"] is not None
