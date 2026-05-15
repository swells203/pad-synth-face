import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_poc_end_to_end(tmp_path: Path):
    import yaml

    def gen(name, domain, seed):
        cfg = {
            "run": {"name": name, "output": str(tmp_path / name), "seed": seed},
            "ontology_dir": str(REPO_ROOT / "ontology" / "behavioral"),
            "domain": domain,
            "subjects": 10,
            "sessions_per_subject": 8,
        }
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        r = subprocess.run(
            [sys.executable, "-m", "defid.cli", "generate", "--config", str(p)],
            capture_output=True, text=True, check=False,
        )
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    sa = gen("seta", "a", 20260515)
    sb = gen("setb", "b", 20260516)
    assert sa["generated"] == 10 * 8 * 3
    assert sb["generated"] == 10 * 8 * 3
    assert sa["failed"] == 0

    r = subprocess.run(
        [sys.executable, "-m", "defid.cli", "eval",
         "--train-root", str(tmp_path / "seta"),
         "--eval-root", str(tmp_path / "setb")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["auth_eer_in_domain"] < 0.45
    assert out["bot_accuracy_in_domain"] > 0.75
    assert out["auth_eer_cross_domain"] is not None
    assert out["bot_accuracy_cross_domain"] is not None
