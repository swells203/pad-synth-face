import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _seed_dataset(root: Path, n_bonafide: int, n_attack: int) -> Path:
    base = root / "face"
    for label_dir, n in (("bonafide", n_bonafide), ("print", n_attack)):
        d = base / label_dir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{i:04d}.jpg")
    return root


def test_one_cell_end_to_end_cpu(tmp_path):
    set_a = _seed_dataset(tmp_path / "set_a", n_bonafide=12, n_attack=12)
    set_b = _seed_dataset(tmp_path / "set_b", n_bonafide=12, n_attack=12)
    out_dir = tmp_path / "out"

    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "spark_sweep.py"),
            "--set-a-d1", str(set_a), "--set-b-d1", str(set_b),
            "--set-a-d2", str(set_a), "--set-b-d2", str(set_b),
            "--set-a-d3", str(set_a), "--set-b-d3", str(set_b),
            "--output-dir", str(out_dir),
            "--device", "cpu",
            "--epochs", "1",
            "--batch-size", "4",
            "--cells", "L1:D1:0",
        ],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr

    runs = list((out_dir / "runs").glob("*.json"))
    assert len(runs) == 1
    rec = json.loads(runs[0].read_text())
    assert rec["capacity"] == "L1"
    assert rec["data_level"] == "D1"
    assert rec["seed"] == 0
    for k in ("eer_in_domain", "eer_cross_domain", "train_seconds",
              "git_sha", "torch_version", "device"):
        assert k in rec
    assert 0.0 <= rec["eer_in_domain"] <= 1.0

    summary = list(csv.DictReader((out_dir / "summary.csv").open()))
    assert len(summary) == 1
    assert summary[0]["capacity"] == "L1"
