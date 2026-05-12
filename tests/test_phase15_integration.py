import json
import subprocess
import sys
from pathlib import Path

import yaml

from pad_synth_face._fixtures import (
    build_extended_fixture_bonafide,
    build_fixture_bonafide,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config))


def _generate(cfg_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "generate",
         "--config", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_phase15_cross_domain_eval_end_to_end(tmp_path: Path):
    # --- Build both fixtures ---
    set_a_fixture = build_fixture_bonafide(tmp_path / "fixture_a")
    set_b_fixture = build_extended_fixture_bonafide(tmp_path / "fixture_b")

    # --- Generate Set A (mobile-front sensor) ---
    set_a_config = {
        "run": {"name": "set_a", "output": str(tmp_path / "set_a"),
                "seed": 20260511, "deterministic": True},
        "modality": "face",
        "bonafide": {
            "root": str(set_a_fixture),
            "samples_per_bonafide": 4,
            "splits": {"train": 0.5, "dev": 0.25, "test": 0.25},
        },
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
            "replay": {"weight": 1.0,
                       "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    set_a_cfg = tmp_path / "set_a.yaml"
    _write_config(set_a_cfg, set_a_config)
    summary_a = _generate(set_a_cfg)
    # 8 IDs x 4 samples each, separately for bonafide and attacks
    assert summary_a["samples_generated"] == 32
    assert summary_a["bonafide_emitted"] == 32

    # --- Generate Set B (webcam-1080p sensor, all-test split, extended fixture) ---
    set_b_config = {
        "run": {"name": "set_b", "output": str(tmp_path / "set_b"),
                "seed": 20260512, "deterministic": True},
        "modality": "face",
        "bonafide": {
            "root": str(set_b_fixture),
            "samples_per_bonafide": 4,
            "splits": {"train": 0.0, "dev": 0.0, "test": 1.0},
        },
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
            "replay": {"weight": 1.0,
                       "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml")},
        },
        "sensor_preset": "webcam-1080p",
    }
    set_b_cfg = tmp_path / "set_b.yaml"
    _write_config(set_b_cfg, set_b_config)
    summary_b = _generate(set_b_cfg)
    # 16 IDs x 4 samples
    assert summary_b["samples_generated"] == 64
    assert summary_b["bonafide_emitted"] == 64

    # --- Cross-domain eval ---
    result = subprocess.run(
        [sys.executable, "-m", "pad_synth_face.cli", "eval",
         "--train-root", str(tmp_path / "set_a"),
         "--eval-root", str(tmp_path / "set_b"),
         "--epochs", "5",
         "--seed", "0"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["eer_in_domain"] is not None
    assert out["eer_cross_domain"] is not None
    assert 0.0 <= out["eer_in_domain"] <= 1.0
    assert 0.0 <= out["eer_cross_domain"] <= 1.0
    assert out["n_val_cross_domain"] == 128  # 64 bonafide + 64 attack in Set B
