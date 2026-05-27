from pathlib import Path

import yaml

from pad_synth_core.eval.baseline import train_and_cross_domain_eval
from pad_synth_face._fixtures import build_fixture_real_attack
from pad_synth_face.pipeline import run_pipeline
from pad_synth_face.real_attack import ingest_real_attack

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synth_train(fixture_bonafide_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "synth"
    cfg = {
        "run": {"name": "synth", "output": str(out), "seed": 1, "deterministic": True},
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 2},
        "attacks": {
            "print": {"weight": 1.0,
                      "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml")},
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "synth.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_pipeline(cfg_path)
    return out


def test_synth_to_real_eval_runs(fixture_bonafide_dir: Path, tmp_path: Path):
    train_root = _make_synth_train(fixture_bonafide_dir, tmp_path)

    real_src = build_fixture_real_attack(tmp_path / "real_src")
    real_out = tmp_path / "real"
    ingest_real_attack(
        src=real_src, out=real_out,
        dataset_name="FIXTURE-RA", license="test-only",
        source_url="https://example.org/fixture",
    )

    result = train_and_cross_domain_eval(
        train_root=train_root,
        eval_root=real_out,
        epochs=1,
        batch_size=8,
        seed=0,
        device="cpu",
    )
    # Cross-domain EER is finite and the real eval set was actually read
    # (6 bonafide + 12 attack = 18 real samples).
    assert 0.0 <= float(result["eer_cross_domain"]) <= 1.0
    assert result["n_val_cross_domain"] == 18
