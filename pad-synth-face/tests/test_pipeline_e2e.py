import json
from pathlib import Path

import yaml

from pad_synth_face.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_pipeline_produces_manifest_and_images(
    fixture_bonafide_dir: Path, tmp_path: Path
):
    config = {
        "run": {
            "name": "smoke",
            "output": str(tmp_path / "out"),
            "seed": 1234,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_bonafide_dir),
            "samples_per_bonafide": 2,
            "splits": {"train": 0.5, "dev": 0.25, "test": 0.25},
        },
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    # samples_per_bonafide=2, 8 identities, 2 attack types — the orchestrator
    # emits 16 attack slots and 16 bonafide slots (samples_per_bonafide × ids).
    assert summary["samples_generated"] == 16
    assert summary["samples_failed"] == 0
    assert summary["bonafide_emitted"] == 16
    assert summary["bonafide_failed"] == 0

    manifest_path = Path(config["run"]["output"]) / "manifest.jsonl"
    lines = manifest_path.read_text().strip().split("\n")
    assert len(lines) == 32  # 16 attacks + 16 bonafide
    sample = json.loads(lines[0])
    assert sample["modality"] == "face"
    assert (Path(config["run"]["output"]) / sample["output_path"]).exists()

    bonafide_jpegs = list((Path(config["run"]["output"]) / "face" / "bonafide").glob("*.jpg"))
    assert len(bonafide_jpegs) == 16

    labels = [json.loads(line)["label"] for line in lines]
    assert "bonafide" in labels
    assert "attack" in labels


def test_run_pipeline_is_resumable(fixture_bonafide_dir: Path, tmp_path: Path):
    config = {
        "run": {
            "name": "smoke",
            "output": str(tmp_path / "out"),
            "seed": 99,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {
            "root": str(fixture_bonafide_dir),
            "samples_per_bonafide": 1,
        },
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    first = run_pipeline(cfg_path)
    second = run_pipeline(cfg_path)  # everything already done
    assert first["samples_generated"] == 8
    assert first["bonafide_emitted"] == 8
    assert second["samples_generated"] == 0
    assert second["samples_skipped_existing"] == 8
    assert second["bonafide_emitted"] == 0


def test_run_pipeline_counts_empty_identity_as_failure(tmp_path: Path):
    # Build a bonafide root with one identity that has no PNGs.
    fixture_root = tmp_path / "bad_fixture"
    (fixture_root / "00000000").mkdir(parents=True)  # empty directory

    config = {
        "run": {
            "name": "smoke",
            "output": str(tmp_path / "out"),
            "seed": 1,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_root), "samples_per_bonafide": 1},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    assert summary["samples_generated"] == 0
    assert summary["samples_failed"] == 1
    assert summary["bonafide_failed"] == 1


def test_pipeline_accepts_webcam_1080p_preset(
    fixture_bonafide_dir: Path, tmp_path: Path
):
    config = {
        "run": {
            "name": "webcam_smoke",
            "output": str(tmp_path / "out"),
            "seed": 1,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 1},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
        },
        "sensor_preset": "webcam-1080p",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    assert summary["samples_generated"] == 8
    assert summary["bonafide_emitted"] == 8

    # Spot-check that one manifest record records the webcam preset.
    manifest_path = Path(config["run"]["output"]) / "manifest.jsonl"
    first = json.loads(manifest_path.read_text().splitlines()[0])
    assert first["sensor_preset"] == "webcam-1080p"


def test_run_pipeline_with_three_attacks(
    fixture_bonafide_dir: Path, tmp_path: Path
):
    config = {
        "run": {
            "name": "three_attacks",
            "output": str(tmp_path / "out"),
            "seed": 2024,
            "deterministic": True,
        },
        "modality": "face",
        "bonafide": {"root": str(fixture_bonafide_dir), "samples_per_bonafide": 2},
        "attacks": {
            "print": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "print.yaml"),
            },
            "replay": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "replay.yaml"),
            },
            "mask": {
                "weight": 1.0,
                "ontology": str(REPO_ROOT / "ontology" / "face" / "mask.yaml"),
            },
        },
        "sensor_preset": "mobile-front-2024",
    }
    cfg_path = tmp_path / "three.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    summary = run_pipeline(cfg_path)
    # 8 identities x samples_per_bonafide=2 = 16 attack slots, 16 bonafide.
    # (Attack type per slot is an independent weighted draw, so we assert the
    # three-attack path runs cleanly and only valid types appear, not that
    # mask appears for this specific seed.)
    assert summary["samples_generated"] == 16
    assert summary["bonafide_emitted"] == 16
    assert summary["samples_failed"] == 0

    manifest = (Path(config["run"]["output"]) / "manifest.jsonl").read_text()
    attack_types = {
        json.loads(line)["attack_type"]
        for line in manifest.splitlines()
        if json.loads(line)["label"] == "attack"
    }
    assert attack_types.issubset({"print", "replay", "mask"})
    assert attack_types  # at least one attack was emitted
