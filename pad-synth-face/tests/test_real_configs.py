from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "configs" / "runs"

EXPECTED = {
    "real_seta_d1.yaml": (20260522, "mobile-front-2024", "./datasets/_real/digiface_224",
                          6, "./configs/digiface_identities_seta.txt"),
    "real_seta_d2.yaml": (20260522, "mobile-front-2024", "./datasets/_real/digiface_224",
                          32, "./configs/digiface_identities_seta.txt"),
    "real_seta_d3.yaml": (20260522, "mobile-front-2024", "./datasets/_real/digiface_224",
                          256, "./configs/digiface_identities_seta.txt"),
    "real_setb_d1.yaml": (20260523, "webcam-1080p", "./datasets/_real/digiface_224",
                          4, "./configs/digiface_identities_setb.txt"),
    "real_setb_d2.yaml": (20260523, "webcam-1080p", "./datasets/_real/digiface_224",
                          32, "./configs/digiface_identities_setb.txt"),
    "real_setb_d3.yaml": (20260523, "webcam-1080p", "./datasets/_real/digiface_224",
                          256, "./configs/digiface_identities_setb.txt"),
}


def test_real_configs_present_and_well_formed():
    for fname, (seed, sensor, fixture, spb, ids_file) in EXPECTED.items():
        cfg = yaml.safe_load((CFG_DIR / fname).read_text())
        assert cfg["run"]["seed"] == seed, fname
        assert cfg["run"]["deterministic"] is True, fname
        assert cfg["run"]["output"] == f"./datasets/{Path(fname).stem}", fname
        assert cfg["modality"] == "face", fname
        assert cfg["sensor_preset"] == sensor, fname
        assert cfg["bonafide"]["root"] == fixture, fname
        assert cfg["bonafide"]["samples_per_bonafide"] == spb, fname
        assert cfg["bonafide"]["identities_file"] == ids_file, fname
        assert set(cfg["attacks"].keys()) == {"print", "replay"}, fname
        assert cfg["attacks"]["print"]["weight"] == 1.0, fname
        assert cfg["attacks"]["replay"]["weight"] == 1.0, fname
