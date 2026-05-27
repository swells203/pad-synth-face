import numpy as np
from PIL import Image

from pad_synth_face._fixtures import build_fixture_real_attack


def test_fixture_real_attack_layout(tmp_path):
    root = build_fixture_real_attack(tmp_path / "src")

    bonafide = sorted((root / "bonafide").rglob("*.png"))
    print_a = sorted((root / "attack" / "print").rglob("*.png"))
    replay_a = sorted((root / "attack" / "replay").rglob("*.png"))

    assert len(bonafide) >= 4
    assert len(print_a) >= 4
    assert len(replay_a) >= 4

    # Images are larger than 64 (resize must do work) and non-degenerate.
    arr = np.array(Image.open(bonafide[0]).convert("RGB"))
    assert arr.shape[0] > 64 and arr.shape[1] > 64
    assert float(arr.std()) >= 5.0
