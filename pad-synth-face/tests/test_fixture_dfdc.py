import json
import shutil

import pytest

from pad_synth_face._fixtures import build_fixture_dfdc


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def test_fixture_dfdc_layout(tmp_path):
    root = build_fixture_dfdc(tmp_path / "src")
    chunk = root / "chunk_00"
    assert (chunk / "metadata.json").is_file()
    metadata = json.loads((chunk / "metadata.json").read_text())
    # Two REALs and one FAKE.
    real = [k for k, v in metadata.items() if v["label"] == "REAL"]
    fake = [k for k, v in metadata.items() if v["label"] == "FAKE"]
    assert len(real) == 2
    assert len(fake) == 1
    # FAKE references one of the REALs.
    assert metadata[fake[0]]["original"] in real
    # Every referenced file exists.
    for name in metadata:
        assert (chunk / name).is_file()
        assert (chunk / name).stat().st_size > 0
