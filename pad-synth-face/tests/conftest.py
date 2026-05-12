from __future__ import annotations

from pathlib import Path

import pytest

from pad_synth_face._fixtures import build_fixture_bonafide


@pytest.fixture
def fixture_bonafide_dir(tmp_path: Path) -> Path:
    return build_fixture_bonafide(tmp_path / "digiface_fixture")
