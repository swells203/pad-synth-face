"""Commercial-bonafide ingest: canonical contract -> 224 root + provenance.

Uses only generated images — no real or licensed faces ever touch the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pad_synth_face.commercial_bonafide import ingest_commercial_bonafide


def _make_canonical_tree(root: Path, n_ids: int = 3, per_id: int = 2) -> None:
    rng = np.random.default_rng(0)
    for i in range(n_ids):
        d = root / f"subj_{i:03d}"
        d.mkdir(parents=True)
        for j in range(per_id):
            arr = rng.integers(0, 256, size=(96, 80, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{j}.jpg")


def test_ingest_produces_canonical_224_root(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src)
    summary = ingest_commercial_bonafide(
        src=src, out=out,
        license="Acme commercial face licence v1",
        source_url="https://vendor.example/sample",
        vendor="acme",
    )
    # 3 identities, 2 samples each, all resized to 224x224 PNG
    id_dirs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(id_dirs) == 3
    for d in id_dirs:
        pngs = sorted(d.glob("*.png"))
        assert len(pngs) == 2
        with Image.open(pngs[0]) as im:
            assert im.size == (224, 224)
    assert summary["identities"] == 3
    assert summary["samples_written"] == 6


def test_ingest_records_licence_provenance(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src)
    ingest_commercial_bonafide(
        src=src, out=out,
        license="Acme commercial face licence v1",
        source_url="https://vendor.example/sample",
        vendor="acme",
    )
    prov_lines = (out / "provenance.jsonl").read_text().strip().splitlines()
    assert len(prov_lines) == 1
    rec = json.loads(prov_lines[0])
    assert rec["type"] == "bonafide_dataset_ingested"
    assert rec["license"] == "Acme commercial face licence v1"
    assert rec["source_url"] == "https://vendor.example/sample"
    meta = json.loads((out / "_meta.json").read_text())
    assert meta["vendor"] == "acme"
    assert meta["target_size"] == 224
    assert meta["identities"] == 3


def test_ingest_is_idempotent(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src)
    ingest_commercial_bonafide(
        src=src, out=out, license="L", source_url="U", vendor="acme",
    )
    second = ingest_commercial_bonafide(
        src=src, out=out, license="L", source_url="U", vendor="acme",
    )
    # Re-run skips already-written files; no duplicate provenance event.
    assert second["samples_written"] == 0
    assert second["samples_skipped_existing"] == 6
    prov_lines = (out / "provenance.jsonl").read_text().strip().splitlines()
    assert len(prov_lines) == 1


def test_ingest_respects_max_per_identity(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_canonical_tree(src, n_ids=2, per_id=5)
    summary = ingest_commercial_bonafide(
        src=src, out=out, license="L", source_url="U",
        vendor="acme", max_per_identity=3,
    )
    assert summary["samples_written"] == 6  # 2 ids * 3 capped
