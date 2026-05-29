"""Ingest a real-attack PAD dataset into the canonical eval layout.

Reads the folder convention
    <src>/bonafide/**/*.{jpg,jpeg,png}
    <src>/attack/<attack_type>/**/*.{jpg,jpeg,png}
and writes the canonical 64x64 dataset that pad_synth_core.eval consumes:
    <out>/face/bonafide/real-bonafide-NNNNNNNN.jpg
    <out>/face/<attack_type>/real-<attack_type>-NNNNNNNN.jpg
    <out>/manifest.jsonl
    <out>/provenance.jsonl   (RealAttackIngested -- records dataset + licence)

Input is extracted image frames (video decoding is the caller's pre-step).
Deterministic (sorted source order) and idempotent (existing sample IDs are
skipped; a no-op re-run records no new provenance event). Real images are
never committed -- write under datasets/_real_attack/ which the gitignored
datasets/ covers.

Manifest attribution convention: SampleRecord requires a `bonafide_source`,
so it is reused as dataset-level attribution for EVERY row (bonafide and
attack) -- `dataset`/`license` are the dataset's, and `id` holds the source
file's relative path. The manifest schema has no dedicated attack_source
field yet; this reuse is the documented convention until one is added.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import pad_synth_core
import pad_synth_face
from pad_synth_core import IMAGE_SHAPE, IMAGE_SIZE
from pad_synth_core.manifest import BonafideSource, ManifestWriter, SampleRecord
from pad_synth_core.provenance import ProvenanceLedger, RealAttackIngested
from pad_synth_core.qc.per_sample import check_image_basic

_EXTS = {".jpg", ".jpeg", ".png"}
_TARGET = IMAGE_SHAPE


def _list_images(d: Path) -> list[Path]:
    return sorted(
        p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _EXTS
    )


def _load_target(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def ingest_real_attack(
    src: Path,
    out: Path,
    dataset_name: str,
    license: str,
    source_url: str,
    max_per_class: int | None = None,
) -> dict[str, Any]:
    src, out = Path(src), Path(out)
    face_root = out / "face"
    counts: dict[str, int] = {}
    qc_skipped: dict[str, int] = {}
    index_paths: list[str] = []
    attack_types: list[str] = []

    with ManifestWriter(out / "manifest.jsonl") as manifest:
        existing = manifest.existing_sample_ids()

        def _process(subdir: str, attack_type: str | None, src_dir: Path, prefix: str) -> None:
            out_dir = face_root / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            files = _list_images(src_dir)
            if max_per_class is not None:
                files = files[:max_per_class]
            written = 0
            skipped = 0
            for i, fp in enumerate(files):
                index_paths.append(str(fp.relative_to(src)))
                sid = f"{prefix}-{i:08d}"
                if sid in existing:
                    continue
                arr = _load_target(fp)
                if not check_image_basic(arr, _TARGET).ok:
                    skipped += 1
                    continue
                out_rel = f"face/{subdir}/{sid}.jpg"
                Image.fromarray(arr).save(out / out_rel, format="JPEG", quality=92)
                sha = hashlib.sha256((out / out_rel).read_bytes()).hexdigest()
                manifest.append(SampleRecord(
                    sample_id=sid,
                    modality="face",
                    label="bonafide" if attack_type is None else "attack",
                    attack_type=attack_type,
                    bonafide_source=BonafideSource(
                        dataset=dataset_name, id=str(fp.relative_to(src)), license=license
                    ),
                    pipeline_version=f"pad-synth-face@{pad_synth_face.__version__}",
                    core_version=f"pad-synth-core@{pad_synth_core.__version__}",
                    ontology_version="real-attack-capture",
                    seed=0,
                    output_path=out_rel,
                    output_sha256=sha,
                ))
                written += 1
            counts[subdir] = written
            qc_skipped[subdir] = skipped

        bonafide_dir = src / "bonafide"
        if bonafide_dir.is_dir():
            _process("bonafide", None, bonafide_dir, "real-bonafide")

        attack_root = src / "attack"
        if attack_root.is_dir():
            for tdir in sorted(p for p in attack_root.iterdir() if p.is_dir()):
                attack_types.append(tdir.name)
                _process(tdir.name, tdir.name, tdir, f"real-{tdir.name}")

    # Idempotent: only log an ingest event when this run actually wrote samples.
    if sum(counts.values()) > 0:
        with ProvenanceLedger(out / "provenance.jsonl") as led:
            led.record(RealAttackIngested(
                name=dataset_name,
                license=license,
                source_url=source_url,
                sha256_of_index=hashlib.sha256(
                    "|".join(sorted(index_paths)).encode()
                ).hexdigest(),
                attack_types=attack_types,
            ))

    return {
        "out": str(out),
        "counts": counts,
        "qc_skipped": sum(qc_skipped.values()),
        "attack_types": attack_types,
    }
