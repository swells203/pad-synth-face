"""Ingest a commercially-licensed bonafide face set into the canonical
224 bonafide root, recording licence provenance.

Input contract (canonical): <src>/<identity>/<sample>.{png,jpg,jpeg},
one directory per subject. Per-vendor layouts are reshaped into this
contract by a thin shim BEFORE calling this — see docs/commercial-bonafide.md.

Real images are never committed (datasets/ is gitignored). Only the licence
string + source URL travel with the data via provenance.jsonl.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from pad_synth_core import IMAGE_SIZE
from pad_synth_core.provenance import BonafideIngested, ProvenanceLedger

_IMG_EXT = {".png", ".jpg", ".jpeg"}


def ingest_commercial_bonafide(
    src: Path,
    out: Path,
    license: str,
    source_url: str,
    vendor: str = "unknown",
    size: int = IMAGE_SIZE,
    max_per_identity: int | None = None,
) -> dict[str, Any]:
    """Resize <src>/<id>/<sample> images to size x size PNGs under <out>/<id>/.

    Idempotent: skips destination files that already exist; records a single
    BonafideIngested provenance event only when at least one new file is
    written. Returns a summary dict.
    """
    src = Path(src)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    n_ids = 0
    n_written = 0
    n_skipped = 0
    for id_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        out_dir = out / id_dir.name
        out_dir.mkdir(exist_ok=True)
        n_ids += 1
        kept = 0
        for sample_path in sorted(id_dir.iterdir()):
            if sample_path.suffix.lower() not in _IMG_EXT:
                continue
            if max_per_identity is not None and kept >= max_per_identity:
                break
            kept += 1
            out_path = out_dir / f"{sample_path.stem}.png"
            if out_path.exists():
                n_skipped += 1
                continue
            with Image.open(sample_path) as im:
                im = im.convert("RGB").resize((size, size), Image.LANCZOS)
                im.save(out_path, format="PNG")
            n_written += 1

    sha_of_index = hashlib.sha256(
        "|".join(sorted(p.name for p in out.iterdir() if p.is_dir())).encode()
    ).hexdigest()

    if n_written > 0:
        with ProvenanceLedger(out / "provenance.jsonl") as led:
            led.record(BonafideIngested(
                name=f"commercial-bonafide:{vendor}",
                license=license,
                source_url=source_url,
                sha256_of_index=sha_of_index,
            ))

    meta = {
        "vendor": vendor,
        "target_size": size,
        "src": str(src),
        "identities": n_ids,
        "samples_written": n_written,
        "samples_skipped_existing": n_skipped,
        "license": license,
        "source_url": source_url,
    }
    (out / "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
