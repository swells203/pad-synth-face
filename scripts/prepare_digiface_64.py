#!/usr/bin/env python3
"""Resize DigiFace-1M images to 64x64, preserving <root>/<id>/<sample> layout.

Idempotent: skips files that already exist at the destination. Uses PIL's
LANCZOS resampling for quality. Writes a `_meta.json` recording the
operation summary (counts, target size).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TARGET_SIZE = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source DigiFace root: <src>/<identity>/<sample>.{png,jpg}")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Destination root for resized images")
    args = ap.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst
    dst_root.mkdir(parents=True, exist_ok=True)

    n_ids = 0
    n_samples = 0
    n_skipped = 0
    for id_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        identity = id_dir.name
        out_dir = dst_root / identity
        out_dir.mkdir(exist_ok=True)
        n_ids += 1
        for sample_path in sorted(id_dir.iterdir()):
            if sample_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            out_path = out_dir / f"{sample_path.stem}.png"
            if out_path.exists():
                n_skipped += 1
                continue
            with Image.open(sample_path) as im:
                im = im.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                im.save(out_path, format="PNG")
            n_samples += 1

    meta = {
        "target_size": TARGET_SIZE,
        "src": str(src_root),
        "identities": n_ids,
        "samples_total": n_samples + n_skipped,
        "samples_written": n_samples,
        "samples_skipped_existing": n_skipped,
    }
    (dst_root / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
