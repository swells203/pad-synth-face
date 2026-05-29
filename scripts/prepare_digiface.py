#!/usr/bin/env python3
"""Resize DigiFace-1M images to the canonical IMAGE_SIZE (default 224x224),
preserving <root>/<id>/<sample> layout.

Idempotent: skips files that already exist at the destination. Uses PIL's
LANCZOS resampling. Writes `_meta.json` recording the target size,
identity count, and sample counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_core import IMAGE_SIZE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source DigiFace root: <src>/<identity>/<sample>.{png,jpg}")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Destination root for resized images")
    ap.add_argument("--size", type=int, default=IMAGE_SIZE,
                    help=f"Target square size (default: IMAGE_SIZE={IMAGE_SIZE})")
    args = ap.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst
    target_size: int = args.size
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
                im = im.convert("RGB").resize((target_size, target_size), Image.LANCZOS)
                im.save(out_path, format="PNG")
            n_samples += 1

    meta = {
        "target_size": target_size,
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
