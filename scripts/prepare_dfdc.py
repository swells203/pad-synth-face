#!/usr/bin/env python3
"""CLI wrapper: ingest a DFDC source tree into a DigiFace-shaped bonafide root.

Thin shim over `pad_synth_face.dfdc.extract_dfdc_bonafide`. Default detector
is MediaPipe Face Detection -- requires `pip install -e '.[dfdc]'` first.
See docs/dfdc-bonafide.md for the source-folder convention and the
sweep-swap recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.dfdc import extract_dfdc_bonafide  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="DFDC source tree (each chunk_dir has metadata.json + videos)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination DigiFace-shaped bonafide root (datasets/_real/dfdc_<res>/)")
    ap.add_argument("--license", required=True, help="Dataset licence / EULA string")
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--crop-margin", type=float, default=1.3)
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    summary = extract_dfdc_bonafide(
        src=args.src, out=args.out,
        license=args.license, source_url=args.source_url,
        res=args.res, frames_per_video=args.frames_per_video,
        crop_margin=args.crop_margin, max_videos=args.max_videos,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
