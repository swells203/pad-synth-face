#!/usr/bin/env python3
"""CLI wrapper: ingest a real-attack PAD dataset into the canonical eval layout.

Thin shim over pad_synth_face.real_attack.ingest_real_attack. See
docs/real-attack-capture.md for the folder convention and the synth->real
sweep command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.real_attack import ingest_real_attack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source: <src>/bonafide/** and <src>/attack/<type>/**")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination canonical dataset dir (under datasets/_real_attack/)")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--license", required=True, help="Dataset licence / EULA string")
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--max-per-class", type=int, default=None)
    args = ap.parse_args()

    summary = ingest_real_attack(
        src=args.src, out=args.out,
        dataset_name=args.dataset_name, license=args.license,
        source_url=args.source_url, max_per_class=args.max_per_class,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
