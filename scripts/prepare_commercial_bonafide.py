#!/usr/bin/env python3
"""CLI wrapper: ingest a commercially-licensed bonafide set into the canonical
224 bonafide root. Thin shim over pad_synth_face.commercial_bonafide.

See docs/commercial-bonafide.md for the input contract and the validation flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.commercial_bonafide import ingest_commercial_bonafide  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path,
                    help="Canonical source root: <src>/<identity>/<sample>.{png,jpg,jpeg}")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination bonafide root (use datasets/_real/commercial_224)")
    ap.add_argument("--license", required=True, help="Commercial licence / EULA string")
    ap.add_argument("--source-url", required=True,
                    help="URL the dataset was obtained from (recorded in provenance)")
    ap.add_argument("--vendor", default="unknown")
    ap.add_argument("--max-per-identity", type=int, default=None)
    args = ap.parse_args()

    summary = ingest_commercial_bonafide(
        src=args.src, out=args.out,
        license=args.license, source_url=args.source_url,
        vendor=args.vendor, max_per_identity=args.max_per_identity,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
