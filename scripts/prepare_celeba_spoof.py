#!/usr/bin/env python3
"""CLI: stage a CelebA-Spoof tree -> canonical real-attack dataset with
person-disjoint subject ids. See docs/celeba-spoof-b1.md.

CelebA-Spoof is non-commercial research-only -- this answers the B1 buy
decision, it does not produce a shippable model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_face.celeba_spoof import stage_celeba_spoof  # noqa: E402
from pad_synth_face.real_attack import ingest_real_attack  # noqa: E402

_DEFAULT_LICENSE = (
    "CelebA-Spoof: non-commercial research and educational use only "
    "(see github.com/ZhangYuanhan-AI/CelebA-Spoof)"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="CelebA-Spoof root")
    ap.add_argument("--out", type=Path,
                    default=REPO / "datasets/_real_attack/celeba_spoof")
    ap.add_argument("--staging", type=Path,
                    default=REPO / "datasets/_real_attack/_staging_celeba")
    ap.add_argument("--license", default=_DEFAULT_LICENSE)
    ap.add_argument("--source-url",
                    default="https://github.com/ZhangYuanhan-AI/CelebA-Spoof")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the representative subject shuffle (when --max-subjects caps).")
    args = ap.parse_args()

    staging = args.staging
    stage_summary = stage_celeba_spoof(
        args.src, staging, max_subjects=args.max_subjects, seed=args.seed)

    def subject_id_fn(fp: Path) -> str:
        parts = fp.relative_to(staging).parts
        return parts[1] if parts[0] == "bonafide" else parts[2]

    ingest_summary = ingest_real_attack(
        src=staging, out=args.out, dataset_name="CelebA-Spoof",
        license=args.license, source_url=args.source_url,
        max_per_class=args.max_per_class, subject_id_fn=subject_id_fn,
    )
    json.dump({"stage": stage_summary, "ingest": ingest_summary},
              sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
