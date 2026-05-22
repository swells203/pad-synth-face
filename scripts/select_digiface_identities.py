#!/usr/bin/env python3
"""Deterministically select 8 identities for Set A and 16 disjoint identities
for Set B from a DigiFace-1M root directory. Writes the two committed text
files. Seeded for reproducibility."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="DigiFace root (resized 64x64 dir)")
    ap.add_argument("--seta-out", type=Path,
                    default=REPO / "configs" / "digiface_identities_seta.txt")
    ap.add_argument("--setb-out", type=Path,
                    default=REPO / "configs" / "digiface_identities_setb.txt")
    ap.add_argument("--seta-count", type=int, default=8)
    ap.add_argument("--setb-count", type=int, default=16)
    args = ap.parse_args()

    all_ids = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    assert len(all_ids) >= args.seta_count + args.setb_count, (
        f"need at least {args.seta_count + args.setb_count} identities; "
        f"found {len(all_ids)}"
    )

    rng = np.random.default_rng(20260522)
    order = rng.permutation(len(all_ids)).tolist()
    shuffled = [all_ids[i] for i in order]
    seta = sorted(shuffled[: args.seta_count])
    setb = sorted(shuffled[args.seta_count : args.seta_count + args.setb_count])

    args.seta_out.write_text("\n".join(seta) + "\n")
    args.setb_out.write_text("\n".join(setb) + "\n")

    print(f"Wrote {len(seta)} Set A identities to {args.seta_out}")
    print(f"Wrote {len(setb)} Set B identities to {args.setb_out}")
    print(f"Set A: {seta}")
    print(f"Set B: {setb}")


if __name__ == "__main__":
    main()
