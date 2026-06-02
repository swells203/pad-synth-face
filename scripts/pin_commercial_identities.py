#!/usr/bin/env python3
"""Pin disjoint commercial Set A / Set B identity lists from an ingested root.

Run ONCE after scripts/prepare_commercial_bonafide.py has populated
datasets/_real/commercial_224/<identity>/NNN.png. Writes the two identity
files referenced by configs/runs/commercial_set*_d*.yaml:

    configs/commercial_identities_seta.txt   (8 identities)
    configs/commercial_identities_setb.txt   (next 16 identities)

Deterministic seeded shuffle (idempotent for a given ingested set); Set A and
Set B are identity-disjoint, matching the subject-disjoint discipline of the
DigiFace real_set* baselines.
"""

from __future__ import annotations

import argparse
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=REPO / "datasets/_real/commercial_224",
        help="Ingested commercial bonafide root (one dir per subject).",
    )
    ap.add_argument("--seta-count", type=int, default=8, help="Identities in Set A.")
    ap.add_argument("--setb-count", type=int, default=16, help="Identities in Set B.")
    ap.add_argument("--seed", type=int, default=20260528, help="Shuffle seed.")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(
            f"Commercial root not found: {args.root}\n"
            "Run scripts/prepare_commercial_bonafide.py first "
            "(see docs/commercial-bonafide.md)."
        )

    ids = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    needed = args.seta_count + args.setb_count
    if len(ids) < needed:
        raise SystemExit(
            f"Only {len(ids)} ingested identities under {args.root}; "
            f"need >= {needed} (Set A {args.seta_count} + Set B {args.setb_count})."
        )

    random.Random(args.seed).shuffle(ids)
    seta = sorted(ids[: args.seta_count])
    setb = sorted(ids[args.seta_count : args.seta_count + args.setb_count])

    seta_path = REPO / "configs/commercial_identities_seta.txt"
    setb_path = REPO / "configs/commercial_identities_setb.txt"
    seta_path.write_text("\n".join(seta) + "\n")
    setb_path.write_text("\n".join(setb) + "\n")

    print(f"pinned: {len(seta)} Set A -> {seta_path}")
    print(f"        {len(setb)} Set B -> {setb_path}")
    print("next: git add configs/commercial_identities_set*.txt && commit")


if __name__ == "__main__":
    main()
