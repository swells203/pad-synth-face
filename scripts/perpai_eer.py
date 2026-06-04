#!/usr/bin/env python3
"""Per-PAI threshold-free EER for a synth->real (or any train->eval) pair.

Trains `--model` on `--synth-root`, scores `--real-root`, and reports EER for
bonafide-vs-each-attack-type separately (plus pooled), averaged over `--seeds`.
Threshold-free, so it sidesteps the synth-fixed-threshold collapse that makes
the stored ACER/APCER degenerate on real data. Analysis tool -- writes no files.

Example (the 2026-06-03 synth->real reality check, per PAI):
  python scripts/perpai_eer.py --synth-root datasets/mix_seta_d3 \
    --real-root datasets/_real_attack/axondata --model L4 --device cuda
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))

import torch  # noqa: E402

from pad_synth_core.eval.baseline import (  # noqa: E402
    TinyPADDataset,
    _score_dataset,
    pretrain_on_synth,
)
from pad_synth_core.eval.metrics import compute_eer  # noqa: E402
from pad_synth_core.eval.models_zoo import FACTORIES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-root", required=True)
    ap.add_argument("--real-root", required=True)
    ap.add_argument("--model", default="L4", choices=list(FACTORIES))
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    factory = FACTORIES[args.model]
    res: dict[str, list[float]] = collections.defaultdict(list)
    labels: list[int] = []
    atypes: list[str | None] = []
    for seed in seeds:
        model = pretrain_on_synth(args.synth_root, factory, epochs=args.epochs,
                                  batch_size=args.batch_size, seed=seed, device=args.device)
        model.eval()
        real = TinyPADDataset(args.real_root)
        scores, labels, atypes = _score_dataset(
            model, real, args.batch_size, torch.device(args.device))
        res["pooled"].append(compute_eer(scores, labels))
        bona = [i for i, l in enumerate(labels) if l == 0]
        for t in sorted({a for a in atypes if a is not None}):
            sel = bona + [i for i, (l, a) in enumerate(zip(labels, atypes)) if l == 1 and a == t]
            res[t].append(compute_eer([scores[i] for i in sel], [labels[i] for i in sel]))

    n_bona = sum(1 for l in labels if l == 0)
    by_t = collections.Counter(a for l, a in zip(labels, atypes) if l == 1)
    print(f"{args.model} trained on {args.synth_root} -> {args.real_root}")
    print(f"real set: {n_bona} bonafide + attacks {dict(by_t)}  | seeds {seeds}")
    print(f"{'metric':14} {'EER mean':>9} {'std':>6}   raw(seeds)")
    for k in ["pooled"] + sorted(k for k in res if k != "pooled"):
        v = res[k]
        label = k if k == "pooled" else f"bona-vs-{k}"
        print(f"{label:14} {statistics.mean(v):>9.3f} {statistics.pstdev(v):>6.3f}   "
              f"{[round(x, 3) for x in v]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
