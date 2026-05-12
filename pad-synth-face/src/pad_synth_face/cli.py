"""Minimal Phase-1+1.5 CLI: `pad-synth-face {generate,eval} --config <yaml>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pad_synth_face.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pad-synth-face")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate a synthetic PAD dataset")
    gen.add_argument("--config", required=True, type=Path)

    ev = sub.add_parser(
        "eval",
        help="Train a baseline PAD detector on train-root, optionally evaluate cross-domain on eval-root",
    )
    ev.add_argument("--train-root", required=True, type=Path)
    ev.add_argument("--eval-root", required=False, type=Path, default=None)
    ev.add_argument("--epochs", type=int, default=8)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        summary = run_pipeline(args.config)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "eval":
        # Lazy import — torch is heavy and only required for eval.
        from pad_synth_core.eval.baseline import train_and_cross_domain_eval

        result = train_and_cross_domain_eval(
            train_root=args.train_root,
            eval_root=args.eval_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
