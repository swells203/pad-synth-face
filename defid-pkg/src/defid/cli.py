"""DefinitiveID PoC CLI: `defid generate` / `defid eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from defid.pipeline import run_generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="defid")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a synthetic behavioral dataset")
    g.add_argument("--config", required=True, type=Path)

    e = sub.add_parser("eval", help="Train + evaluate auth EER and bot accuracy")
    e.add_argument("--train-root", required=True, type=Path)
    e.add_argument("--eval-root", required=False, type=Path, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        json.dump(run_generation(args.config), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "eval":
        from defid.evaluate import evaluate

        json.dump(
            evaluate(args.train_root, args.eval_root), sys.stdout, indent=2
        )
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
