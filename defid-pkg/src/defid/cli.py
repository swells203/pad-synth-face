"""DefinitiveID PoC CLI: `defid generate` (eval added in Task 10)."""

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
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        summary = run_generation(args.config)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
