"""Minimal Phase-1 CLI: `pad-synth-face generate --config <yaml>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pad_synth_face.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pad-synth-face")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        summary = run_pipeline(args.config)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
