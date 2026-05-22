"""Run the PAD capacity-x-data factorial sweep.

Cells: capacity L in {L1,L2,L3} x data level D in {D1,D2,D3} x seed in {0,1,2}.
Per cell, train on Set A at the matching D, eval cross-domain on Set B at the
matching D. Writes per-cell JSON to <out>/runs/<L>_<D>_<seed>.json plus a
summary.csv across all completed cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))

from pad_synth_core.eval.baseline import train_and_cross_domain_eval  # noqa: E402
from pad_synth_core.eval.models_zoo import FACTORIES  # noqa: E402

DATA_LEVELS = ("D1", "D2", "D3")
CAPACITIES = ("L1", "L2", "L3")
SEEDS = (0, 1, 2)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _parse_cells(spec: str | None) -> list[tuple[str, str, int]]:
    if not spec:
        return [(L, D, s) for L in CAPACITIES for D in DATA_LEVELS for s in SEEDS]
    cells = []
    for tok in spec.split(","):
        L, D, s = tok.strip().split(":")
        cells.append((L, D, int(s)))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    for L in ("a", "b"):
        for D in DATA_LEVELS:
            ap.add_argument(f"--set-{L}-{D.lower()}", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--cells",
        default=None,
        help="Comma-separated L:D:seed (e.g. 'L1:D1:0,L2:D2:1'); default = all 27.",
    )
    args = ap.parse_args()

    set_roots = {
        ("a", D): getattr(args, f"set_a_{D.lower()}") for D in DATA_LEVELS
    } | {
        ("b", D): getattr(args, f"set_b_{D.lower()}") for D in DATA_LEVELS
    }

    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha()
    torch_v = torch.__version__
    cuda_v = torch.version.cuda or "none"

    cells = _parse_cells(args.cells)
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["capacity", "data_level", "seed", "eer_in_domain",
                    "eer_cross_domain", "train_seconds"])

    for L, D, seed in cells:
        train_root = set_roots[("a", D)]
        eval_root = set_roots[("b", D)]
        t0 = time.time()
        out = train_and_cross_domain_eval(
            train_root=train_root,
            eval_root=eval_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
            model_factory=FACTORIES[L],
        )
        elapsed = time.time() - t0
        rec = {
            "capacity": L,
            "data_level": D,
            "seed": seed,
            "n_train": out["n_train"],
            "n_val_in_domain": out["n_val_in_domain"],
            "n_val_cross_domain": out["n_val_cross_domain"],
            "eer_in_domain": float(out["eer_in_domain"]),
            "eer_cross_domain": (
                float(out["eer_cross_domain"])
                if out["eer_cross_domain"] is not None
                else None
            ),
            "val_accuracy_in_domain": float(out["val_accuracy_in_domain"]),
            "val_accuracy_cross_domain": (
                float(out["val_accuracy_cross_domain"])
                if out["val_accuracy_cross_domain"] is not None
                else None
            ),
            "train_seconds": elapsed,
            "git_sha": git_sha,
            "torch_version": torch_v,
            "cuda_version": cuda_v,
            "device": args.device,
            "train_root": str(train_root),
            "eval_root": str(eval_root),
        }
        out_path = runs_dir / f"{L}_{D}_{seed}.json"
        out_path.write_text(json.dumps(rec, indent=2))
        with summary_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([
                L, D, seed, rec["eer_in_domain"], rec["eer_cross_domain"],
                f"{elapsed:.2f}",
            ])
        ec = rec["eer_cross_domain"]
        ec_s = f"{ec:.3f}" if ec is not None else "N/A"
        print(
            f"{L} {D} seed={seed}  eer_in={rec['eer_in_domain']:.3f}"
            f"  eer_cross={ec_s}  {elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
