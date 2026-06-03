#!/usr/bin/env python3
"""B1 synth-pretrain -> real-finetune curve runner.

Splits the real set once (subject-disjoint) into (finetune pool, real test),
pretrains a model on the synthetic root once, then finetunes on N real samples
for each N and reports real-test EER -- the hybrid curve. See
docs/superpowers/specs/2026-06-03-pad-b1-finetune-curve-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pad-synth-core" / "src"))
sys.path.insert(0, str(REPO / "pad-synth-face" / "src"))

from pad_synth_core.eval.baseline import (  # noqa: E402
    TinyPADDataset,
    finetune_and_eval_on_real,
    pretrain_on_synth,
    subject_disjoint_split,
)
from pad_synth_core.eval.models_zoo import FACTORIES  # noqa: E402


def split_real(real_root: Path, test_fraction: float, seed: int):
    """Split the real set into (real_ds, pool_indices, test_ds), subject-disjoint.

    Guards that the test partition holds both classes -- EER is undefined on a
    single-class test set, so a degenerate split raises rather than emitting a
    meaningless number.
    """
    real_ds = TinyPADDataset(real_root)
    pool_sub, test_sub = subject_disjoint_split(real_ds, val_fraction=test_fraction, seed=seed)
    test_labels = {real_ds.items[i][1] for i in test_sub.indices}
    if len(test_labels) < 2:
        raise ValueError(
            f"real-test split has a single class {test_labels}; need both "
            "bonafide and attack. Increase --test-fraction or use more real data.")
    return real_ds, list(pool_sub.indices), test_sub


def run_curve(
    synth_root: Path, real_root: Path, n_list: list[int], output_dir: Path,
    model_factory: Callable[[], Any], mode: str = "full",
    test_fraction: float = 0.3, pretrain_epochs: int = 8, finetune_epochs: int = 8,
    finetune_lr: float = 1e-4, batch_size: int = 8, seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    real_ds, pool_indices, test_ds = split_real(real_root, test_fraction, seed)
    rng = np.random.default_rng(seed)
    pool_indices = list(pool_indices)
    rng.shuffle(pool_indices)
    pool_size = len(pool_indices)

    model = pretrain_on_synth(
        synth_root, model_factory, epochs=pretrain_epochs,
        batch_size=batch_size, seed=seed, device=device)
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    runs_dir = Path(output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n in n_list:
        if n > pool_size:
            print(f"[b1] requested N={n}, pool has {pool_size} -- skipped "
                  "(no silent capping)")
            rows.append({"n_real": n, "eer": None, "acer": None, "skipped": True})
            continue
        ft_ds = torch.utils.data.Subset(real_ds, pool_indices[:n])
        res = finetune_and_eval_on_real(
            state, model_factory, ft_ds, test_ds, mode=mode,
            epochs=finetune_epochs, lr=finetune_lr, batch_size=batch_size,
            seed=seed, device=device)
        (runs_dir / f"N{n}_seed{seed}.json").write_text(json.dumps(res, indent=2))
        rows.append({"n_real": n, "eer": res["eer_cross_domain"],
                     "acer": res["acer_cross_domain"], "skipped": False})

    summary = {"rows": rows, "pool_size": pool_size, "n_test": len(test_ds),
               "mode": mode, "seed": seed}
    (Path(output_dir) / "curve_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _render_curve(summary: dict[str, Any]) -> str:
    lines = [
        f"B1 finetune curve (mode={summary['mode']}, n_test={summary['n_test']}, "
        f"pool={summary['pool_size']})",
        "",
        f"{'N':>8} {'real-test EER':>14} {'ACER':>8}",
        "-" * 34,
    ]
    done = [r for r in summary["rows"] if not r["skipped"]]
    for r in summary["rows"]:
        if r["skipped"]:
            lines.append(f"{r['n_real']:>8} {'(skipped: N>pool)':>23}")
        else:
            eer = "n/a" if r["eer"] is None else f"{r['eer']:.3f}"
            acer = "n/a" if r["acer"] is None else f"{r['acer']:.3f}"
            lines.append(f"{r['n_real']:>8} {eer:>14} {acer:>8}")
    lines.append("")
    base = next((r for r in done if r["n_real"] == 0), None)
    top = done[-1] if done else None
    if base is not None and top is not None and top["n_real"] > 0 \
            and base["eer"] is not None and top["eer"] is not None:
        delta = top["eer"] - base["eer"]
        verdict = "helps" if delta < 0 else ("no change" if delta == 0 else "hurts")
        lines.append(f"finetuning {verdict}: EER N=0 {base['eer']:.3f} -> "
                     f"N={top['n_real']} {top['eer']:.3f} (delta {delta:+.3f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-root", required=True, type=Path)
    ap.add_argument("--real-root", required=True, type=Path)
    ap.add_argument("--n-list", default="0,50,200,1000",
                    help="Comma-separated finetune sample counts.")
    ap.add_argument("--finetune-mode", choices=("full", "head"), default="full")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--model", default="L4", choices=list(FACTORIES))
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    n_list = [int(x) for x in args.n_list.split(",") if x.strip() != ""]
    summary = run_curve(
        synth_root=args.synth_root, real_root=args.real_root, n_list=n_list,
        output_dir=args.output_dir, model_factory=FACTORIES[args.model],
        mode=args.finetune_mode, test_fraction=args.test_fraction,
        pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr, batch_size=args.batch_size, seed=args.seed,
        device=args.device)
    print(_render_curve(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
