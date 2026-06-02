#!/usr/bin/env python3
"""Matched-scale A/B verdict: does swapping DigiFace bonafide for a
commercially-licensed set preserve cross-domain EER?

Reads two sweep output dirs (each <dir>/runs/<CAP>_<DLEVEL>_<seed>.json with
eer_cross_domain), aggregates by (capacity, data_level), and prints a per-cell
delta table. PASS iff every shared cell has |Δ| <= band AND no commercial cell
mean <= collapse. Exits non-zero on FAIL. See
docs/superpowers/specs/2026-06-02-pad-commercial-bonafide-validation-design.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
_DEFAULT_BASELINE = (
    REPO / "docs/superpowers/reports/2026-05-22-pad-spark-sweep-results"
    / "runs_mix_224_L4_A2"
)


def aggregate(sweep_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Group cell JSONs by (capacity, data_level) -> aggregated stats."""
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for f in sorted((Path(sweep_dir) / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        rows.setdefault((r["capacity"], r["data_level"]), []).append(r)
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rs in rows.items():
        eers = [r["eer_cross_domain"] for r in rs]
        agg[key] = {
            "mean": statistics.mean(eers),
            "std": statistics.pstdev(eers) if len(eers) > 1 else 0.0,
            "n_seeds": len(eers),
            "n_train": rs[0].get("n_train"),
            "n_val_cross_domain": rs[0].get("n_val_cross_domain"),
        }
    return agg


def compare(
    baseline: dict[tuple[str, str], dict[str, Any]],
    commercial: dict[tuple[str, str], dict[str, Any]],
    band: float,
    collapse: float,
) -> dict[str, Any]:
    """Build the per-cell verdict table + overall pass/fail + warnings."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    passed = True
    shared = sorted(set(baseline) & set(commercial))
    for cap, d in shared:
        b = baseline[(cap, d)]
        c = commercial[(cap, d)]
        delta = c["mean"] - b["mean"]
        if c["mean"] <= collapse:
            verdict = "collapsed"
            passed = False
        elif abs(delta) > band:
            verdict = "delta_exceeds_band"
            passed = False
        else:
            verdict = "ok"
        if b.get("n_train") != c.get("n_train") or \
           b.get("n_val_cross_domain") != c.get("n_val_cross_domain"):
            warnings.append(
                f"{cap}/{d}: scale mismatch (baseline n_train={b.get('n_train')}, "
                f"commercial n_train={c.get('n_train')}) — not matched-scale"
            )
        rows.append({
            "capacity": cap, "data_level": d,
            "baseline_mean": b["mean"], "commercial_mean": c["mean"],
            "delta": delta, "verdict": verdict,
        })
    only_base = sorted(set(baseline) - set(commercial))
    only_comm = sorted(set(commercial) - set(baseline))
    for cap, d in only_base:
        warnings.append(f"{cap}/{d}: present in baseline only — not compared")
    for cap, d in only_comm:
        warnings.append(f"{cap}/{d}: present in commercial only — not compared")
    if not shared:
        passed = False
        warnings.append("no shared cells between the two sweeps")
    return {"passed": passed, "rows": rows, "warnings": warnings}


def _render(result: dict[str, Any], band: float) -> str:
    lines = [
        f"Commercial-bonafide vs DigiFace baseline (band ±{band:.3f})",
        "",
        f"{'cell':<8} {'DigiFace':>9} {'Commercial':>11} {'Δ':>8}  verdict",
        "-" * 48,
    ]
    for r in result["rows"]:
        cell = f"{r['capacity']}·{r['data_level']}"
        lines.append(
            f"{cell:<8} {r['baseline_mean']:>9.3f} {r['commercial_mean']:>11.3f} "
            f"{r['delta']:>+8.3f}  {r['verdict']}"
        )
    for w in result["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append("")
    lines.append("PASS — commercial bonafide ships" if result["passed"]
                 else "FAIL — commercial bonafide does NOT preserve EER")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commercial-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=_DEFAULT_BASELINE)
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--collapse", type=float, default=0.001)
    args = ap.parse_args(argv)

    result = compare(
        aggregate(args.baseline_dir), aggregate(args.commercial_dir),
        band=args.band, collapse=args.collapse,
    )
    print(_render(result, args.band))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
