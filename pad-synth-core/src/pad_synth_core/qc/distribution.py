"""Post-batch distribution-level QC over a finished manifest."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reason: str | None = None


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def coverage_report(manifest_path: Path) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    type_counts = Counter(r.get("attack_type") for r in rows)
    return {
        "total": len(rows),
        "attack_type_counts": dict(type_counts),
    }


def verify_identity_disjoint(manifest_path: Path) -> QCResult:
    rows = _read_manifest(manifest_path)
    split_to_ids: dict[str, set[str]] = {}
    for r in rows:
        split = r.get("split")
        if split is None:
            continue
        ident = r["bonafide_source"]["id"]
        split_to_ids.setdefault(split, set()).add(ident)
    splits = list(split_to_ids.items())
    for i, (a_name, a_ids) in enumerate(splits):
        for b_name, b_ids in splits[i + 1 :]:
            overlap = a_ids & b_ids
            if overlap:
                example = sorted(overlap)[0]
                return QCResult(
                    False,
                    f"identity {example!r} appears in both {a_name!r} and {b_name!r}",
                )
    return QCResult(True)
