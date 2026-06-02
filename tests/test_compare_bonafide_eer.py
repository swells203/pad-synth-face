"""Verdict logic for compare_bonafide_eer: matched-scale A/B EER delta."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compare_bonafide_eer",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_bonafide_eer.py",
)
cbe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cbe)


def _write_cell(d: Path, cap: str, dlevel: str, seed: int, eer: float,
               n_train: int = 384, n_val: int = 1024) -> None:
    runs = d / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{cap}_{dlevel}_{seed}.json").write_text(json.dumps({
        "capacity": cap, "data_level": dlevel, "seed": seed,
        "eer_cross_domain": eer, "n_train": n_train,
        "n_val_cross_domain": n_val,
    }))


def test_aggregate_groups_by_cell(tmp_path):
    d = tmp_path / "base"
    _write_cell(d, "L4", "D3", 0, 0.05)
    _write_cell(d, "L4", "D3", 1, 0.07)
    agg = cbe.aggregate(d)
    assert ("L4", "D3") in agg
    cell = agg[("L4", "D3")]
    assert abs(cell["mean"] - 0.06) < 1e-9
    assert cell["n_train"] == 384


def test_compare_passes_when_delta_small(tmp_path):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    for seed in (0, 1, 2):
        _write_cell(base, "L4", "D3", seed, 0.06)
        _write_cell(comm, "L4", "D3", seed, 0.07)  # delta 0.01 < band
    result = cbe.compare(cbe.aggregate(base), cbe.aggregate(comm),
                         band=0.03, collapse=0.001)
    assert result["passed"] is True
    assert result["rows"][0]["verdict"] == "ok"
    assert result["warnings"] == []


def test_compare_fails_on_large_delta_and_collapse(tmp_path):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    for seed in (0, 1, 2):
        _write_cell(base, "L4", "D2", seed, 0.06)
        _write_cell(comm, "L4", "D2", seed, 0.15)  # delta 0.09 > band
        _write_cell(base, "L4", "D3", seed, 0.06)
        _write_cell(comm, "L4", "D3", seed, 0.0005)  # collapsed
    result = cbe.compare(cbe.aggregate(base), cbe.aggregate(comm),
                         band=0.03, collapse=0.001)
    assert result["passed"] is False
    verdicts = {(r["capacity"], r["data_level"]): r["verdict"] for r in result["rows"]}
    assert verdicts[("L4", "D2")] == "delta_exceeds_band"
    assert verdicts[("L4", "D3")] == "collapsed"


def test_compare_warns_on_scale_mismatch(tmp_path):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    _write_cell(base, "L4", "D3", 0, 0.06, n_train=384)
    _write_cell(comm, "L4", "D3", 0, 0.06, n_train=48)  # different scale
    result = cbe.compare(cbe.aggregate(base), cbe.aggregate(comm),
                         band=0.03, collapse=0.001)
    assert any("scale" in w.lower() for w in result["warnings"])


def test_main_exits_nonzero_on_fail(tmp_path, capsys):
    base = tmp_path / "base"
    comm = tmp_path / "comm"
    _write_cell(base, "L4", "D3", 0, 0.06)
    _write_cell(comm, "L4", "D3", 0, 0.20)
    rc = cbe.main(["--baseline-dir", str(base), "--commercial-dir", str(comm)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
