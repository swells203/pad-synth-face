"""Attack-parameter ontology with literature-citation enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator


class OntologyLintError(ValueError):
    pass


class Provenance(BaseModel):
    paper: str
    doi: str | None = None
    url: str | None = None


class Axis(BaseModel):
    type: str
    provenance: Provenance
    values: list[Any] | None = None
    weights: list[float] | None = None
    low: float | None = None
    high: float | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "Axis":
        if self.type == "categorical":
            if not self.values or not self.weights:
                raise OntologyLintError("categorical axis needs values and weights")
            if len(self.values) != len(self.weights):
                raise OntologyLintError("values and weights length mismatch")
        elif self.type == "uniform":
            if self.low is None or self.high is None:
                raise OntologyLintError("uniform axis needs low and high")
        else:
            raise OntologyLintError(f"unknown axis type {self.type!r}")
        return self


class Ontology(BaseModel):
    version: str
    attack_type: str
    axes: dict[str, Axis]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, axis in self.axes.items():
            if axis.type == "categorical":
                idx = rng.choice(len(axis.values), p=np.array(axis.weights, dtype=float))
                value = axis.values[int(idx)]
            else:  # uniform
                value = float(rng.uniform(axis.low, axis.high))
            out[name] = value
        return out


def load_ontology(path: Path) -> Ontology:
    raw = yaml.safe_load(Path(path).read_text())
    # Pre-lint: catch missing provenance with a clear error referencing the axis.
    axes = raw.get("axes", {})
    for axis_name, axis_data in axes.items():
        if "provenance" not in axis_data:
            raise OntologyLintError(
                f"axis {axis_name!r} in {path}: missing required 'provenance' field"
            )
    return Ontology.model_validate(raw)
