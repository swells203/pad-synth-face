"""Protocol every face attack module implements."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from pad_synth_core.ontology import Ontology


class FaceAttackModule(Protocol):
    name: str
    ontology: Ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]: ...

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray: ...
