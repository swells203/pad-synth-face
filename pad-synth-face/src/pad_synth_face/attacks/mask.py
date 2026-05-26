"""Phase 2 mask-attack simulator (worn paper / silicone / resin masks).

2D image-space approximation of a worn face mask. No real 3D geometry: the
"3D-ness" is faked with an analytic elliptical-dome shading field, a soft
specular term, eye/mouth aperture misregistration, a non-rigid drape warp,
and a perimeter seam.

Artifact discipline (the v2/v2.1 print lesson, designed in): the pipeline
stays in continuous float until the final uint8 cast -- NO binary
thresholding or colour quantisation anywhere -- and every spatial pattern is
per-sample jittered from the rng so Set A and Set B never share a fixed
geometry a detector could memorise.

mask_type selects a material-property bundle; the continuous ontology axes
modulate it per sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pad_synth_core.ontology import Ontology


class MaskAttack:
    name = "mask"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "mask"
        self.ontology = ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.ontology.sample_params(rng)

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Pass-through placeholder; physics added in Task 3.
        img = bonafide.astype(np.float32) / 255.0
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
