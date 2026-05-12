"""Phase 1 print-attack simulator.

Pipeline (MVP):
  1. Paper-color tint (matte/glossy/photo per ontology)
  2. Paper-texture multiply (procedural grain)
  3. Perspective warp simulating a tilted printed page
  4. Optional cutout (eyes / eyes+mouth) by zeroing pixels in the cut regions

The DPI axis is currently informational (recorded in params, not yet used to
band-limit). Halftoning, ICC profiling, and anisotropic specular highlights
are explicitly Phase 2 work.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pad_synth_core.ontology import Ontology


_PAPER_TINTS: dict[str, tuple[float, float, float]] = {
    "matte": (0.96, 0.95, 0.92),
    "glossy": (1.02, 1.02, 1.04),
    "photo": (1.00, 0.99, 0.97),
}


def _paper_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=1.0, scale=0.03, size=(h, w, 1))
    return np.clip(noise, 0.85, 1.10).astype(np.float32)


def _perspective_warp(
    img: np.ndarray, tilt_degrees: float, rng: np.random.Generator
) -> np.ndarray:
    h, w = img.shape[:2]
    shift = int(abs(tilt_degrees) / 25.0 * (w * 0.10))
    sign = 1 if tilt_degrees >= 0 else -1
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array(
        [
            [shift * sign, 0],
            [w - shift * sign, shift // 2],
            [w, h],
            [0, h - shift // 2],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _apply_cutout(img: np.ndarray, cutout: str) -> np.ndarray:
    if cutout == "none":
        return img
    out = img.copy()
    h, w = out.shape[:2]
    # Cheap fixed-position cutouts; consistent with "wearable print" attacks.
    eye_y1, eye_y2 = int(h * 0.30), int(h * 0.45)
    eye_x_left = (int(w * 0.20), int(w * 0.40))
    eye_x_right = (int(w * 0.60), int(w * 0.80))
    out[eye_y1:eye_y2, eye_x_left[0] : eye_x_left[1]] = 0
    out[eye_y1:eye_y2, eye_x_right[0] : eye_x_right[1]] = 0
    if cutout == "eyes_mouth":
        m_y1, m_y2 = int(h * 0.62), int(h * 0.78)
        m_x1, m_x2 = int(w * 0.35), int(w * 0.65)
        out[m_y1:m_y2, m_x1:m_x2] = 0
    return out


class PrintAttack:
    name = "print"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "print"
        self.ontology = ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.ontology.sample_params(rng)

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        img = bonafide.astype(np.float32) / 255.0

        tint = np.array(_PAPER_TINTS[params["paper_type"]], dtype=np.float32)
        img = img * tint

        texture = _paper_texture(img.shape[0], img.shape[1], rng)
        img = img * texture

        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = _perspective_warp(img, params["tilt_degrees"], rng)
        img = _apply_cutout(img, params["cutout"])
        return img
