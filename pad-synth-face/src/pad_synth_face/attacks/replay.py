"""Phase 1 replay-attack simulator.

Pipeline (MVP):
  1. Display gamma forward (sRGB-ish)
  2. Subpixel grid attenuation (column-stripe attenuation modeling phone OLED/LCD)
  3. Moire pattern: 2D sinusoid at a frequency near the subpixel grid for beating
  4. Bezel masking: darken pixels in a bezel_pct frame
  5. Display gamma inverse
  6. Optional ambient_reflection low-frequency sheen overlay
  7. Viewing-angle skew: small affine shear

Refresh-rate banding and per-device subpixel-pattern shapes are simplified to
a single column-stripe model for Phase 1.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from pad_synth_core.ontology import Ontology


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(img**gamma, 0.0, 1.0)


def _subpixel_grid(h: int, w: int) -> np.ndarray:
    # Image-fraction pitch: at any image dim the column-stripe pattern
    # occupies the same visible angular fraction. At h=64 -> pitch=3
    # (pre-A1-bump back-compat); at h=224 -> pitch=11. See spec §4
    # (2026-05-29 resolution bump).
    pitch = max(1, math.floor(h / 64.0 * 3 + 0.5))  # round-half-up (not banker's)
    levels = np.array([0.92, 0.96, 0.90], dtype=np.float32)
    # Distribute pitch columns across 3 sub-pixel color stripes.
    # Red (0.92) always gets 1 column so it reappears exactly once per tile
    # period (enables period-detection via peak-search). Green (0.96) gets
    # pitch//3 columns; Blue (0.90) fills the remainder.  At pitch=3 this
    # gives [0.92, 0.96, 0.90] -- byte-identical to the pre-bump pattern.
    n_r = 1
    n_g = pitch // 3
    n_b = pitch - n_r - n_g
    expanded = np.repeat(levels, [n_r, n_g, n_b])  # length == pitch
    pattern = np.tile(
        expanded[None, :, None],
        (h, w // pitch + 1, 3),
    )[:, :w]
    return pattern.astype(np.float32)


def _moire(h: int, w: int, refresh_hz: int, rng: np.random.Generator) -> np.ndarray:
    # Image-fraction freq: bands-per-image-width stays constant across
    # resolutions. At h=64 the multiplier 64/h is 1.0 (pre-A1-bump back-
    # compat); at h=224 the freq is divided by 3.5. See spec §4
    # (2026-05-29 resolution bump).
    freq = (0.18 + (refresh_hz - 60) * 0.0015) * (64.0 / h)
    angle = float(rng.uniform(-0.4, 0.4))
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    pattern = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    return (1.0 + 0.04 * pattern).astype(np.float32)[:, :, None]


def _bezel_mask(h: int, w: int, bezel_pct: float) -> np.ndarray:
    inset_y = int(round(h * bezel_pct / 100.0))
    inset_x = int(round(w * bezel_pct / 100.0))
    mask = np.zeros((h, w, 1), dtype=np.float32)
    mask[inset_y : h - inset_y, inset_x : w - inset_x] = 1.0
    return mask


def _view_angle_shear(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    shear = np.tan(np.deg2rad(angle_deg)) * 0.15
    M = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


class ReplayAttack:
    name = "replay"

    def __init__(self, ontology: Ontology) -> None:
        assert ontology.attack_type == "replay"
        self.ontology = ontology

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.ontology.sample_params(rng)

    def simulate(
        self,
        bonafide: np.ndarray,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        h, w = bonafide.shape[:2]
        img = bonafide.astype(np.float32) / 255.0

        img = _apply_gamma(img, 2.2)
        img = img * _subpixel_grid(h, w)
        img = img * _moire(h, w, int(params["refresh_hz"]), rng)
        img = img * _bezel_mask(h, w, float(params["bezel_pct"]))
        img = _apply_gamma(img, 1.0 / 2.2)

        ambient = float(params["ambient_reflection"])
        if ambient > 0:
            yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
            sheen = (
                0.5
                * (1.0 + np.cos((xv + yv) / max(h, w) * np.pi))
                * ambient
            )[:, :, None]
            img = np.clip(img + sheen, 0.0, 1.0)

        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_u8 = _view_angle_shear(img_u8, float(params["viewing_angle"]))
        return img_u8
