"""Phase 2 mask-attack simulator (worn paper / silicone / resin masks).

2D image-space approximation of a worn face mask. No real 3D geometry: the
"3D-ness" is faked with an analytic elliptical-dome shading field, a soft
Blinn-Phong specular term, eye/mouth aperture misregistration, a non-rigid
drape warp, and a perimeter seam.

Artifact discipline (the v2/v2.1 print lesson, designed in): the pipeline
stays in continuous float until the final uint8 cast -- NO binary
thresholding or colour quantisation anywhere -- and every spatial pattern is
per-sample jittered from the rng so Set A and Set B never share a fixed
geometry a detector could memorise.

mask_type selects a material-property bundle (colour cast, specular scale,
texture-loss sigma, subsurface tint); the continuous ontology axes modulate
it per sample. This module corresponds to mask ontology_version 2026-05-22.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pad_synth_core.ontology import Ontology


@dataclass(frozen=True)
class MaskMaterial:
    color_cast: tuple[float, float, float]
    specular_scale: float
    texture_loss_sigma: float
    subsurface_tint: tuple[float, float, float]


# Material constants only; ALL per-sample variation comes from the jittered
# ontology axes, never from this table. Values informed by SMAD (silicone
# gloss/translucency) and HiFiMask (paper/resin appearance) -- see ontology
# provenance.
_MASK_MATERIALS: dict[str, MaskMaterial] = {
    "paper":    MaskMaterial((0.98, 0.97, 0.94), 0.15, 0.6, (0.00, 0.00, 0.00)),
    "silicone": MaskMaterial((1.02, 0.98, 0.95), 0.55, 1.0, (0.06, 0.02, 0.02)),
    "resin":    MaskMaterial((0.97, 0.98, 1.03), 0.85, 1.4, (0.00, 0.00, 0.00)),
}


def _texture_loss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass: masks lack fine skin-pore detail. Continuous."""
    k = max(3, 2 * int(np.ceil(3.0 * sigma)) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def _dome_normals(h: int, w: int) -> np.ndarray:
    """Unit normals (H,W,3) of an elliptical dome over the frame. Image-space
    parametric field -- NOT a per-face depth estimate."""
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    nx = (xv - cx) / (w * 0.5)
    ny = (yv - cy) / (h * 0.5)
    z = np.sqrt(np.clip(1.0 - nx**2 - ny**2, 0.0, 1.0))
    n = np.stack([nx, ny, z], axis=-1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def _light_dir(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit light-direction vector from azimuth/elevation in degrees."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float32,
    )


def _shading(img: np.ndarray, normals: np.ndarray, light: np.ndarray,
             strength: float = 0.35) -> np.ndarray:
    """Lambertian shading gradient lit by `light`. Output clipped to [0,1]."""
    lam = np.clip((normals * light).sum(axis=-1), 0.0, 1.0)
    factor = (1.0 - strength) + strength * lam[..., None]
    return np.clip(img * factor, 0.0, 1.0)


def _specular(img: np.ndarray, normals: np.ndarray, light: np.ndarray,
              intensity: float) -> np.ndarray:
    """Soft Blinn-Phong highlight; view fixed at +z. Continuous additive."""
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = light + view
    half = half / (np.linalg.norm(half) + 1e-6)
    spec = np.clip((normals * half).sum(axis=-1), 0.0, 1.0) ** 32
    return np.clip(img + intensity * spec[..., None], 0.0, 1.0)


def _aperture_mismatch(img: np.ndarray, offset_px: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Soft darkening at eye/mouth regions, centre offset by offset_px in a
    jittered direction (models the wearer's features not lining up)."""
    h, w = img.shape[:2]
    angle = float(rng.uniform(-np.pi, np.pi))
    dy = offset_px * np.sin(angle)
    dx = offset_px * np.cos(angle)
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    factor = np.ones((h, w, 1), dtype=np.float32)
    for cy_f, cx_f, ry_f, rx_f in (
        (0.36, 0.30, 0.06, 0.10),  # left eye
        (0.36, 0.70, 0.06, 0.10),  # right eye
        (0.70, 0.50, 0.08, 0.16),  # mouth
    ):
        cy = cy_f * h + dy
        cx = cx_f * w + dx
        r = ((yv - cy) / (ry_f * h)) ** 2 + ((xv - cx) / (rx_f * w)) ** 2
        factor[..., 0] *= 1.0 - 0.5 * np.exp(-r)
    return img * factor


def _drape_warp(img: np.ndarray, amount: float,
                rng: np.random.Generator) -> np.ndarray:
    """Mild non-rigid perspective warp for the mask draping over the face."""
    h, w = img.shape[:2]
    m = amount * 0.08 * w
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + rng.uniform(-m, m, size=(4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _seam(img: np.ndarray, visibility: float,
          rng: np.random.Generator) -> np.ndarray:
    """Darkened elliptical ring at the mask perimeter, with per-sample
    jittered centre and radii so the seam geometry is not fixed across
    samples."""
    h, w = img.shape[:2]
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (h - 1) / 2.0 + float(rng.uniform(-0.04, 0.04)) * h
    cx = (w - 1) / 2.0 + float(rng.uniform(-0.04, 0.04)) * w
    ry = h * 0.46 * float(rng.uniform(0.95, 1.05))
    rx = w * 0.40 * float(rng.uniform(0.95, 1.05))
    r = np.sqrt(((yv - cy) / ry) ** 2 + ((xv - cx) / rx) ** 2)
    ring = np.exp(-((r - 1.0) ** 2) / (2.0 * 0.08**2))
    factor = 1.0 - visibility * 0.5 * ring[..., None]
    return img * factor


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
        h, w = bonafide.shape[:2]
        img = bonafide.astype(np.float32) / 255.0
        mat = _MASK_MATERIALS[params["mask_type"]]

        # 1. Texture loss (jittered sigma).
        sigma = max(0.3, mat.texture_loss_sigma * float(rng.uniform(0.85, 1.15)))
        img = _texture_loss(img, sigma)

        # 2. Material colour cast + subsurface tint (continuous, no quantise).
        cast = np.array(mat.color_cast, dtype=np.float32)
        sub = np.array(mat.subsurface_tint, dtype=np.float32)
        img = np.clip(img * cast + sub, 0.0, 1.0)

        # 3 + 4. Pseudo-3D shading and specular from the dome normals.
        normals = _dome_normals(h, w)
        light = _light_dir(
            float(params["light_azimuth_deg"]),
            float(params["light_elevation_deg"]),
        )
        img = _shading(img, normals, light)
        spec_intensity = (
            float(params["specular_strength"])
            * mat.specular_scale
            * float(rng.uniform(0.85, 1.15))
        )
        img = _specular(img, normals, light, spec_intensity)

        # 5. Aperture mismatch (jittered offset direction).
        img = _aperture_mismatch(img, float(params["aperture_misalignment_px"]), rng)

        # 6. Drape warp.
        img = _drape_warp(img, float(params["surface_warp"]), rng)

        # 7. Perimeter seam.
        img = _seam(img, float(params["seam_visibility"]), rng)

        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
