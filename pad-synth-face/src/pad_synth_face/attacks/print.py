"""Phase 2 print-attack simulator (v2 physics).

Pipeline:
  1. Paper-color tint (matte/glossy/photo per ontology)
  2. Halftone — per-channel AM dot screening at standard rosette angles
     (C=15°, M=75°, Y=0°, K=45°); dot-cell frequency driven by print_dpi
     with per-sample jitter on cell-size (±10%), angle (σ=3°), and
     sub-pixel offset to break deterministic-pattern artifacts (v2.1).
  3. ICC profile transform — gamut compression + white-point shift +
     tone gamma, parameterized per paper_type and scaled by
     icc_profile_strength.
  4. Paper-texture multiplicative noise (uses RNG).
  5. Perspective warp simulating a tilted printed page (uses RNG).
  6. Optional cutout (eyes / eyes+mouth).

Anisotropic specular highlights remain explicitly deferred to a follow-up.
The v1 single-tier-physics version is captured by ontology_version
2026-05-11; the v2 deterministic-halftone version by 2026-05-22; this
module (v2.1, jittered halftone) corresponds to ontology_version
2026-05-23.
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


# --- v2 physics: halftoning ---------------------------------------------------

# CMYK rosette angles (degrees) per standard 4-color print convention.
_HALFTONE_ANGLES_DEG: tuple[float, float, float, float] = (15.0, 75.0, 0.0, 45.0)


def _to_cmyk(rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] (H,W,3) -> CMYK float [0,1] (H,W,4). No profile math."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.where(k < 1.0, 1.0 - k, 1.0)
    c = np.where(k < 1.0, (1.0 - r - k) / denom, 0.0)
    m = np.where(k < 1.0, (1.0 - g - k) / denom, 0.0)
    y = np.where(k < 1.0, (1.0 - b - k) / denom, 0.0)
    return np.stack([c, m, y, k], axis=-1).astype(np.float32)


def _inv_cmyk(cmyk: np.ndarray) -> np.ndarray:
    """CMYK float [0,1] (H,W,4) -> RGB float [0,1] (H,W,3)."""
    c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _dot_screen(
    h: int,
    w: int,
    cell_px: float,
    angle_deg: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """2D rotated cosine dot screen, range [0,1]. Deterministic — no RNG.
    Optional (dx, dy) shift the screen origin by sub-pixel amounts."""
    theta = np.deg2rad(angle_deg)
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xv - float(dx)
    ys = yv - float(dy)
    xp = xs * np.cos(theta) + ys * np.sin(theta)
    yp = -xs * np.sin(theta) + ys * np.cos(theta)
    grid = np.cos(2.0 * np.pi * xp / cell_px) * np.cos(2.0 * np.pi * yp / cell_px)
    return (0.5 + 0.5 * grid).astype(np.float32)


def _halftone_channel(
    channel: np.ndarray,
    cell_px: float,
    angle_deg: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Binary halftone: pixel ON where channel value > screen threshold."""
    screen = _dot_screen(channel.shape[0], channel.shape[1], cell_px, angle_deg, dx, dy)
    return (channel > screen).astype(np.float32)


def _apply_halftone(
    rgb: np.ndarray,
    print_dpi: int | float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-channel AM halftoning at the standard rosette angles.

    rgb: float [0,1] (H,W,3); print_dpi drives the base dot-cell frequency.

    When rng is None (v2 behavior): deterministic screen, no jitter.
    When rng is provided (v2.1 behavior): per-channel per-sample jitter from
    the spec — cell-size ~U(0.90, 1.10), angle ~N(0, 3°), sub-pixel offset
    (dx, dy) ~U(-cell/2, +cell/2). The draw order per channel C/M/Y/K is
    (k, Δθ, dx, dy) — see spec §3 and the plan reference block.

    Returns float [0,1] (H,W,3).
    """
    # Image-fraction-based: a halftone cell occupies the same fraction of
    # image area regardless of resolution (preserves real-world print
    # geometry across capture resolutions). The constant 0.125 calibrates
    # so that at image dim 64 and print_dpi 150 the formula reproduces the
    # pre-bump cell_px=8 exactly. See spec §4 (2026-05-29 resolution bump).
    image_dim = rgb.shape[0]
    base_cell = max(2.0, image_dim * 0.125 * (150.0 / float(print_dpi)))
    cmyk = _to_cmyk(rgb)
    out = np.empty_like(cmyk)
    for i, base_angle in enumerate(_HALFTONE_ANGLES_DEG):
        if rng is None:
            cell_px, angle, dx, dy = base_cell, base_angle, 0.0, 0.0
        else:
            k = float(rng.uniform(0.90, 1.10))
            cell_px = max(2.0, base_cell * k)
            angle = base_angle + float(rng.normal(0.0, 3.0))
            dx = float(rng.uniform(-cell_px / 2.0, cell_px / 2.0))
            dy = float(rng.uniform(-cell_px / 2.0, cell_px / 2.0))
        out[..., i] = _halftone_channel(cmyk[..., i], cell_px, angle, dx, dy)
    return _inv_cmyk(out)


# --- v2 physics: ICC profile simulation ---------------------------------------

# Per-paper-type tuple: (gamut_compression, (Δx, Δy) white-point shift, tone_gamma).
# Parameters per spec §4.1; references: Lukac & Plataniotis (eds.), 2007;
# Marini & Rizzi 2000 (white-point handling).
_ICC_PARAMS: dict[str, tuple[float, tuple[float, float], float]] = {
    "matte":  (0.12, (+0.012, +0.008), 1.10),
    "glossy": (0.05, (+0.002, +0.001), 0.95),
    "photo":  (0.03, (-0.003, -0.002), 0.92),
}


def _apply_icc(rgb: np.ndarray, paper_type: str, strength: float) -> np.ndarray:
    """sRGB-space parameterized print-profile transform.

    rgb: float [0,1] (H,W,3); paper_type in {matte, glossy, photo};
    strength scales the gamut-compression effect [0,1]. Returns float [0,1].
    """
    gamut, (dx, dy), gamma = _ICC_PARAMS[paper_type]
    out = rgb.astype(np.float32, copy=True)

    # 1. Gamut compression: pull every pixel toward middle gray by c.
    c = float(gamut) * float(strength)
    out = (1.0 - c) * out + c * 0.5

    # 2. White-point shift: chromaticity-to-RGB approximation (clipped later).
    out[..., 0] += float(dx) * 0.5
    out[..., 1] += float(dy) * 0.5
    out[..., 2] -= (float(dx) + float(dy)) * 0.25

    # 3. Tone curve: out := out ** (1/gamma).
    out = np.clip(out, 0.0, 1.0) ** (1.0 / float(gamma))

    return out.astype(np.float32)


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
        # Linear-RGB float space for the float-domain stages.
        img = bonafide.astype(np.float32) / 255.0

        # Paper-color tint (matte/glossy/photo).
        tint = np.array(_PAPER_TINTS[params["paper_type"]], dtype=np.float32)
        img = img * tint

        # v2.1: halftone with per-sample jitter (driven by print_dpi + rng).
        img = _apply_halftone(img, params["print_dpi"], rng)

        # v2: ICC profile (keyed by paper_type, scaled by strength).
        img = _apply_icc(
            img,
            params["paper_type"],
            float(params["icc_profile_strength"]),
        )

        # Paper-texture multiplicative noise (uses RNG).
        texture = _paper_texture(img.shape[0], img.shape[1], rng)
        img = img * texture

        # Back to uint8 for the spatial-domain stages.
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = _perspective_warp(img, params["tilt_degrees"], rng)
        img = _apply_cutout(img, params["cutout"])
        return img
