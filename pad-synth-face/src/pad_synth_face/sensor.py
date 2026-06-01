"""Camera/lens/ISP pipeline applied after attack-specific simulation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SensorPreset:
    name: str
    iso_range: tuple[int, int]
    jpeg_qf_range: tuple[int, int]
    wb_k_range: tuple[int, int]
    vignette_strength: float
    # A2 capture-realism fields (2026-05-31)
    lens_k1_range: tuple[float, float]
    motion_blur_px_range: tuple[int, int]
    jpeg_passes_range: tuple[int, int]


MOBILE_FRONT_2024 = SensorPreset(
    name="mobile-front-2024",
    iso_range=(100, 800),
    jpeg_qf_range=(75, 95),
    wb_k_range=(4200, 6500),
    vignette_strength=0.35,
    lens_k1_range=(-0.10, 0.10),
    motion_blur_px_range=(1, 7),
    jpeg_passes_range=(1, 3),
)

WEBCAM_1080P = SensorPreset(
    name="webcam-1080p",
    iso_range=(200, 1600),
    jpeg_qf_range=(70, 92),
    wb_k_range=(3200, 6000),
    vignette_strength=0.20,
    lens_k1_range=(-0.05, 0.05),
    motion_blur_px_range=(1, 4),
    jpeg_passes_range=(1, 2),
)


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yv - cy) ** 2 + (xv - cx) ** 2)
    r_max = np.sqrt(cy**2 + cx**2)
    fall = 1.0 - strength * (r / r_max) ** 2
    return np.clip(img.astype(np.float32) * fall[:, :, None], 0, 255).astype(np.uint8)


def _white_balance(img: np.ndarray, kelvin: int) -> np.ndarray:
    # Cheap linear WB: warmer = boost R, cooler = boost B.
    t = (kelvin - 5400) / 1300.0  # ~[-1, 1]
    gains = np.array([1.0 - 0.10 * t, 1.0, 1.0 + 0.10 * t], dtype=np.float32)
    out = img.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def _lens_distort(img: np.ndarray, k1: float) -> np.ndarray:
    """Radial (Brown-Conrady k1-only) distortion via cv2.remap.

    k1=0 is identity. k1>0 is pincushion, k1<0 is barrel. Normalised radius
    `r` measured from image centre, displaced by r' = r * (1 + k1*r²).
    """
    h, w = img.shape[:2]
    if k1 == 0.0:
        return img.copy()
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Normalise so r=1 at the image corner (half-diagonal).
    r_norm = float(np.hypot(cy, cx))
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xv - cx) / r_norm
    dy = (yv - cy) / r_norm
    r2 = dx * dx + dy * dy
    factor = 1.0 + k1 * r2
    map_x = (dx * factor) * r_norm + cx
    map_y = (dy * factor) * r_norm + cy
    return cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _motion_blur(img: np.ndarray, length_px: int, angle_rad: float) -> np.ndarray:
    """Directional line-kernel blur. length_px=1 is identity.

    Draws a 1-px line through the centre of an (L, L) kernel at the given
    angle, normalises to sum 1, applies via cv2.filter2D.
    """
    L = int(length_px)
    if L <= 1:
        return img.copy()
    kernel = np.zeros((L, L), dtype=np.float32)
    cx = (L - 1) / 2.0
    cy = (L - 1) / 2.0
    half = (L - 1) / 2.0
    dx = np.cos(angle_rad) * half
    dy = np.sin(angle_rad) * half
    x0 = int(round(cx - dx))
    y0 = int(round(cy - dy))
    x1 = int(round(cx + dx))
    y1 = int(round(cy + dy))
    cv2.line(kernel, (x0, y0), (x1, y1), color=1.0, thickness=1)
    s = kernel.sum()
    if s <= 0.0:  # degenerate (shouldn't happen for L>=2, defensive)
        return img.copy()
    kernel /= s
    return cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)


def _noise(img: np.ndarray, iso: int, rng: np.random.Generator) -> np.ndarray:
    """Shot (signal-dependent) + read (fixed) noise.

    Shot: Poisson approximated as Gaussian with sigma=sqrt(signal),
    scaled by iso/800 * 0.5. Read: fixed-magnitude electronics floor.
    """
    signal = img.astype(np.float32)
    shot_sigma = np.sqrt(np.maximum(signal, 1.0)) * (iso / 800.0) * 0.5
    shot = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32) * shot_sigma
    read = rng.normal(0.0, 1.5, size=signal.shape).astype(np.float32)
    return np.clip(signal + shot + read, 0, 255).astype(np.uint8)


def _jpeg_roundtrip(img: np.ndarray, qf: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(qf))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def apply_sensor(
    img: np.ndarray, preset: SensorPreset, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    iso = int(rng.integers(preset.iso_range[0], preset.iso_range[1] + 1))
    kelvin = int(rng.integers(preset.wb_k_range[0], preset.wb_k_range[1] + 1))
    qf = int(rng.integers(preset.jpeg_qf_range[0], preset.jpeg_qf_range[1] + 1))

    out = _vignette(img, preset.vignette_strength)
    out = _white_balance(out, kelvin)
    out = _noise(out, iso, rng)
    out = _jpeg_roundtrip(out, qf)

    params = {"iso": iso, "wb_k": kelvin, "jpeg_qf": qf, "preset": preset.name}
    return out, params
