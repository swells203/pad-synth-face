"""Camera/lens/ISP pipeline applied after attack-specific simulation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

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


def _noise(img: np.ndarray, iso: int, rng: np.random.Generator) -> np.ndarray:
    sigma = 0.5 + (iso / 800.0) * 4.0
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


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
