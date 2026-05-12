"""Per-sample sanity checks. Cheap and run inline during generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reason: str | None = None


def check_image_basic(
    img: np.ndarray, expected_shape: tuple[int, int, int]
) -> QCResult:
    if img.shape != expected_shape:
        return QCResult(False, f"shape {img.shape} != expected {expected_shape}")
    if img.dtype.kind == "f" and not np.isfinite(img).all():
        return QCResult(False, "image contains NaN or Inf")
    if img.dtype.kind == "u" or img.dtype.kind == "i":
        as_u8 = img.astype(np.int32)
    else:
        if not np.isfinite(img).all():
            return QCResult(False, "image contains NaN or Inf")
        as_u8 = np.clip(img * 255.0, 0, 255).astype(np.int32)
    mean = float(as_u8.mean())
    std = float(as_u8.std())
    if std < 1.0:
        return QCResult(False, f"degenerate histogram (std={std:.3f}, mean={mean:.1f})")
    return QCResult(True)
