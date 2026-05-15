"""Per-session behavioral-plausibility QC. Cheap, runs inline during
generation. Mirrors the PAD project's per-sample QC contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from defid.session import BehavioralSession

_MAX_HUMAN_TOUCH_SPEED = 50000.0  # px/s — generous upper plausibility bound


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reason: str | None = None


def check_session(s: BehavioralSession) -> QCResult:
    if len(s.touch) < 3:
        return QCResult(False, "too few touch events (<3)")
    if len(s.motion) < 3:
        return QCResult(False, "too few motion events (<3)")

    for arr_name, arr, keys in (
        ("touch", s.touch, ("t", "x", "y")),
        ("motion", s.motion, ("t", "ax", "ay", "az")),
    ):
        for ev in arr:
            for k in keys:
                v = ev[k]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    return QCResult(False, f"{arr_name}.{k} not finite")

    for i in range(1, len(s.touch)):
        dt = s.touch[i]["t"] - s.touch[i - 1]["t"]
        if dt <= 0:
            return QCResult(False, "non-monotonic touch timestamps")
        dx = s.touch[i]["x"] - s.touch[i - 1]["x"]
        dy = s.touch[i]["y"] - s.touch[i - 1]["y"]
        speed = math.hypot(dx, dy) / dt
        if speed > _MAX_HUMAN_TOUCH_SPEED:
            return QCResult(False, f"implausible touch speed {speed:.0f}")

    return QCResult(True)
