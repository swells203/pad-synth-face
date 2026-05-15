"""Per-rep capture QC. Reuses defid.qc.QCResult (do not redefine)."""

from __future__ import annotations

import math

from defid.qc import QCResult

_MIN_TOUCH = 8   # a real swipe; below this it's a tap/accident
_MIN_KEYS = 2    # at least two completed keystrokes for dwell+flight


def check_rep(touch: list[dict], key: list[dict]) -> QCResult:
    if len(touch) < _MIN_TOUCH:
        return QCResult(False, f"too few touch points (<{_MIN_TOUCH}) — swipe, don't tap")
    downs = [k for k in key if k["phase"] == "down"]
    if len(downs) < _MIN_KEYS:
        return QCResult(False, f"too few keystrokes (<{_MIN_KEYS}) — type the passphrase")
    for i in range(1, len(touch)):
        if touch[i]["t"] - touch[i - 1]["t"] <= 0:
            return QCResult(False, "non-monotonic touch timestamps")
    for ev in touch:
        for k in ("t", "x", "y"):
            if not isinstance(ev[k], (int, float)) or not math.isfinite(ev[k]):
                return QCResult(False, f"touch.{k} not finite")
    return QCResult(True)
