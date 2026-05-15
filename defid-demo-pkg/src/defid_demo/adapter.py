"""Browser rep payload -> defid-shaped (touch, key) event lists.

Privacy: only physical key `code` is used, solely to pair down/up; it is
consumed here and never stored or forwarded. No character content exists
in the payload by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RepPayload:
    pointer: list[dict[str, Any]]  # [{"x","y","ts"} ...]  ts in ms
    keys: list[dict[str, Any]]     # [{"code","phase","ts"} ...] phase down|up


def payload_to_session(p: RepPayload) -> tuple[list[dict], list[dict]]:
    touch: list[dict] = []
    if p.pointer:
        t0 = p.pointer[0]["ts"]
        for s in p.pointer:
            touch.append(
                {
                    "t": (s["ts"] - t0) / 1000.0,
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                }
            )

    open_down: dict[str, float] = {}
    completed: list[tuple[float, float]] = []  # (down_ts, up_ts)
    for ev in p.keys:
        code = ev["code"]
        if ev["phase"] == "down":
            open_down[code] = ev["ts"]
        elif ev["phase"] == "up" and code in open_down:
            completed.append((open_down.pop(code), ev["ts"]))
    completed.sort(key=lambda du: du[1])

    key: list[dict] = []
    if completed:
        k0 = completed[0][0]
        for down_ts, up_ts in completed:
            key.append({"t": (down_ts - k0) / 1000.0, "phase": "down"})
            key.append({"t": (up_ts - k0) / 1000.0, "phase": "up"})

    return touch, key
