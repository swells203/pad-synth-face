"""Session -> fixed-length feature vector. Versioned feature schema."""

from __future__ import annotations

import numpy as np

from defid.session import BehavioralSession

FEATURE_SCHEMA_VERSION = "defid-feat@0.1.0"
FEATURE_NAMES = [
    "touch_speed_mean",
    "touch_speed_std",
    "touch_curvature_mean",
    "touch_jitter",
    "inter_touch_interval_mean",
    "key_dwell_mean",
    "key_dwell_std",
    "key_flight_mean",
    "key_flight_std",
    "key_paste_ratio",
    "accel_mag_mean",
    "tremor_std",
    "motion_touch_coupling",
    "touch_without_motion_ratio",
]


def _safe(a: np.ndarray, fn, default: float = 0.0) -> float:
    return float(fn(a)) if a.size else default


def extract_features(s: BehavioralSession) -> np.ndarray:
    tx = np.array([[p["t"], p["x"], p["y"]] for p in s.touch], dtype=np.float64)
    if tx.shape[0] >= 2:
        d = np.linalg.norm(np.diff(tx[:, 1:], axis=0), axis=1)
        dt = np.maximum(np.diff(tx[:, 0]), 1e-3)
        speed = d / dt
        ang = np.arctan2(np.diff(tx[:, 2]), np.diff(tx[:, 1]))
        curv = np.abs(np.diff(ang)) if ang.size >= 2 else np.array([0.0])
        iti = np.diff(tx[:, 0])
    else:
        speed = curv = iti = np.array([0.0])

    downs = [k["t"] for k in s.key if k["phase"] == "down"]
    ups = [k["t"] for k in s.key if k["phase"] == "up"]
    m = min(len(downs), len(ups))
    dwell = np.array([ups[i] - downs[i] for i in range(m)]) if m else np.array([0.0])
    flight = (
        np.array([downs[i + 1] - ups[i] for i in range(m - 1)])
        if m >= 2
        else np.array([0.0])
    )
    paste_ratio = float(np.mean(flight < 0.005)) if flight.size else 0.0

    mo = np.array([[e["ax"], e["ay"], e["az"]] for e in s.motion], dtype=np.float64)
    amag = np.linalg.norm(mo, axis=1) if mo.size else np.array([0.0])
    tremor = _safe(amag, np.std)
    coupling = (
        float(np.corrcoef(amag[: speed.size], speed[: amag.size])[0, 1])
        if min(amag.size, speed.size) >= 2
        else 0.0
    )
    coupling = 0.0 if not np.isfinite(coupling) else coupling
    twm = 1.0 if mo.size == 0 and tx.size > 0 else 0.0

    vec = np.array(
        [
            _safe(speed, np.mean),
            _safe(speed, np.std),
            _safe(curv, np.mean),
            _safe(speed, lambda a: np.std(np.diff(a)) if a.size >= 2 else 0.0),
            _safe(iti, np.mean),
            _safe(dwell, np.mean),
            _safe(dwell, np.std),
            _safe(flight, np.mean),
            _safe(flight, np.std),
            paste_ratio,
            _safe(amag, np.mean),
            tremor,
            coupling,
            twm,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
