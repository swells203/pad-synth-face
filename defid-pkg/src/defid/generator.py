"""Deterministic synthetic behavioral-session generator.

A subject has a stable motor profile derived from a per-subject seed. A
session adds per-session variation around that profile. Label semantics:
  genuine  — the subject's own profile + natural jitter
  imposter — a *different* profile (different subject hash) + jitter
  bot      — machine-regular timing, near-zero jitter, no motion coupling

domain="b" applies a fixed parameter shift — the synthetic cross-domain
proxy (mirrors PAD Phase 1.5's Set B).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pad_synth_core.ontology import load_ontology
from pad_synth_core.rng import derive_sample_seed

GENERATOR_VERSION = "defid-gen@0.1.0"
_N_TOUCH = 60
_N_KEY = 25
_N_MOTION = 120


def _subject_seed(subject_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(subject_id.encode()).digest()[:4], "big"
    )


def _domain_scale(domain: str) -> float:
    return 1.0 if domain == "a" else 1.35


def generate_session(
    label: str,
    subject_id: str,
    seed: int,
    ontology_dir: Path,
    domain: str = "a",
):
    from defid.session import BehavioralSession

    if label not in ("genuine", "imposter", "bot"):
        raise ValueError(f"bad label {label!r}")

    touch_ont = load_ontology(Path(ontology_dir) / "touch.yaml")
    key_ont = load_ontology(Path(ontology_dir) / "keystroke.yaml")
    motion_ont = load_ontology(Path(ontology_dir) / "motion.yaml")

    profile_id = subject_id if label != "imposter" else subject_id + "-imp"
    profile_rng = np.random.default_rng(_subject_seed(profile_id))
    tp = touch_ont.sample_params(profile_rng)
    kp = key_ont.sample_params(profile_rng)
    mp = motion_ont.sample_params(profile_rng)

    scale = _domain_scale(domain)
    sess_seed = derive_sample_seed(seed, "defid", label, _subject_seed(subject_id))
    rng = np.random.default_rng(sess_seed)

    is_bot = label == "bot"
    jitter = 0.02 if is_bot else float(tp["touch_jitter"])
    speed = float(tp["touch_speed_mean"]) * scale
    speed_sd = 1.0 if is_bot else float(tp["touch_speed_std"])
    iti = float(tp["inter_touch_interval_ms"]) / 1000.0

    t = 0.0
    x, y = 100.0, 100.0
    touch = []
    for _ in range(_N_TOUCH):
        step_speed = speed if is_bot else max(rng.normal(speed, speed_sd), 10.0)
        ang = 0.0 if is_bot else rng.normal(0.0, float(tp["touch_curvature"]))
        x += step_speed * 0.01 * np.cos(ang) + rng.normal(0.0, jitter)
        y += step_speed * 0.01 * np.sin(ang) + rng.normal(0.0, jitter)
        t += iti if is_bot else max(rng.normal(iti, iti * 0.3), 0.01)
        touch.append({"t": round(t, 5), "x": round(float(x), 4),
                       "y": round(float(y), 4), "phase": "move"})

    dwell = float(kp["key_dwell_mean"]) / 1000.0
    dwell_sd = 0.001 if is_bot else float(kp["key_dwell_std"]) / 1000.0
    flight = float(kp["key_flight_mean"]) / 1000.0
    flight_sd = 0.001 if is_bot else float(kp["key_flight_std"]) / 1000.0
    tk = 0.0
    key = []
    for _ in range(_N_KEY):
        d = dwell if is_bot else max(rng.normal(dwell, dwell_sd), 0.005)
        key.append({"t": round(tk, 5), "phase": "down", "field": "f1"})
        tk += d
        key.append({"t": round(tk, 5), "phase": "up", "field": "f1"})
        tk += flight if is_bot else max(rng.normal(flight, flight_sd), 0.01)

    amag = float(mp["accel_mag_mean"])
    tremor = 0.0 if is_bot else float(mp["tremor_std"])
    coupling = 0.0 if is_bot else float(mp["motion_touch_coupling"])
    motion = []
    for k in range(_N_MOTION):
        tm = k * (t / _N_MOTION if t > 0 else 0.01)
        base = amag + rng.normal(0.0, tremor)
        couple = coupling * (0.3 if (k % 5 == 0) else 0.0)
        motion.append({
            "t": round(tm, 5),
            "ax": round(float(rng.normal(0.0, tremor) + couple), 5),
            "ay": round(float(rng.normal(0.0, tremor) + couple), 5),
            "az": round(float(base), 5),
        })

    # Fix created_at to epoch so model_dump_json() is fully deterministic.
    created_at = datetime(1970, 1, 1, tzinfo=timezone.utc)

    return BehavioralSession(
        session_id=f"{label}-{subject_id}-{seed}-{domain}",
        label=label,
        subject_id=subject_id,
        touch=touch,
        key=key,
        motion=motion,
        ontology_version=touch_ont.version,
        generator_version=GENERATOR_VERSION,
        seed=sess_seed,
        created_at=created_at,
    )
