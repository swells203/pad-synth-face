"""Deterministic seed derivation and RNG construction.

The orchestrator owns a single master seed. Every sample's randomness is
derived from (master_seed, modality, attack_type, sample_index) via SHA-256
so the derivation is reproducible across Python versions and machines.
"""

import hashlib

import numpy as np


def derive_sample_seed(
    master_seed: int, modality: str, attack_type: str, sample_index: int
) -> int:
    parts = [str(master_seed), modality, attack_type, str(sample_index)]
    payload = "|".join(f"{len(p)}:{p}" for p in parts).encode()
    digest = hashlib.sha256(payload).digest()
    # Truncate to uint32 — numpy's default_rng accepts seeds in [0, 2**32).
    return int.from_bytes(digest[:4], "big")


def sample_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
