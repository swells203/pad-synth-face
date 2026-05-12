"""Deterministic work-item enumeration.

A work item is `(sample_id, bonafide_id, attack_type, seed)`. Enumeration is
pure-functional and deterministic from `(master_seed, modality, bonafide_ids,
attack_weights, samples_per_bonafide)`.

Item generation logic:
  - For each bonafide_id, emit `samples_per_bonafide` items.
  - Attack type for each item is drawn from the weighted distribution using
    the derived seed for that item, so re-runs reproduce the same assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pad_synth_core.rng import derive_sample_seed


@dataclass(frozen=True)
class WorkItem:
    sample_id: str
    bonafide_id: str
    attack_type: str
    seed: int


def enumerate_work_items(
    master_seed: int,
    modality: str,
    bonafide_ids: list[str],
    attack_weights: dict[str, float],
    samples_per_bonafide: int,
) -> list[WorkItem]:
    attack_names = sorted(attack_weights.keys())
    weights = np.array(
        [attack_weights[n] for n in attack_names], dtype=np.float64
    )
    weights = weights / weights.sum()

    items: list[WorkItem] = []
    counter = 0
    for bid in sorted(bonafide_ids):
        for sub in range(samples_per_bonafide):
            seed = derive_sample_seed(master_seed, modality, "_dispatch", counter)
            rng = np.random.default_rng(seed)
            idx = int(rng.choice(len(attack_names), p=weights))
            attack = attack_names[idx]
            sample_seed = derive_sample_seed(master_seed, modality, attack, counter)
            sid = f"{modality}-{attack}-{counter:08d}"
            items.append(WorkItem(sid, bid, attack, sample_seed))
            counter += 1
    return items
