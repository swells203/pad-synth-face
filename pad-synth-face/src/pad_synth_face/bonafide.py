"""Bonafide-face loaders.

The fixture-shaped on-disk layout is `<root>/<identity_id>/<sample_index>.png`.
The DigiFace-1M release follows the same identity-per-directory shape, so the
same loader implementation works for both fixture and real data in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BonafideSample:
    identity: str
    path: Path


class DigiFaceLoader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def list_identities(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def samples_for_identity(self, identity: str) -> list[BonafideSample]:
        identity_dir = self.root / identity
        return [
            BonafideSample(identity=identity, path=p)
            for p in sorted(identity_dir.glob("*.png"))
        ]

    def load(self, sample: BonafideSample) -> np.ndarray:
        img = Image.open(sample.path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def identity_disjoint_split(
        self, seed: int, ratios: tuple[float, float, float]
    ) -> tuple[list[str], list[str], list[str]]:
        ids = self.list_identities()
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids)).tolist()
        shuffled = [ids[i] for i in order]
        n = len(shuffled)
        n_train = int(round(n * ratios[0]))
        n_dev = int(round(n * ratios[1]))
        train = shuffled[:n_train]
        dev = shuffled[n_train : n_train + n_dev]
        test = shuffled[n_train + n_dev :]
        return train, dev, test
