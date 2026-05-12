"""Phase-1 baseline PAD detector + EER computation.

This is a SCAFFOLD. It exists so we can wire the cross-domain eval loop end-to-end
and start tracking a number; it is not a state-of-the-art detector. Swap-in for a
real eval set is a `dataset_root` change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyPADDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.items: list[tuple[Path, int]] = []
        face_root = root / "face"
        # Bonafide samples live under face/bonafide/.
        for p in sorted((face_root / "bonafide").glob("*.jpg")):
            self.items.append((p, 0))
        # All other face/<x>/ subdirectories are attack types (print, replay, ...).
        for subdir in sorted(p for p in face_root.iterdir() if p.is_dir()):
            if subdir.name == "bonafide":
                continue
            for p in sorted(subdir.glob("*.jpg")):
                self.items.append((p, 1))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_eer(scores: list[float], labels: list[int]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    thresholds = np.unique(s)
    best = 1.0
    eer = 0.5
    for t in thresholds:
        pred = (s >= t).astype(np.int64)
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        fpr = fp / n_neg
        fnr = fn / n_pos
        diff = abs(fpr - fnr)
        if diff < best:
            best = diff
            eer = (fpr + fnr) / 2.0
    return float(eer)


def train_and_eval_tiny_cnn(
    dataset_root: Path,
    epochs: int = 1,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    ds = TinyPADDataset(dataset_root)
    n_val = max(1, len(ds) // 4)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].tolist()
            scores.extend(probs)
            labels.extend(y.tolist())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.numel())

    return {
        "eer": compute_eer(scores, labels),
        "val_accuracy": correct / max(total, 1),
        "n_train": n_train,
        "n_val": n_val,
    }
