"""Phase-1 baseline PAD detector + EER computation.

This is a SCAFFOLD. It exists so we can wire the cross-domain eval loop end-to-end
and start tracking a number; it is not a state-of-the-art detector. Swap-in for a
real eval set is a `dataset_root` change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pad_synth_core.eval.metrics import compute_eer  # re-exported for backward compatibility


class TinyPADDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.items: list[tuple[Path, int]] = []
        self.subjects: list[str | None] = []
        self.attack_types: list[str | None] = []
        face_root = Path(root) / "face"

        # Manifest provides per-sample subject + attack_type when present;
        # absent or unparseable -> graceful (subjects/attack_types stay None,
        # callers fall back to random splits).
        by_output_path: dict[str, tuple[str | None, str | None]] = {}
        manifest_path = Path(root) / "manifest.jsonl"
        if manifest_path.exists():
            for line in manifest_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    subj = (rec.get("bonafide_source") or {}).get("id")
                    by_output_path[rec["output_path"]] = (subj, rec.get("attack_type"))
                except (json.JSONDecodeError, KeyError):
                    continue

        def _add(path: Path, label: int) -> None:
            self.items.append((path, label))
            rel = str(path.relative_to(Path(root)))
            subj, atype = by_output_path.get(rel, (None, None))
            self.subjects.append(subj)
            self.attack_types.append(atype)

        # Bonafide samples live under face/bonafide/.
        for p in sorted((face_root / "bonafide").glob("*.jpg")):
            _add(p, 0)
        # All other face/<x>/ subdirectories are attack types (print, replay, ...).
        for subdir in sorted(p for p in face_root.iterdir() if p.is_dir()):
            if subdir.name == "bonafide":
                continue
            for p in sorted(subdir.glob("*.jpg")):
                _add(p, 1)

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


def _eval_loader(
    model: nn.Module, dl: DataLoader, device: torch.device | None = None
) -> tuple[float, float]:
    """Run a model over a dataloader; return (EER, accuracy)."""
    dev = device or torch.device("cpu")
    scores: list[float] = []
    labels: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            scores.extend(probs)
            labels.extend(y.cpu().tolist())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.numel())
    return compute_eer(scores, labels), correct / max(total, 1)


def train_and_cross_domain_eval(
    train_root: Path,
    eval_root: Path | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict[str, Any]:
    """Train on train_root; eval in-domain (held-out 25 percent split) and
    optionally cross-domain (full eval_root if provided).

    Defaults preserve the Phase 1/1.5 behavior (TinyCNN on CPU). Pass
    `device="cuda"` and `model_factory=make_small_cnn` etc. for the
    Spark scaling sweep.

    Returns the same dict shape as before, with all numeric fields finite.
    """
    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")

    train_ds_full = TinyPADDataset(train_root)
    n_val = max(1, len(train_ds_full) // 4)
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = (model_factory or TinyCNN)().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    in_eer, in_acc = _eval_loader(model, val_dl, dev)

    cross_eer: float | None = None
    cross_acc: float | None = None
    n_val_cross: int | None = None
    if eval_root is not None:
        cross_ds = TinyPADDataset(eval_root)
        cross_dl = DataLoader(cross_ds, batch_size=batch_size)
        cross_eer, cross_acc = _eval_loader(model, cross_dl, dev)
        n_val_cross = len(cross_ds)

    return {
        "eer_in_domain": in_eer,
        "val_accuracy_in_domain": in_acc,
        "n_train": n_train,
        "n_val_in_domain": n_val,
        "eer_cross_domain": cross_eer,
        "val_accuracy_cross_domain": cross_acc,
        "n_val_cross_domain": n_val_cross,
    }


def train_and_eval_tiny_cnn(
    dataset_root: Path,
    epochs: int = 1,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Backward-compatible wrapper around train_and_cross_domain_eval.

    Returns the original field names: eer, val_accuracy, n_train, n_val.
    """
    full = train_and_cross_domain_eval(
        train_root=dataset_root,
        eval_root=None,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    return {
        "eer": full["eer_in_domain"],
        "val_accuracy": full["val_accuracy_in_domain"],
        "n_train": full["n_train"],
        "n_val": full["n_val_in_domain"],
    }
