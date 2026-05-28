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

from pad_synth_core.eval.metrics import apcer_bpcer_acer, compute_eer, threshold_at_apcer  # noqa: F401  (compute_eer re-exported)


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


def subject_disjoint_split(
    dataset: "TinyPADDataset",
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split a TinyPADDataset into (train, val) with disjoint subjects.

    Groups samples by `dataset.subjects` and assigns whole identities to the
    val side until the running val count reaches roughly `val_fraction` of
    the dataset. Samples with subject=None go to train (no leakage risk).
    Falls back to torch's random_split when every subject is None
    (preserves current behaviour for manifest-less datasets).
    """
    n = len(dataset)
    n_val_target = max(1, int(round(n * val_fraction)))
    n_train_target = max(1, n - n_val_target)

    subjects = getattr(dataset, "subjects", [None] * n)
    if not subjects or all(s is None for s in subjects):
        return torch.utils.data.random_split(
            dataset, [n_train_target, n_val_target],
            generator=torch.Generator().manual_seed(seed),
        )

    by_subj: dict[str, list[int]] = {}
    no_subj: list[int] = []
    for i, s in enumerate(subjects):
        if s is None:
            no_subj.append(i)
        else:
            by_subj.setdefault(s, []).append(i)

    rng = np.random.default_rng(seed)
    order = list(by_subj.keys())
    rng.shuffle(order)

    val_idx: list[int] = []
    val_subjects: set[str] = set()
    for s in order:
        if len(val_idx) >= n_val_target:
            break
        val_idx.extend(by_subj[s])
        val_subjects.add(s)

    train_idx = sorted(
        no_subj + [i for s in order if s not in val_subjects for i in by_subj[s]]
    )
    val_idx = sorted(val_idx)
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
    )


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


def _score_dataset(model, dataset, batch_size, device):
    """Run inference and return (scores, labels, attack_types) aligned 1:1
    with the dataset (or Subset) order, with no shuffling."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1].cpu().tolist()
            scores.extend(probs)
            labels.extend(y.tolist())
    if isinstance(dataset, torch.utils.data.Subset):
        attack_types = [dataset.dataset.attack_types[i] for i in dataset.indices]
    else:
        attack_types = list(dataset.attack_types)
    return scores, labels, attack_types


def train_and_cross_domain_eval(
    train_root: Path,
    eval_root: Path | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    target_apcer: float = 0.05,
) -> dict[str, Any]:
    """Train on train_root; eval in-domain (held-out 25 percent split, now
    subject-disjoint when a manifest is present) and optionally cross-domain
    (full eval_root if provided). Adds ISO 30107-3 metrics at a dev-fixed
    threshold (target APCER = `target_apcer`) on top of the existing EER
    reporting -- all new keys are additive."""
    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")

    train_ds_full = TinyPADDataset(train_root)
    train_ds, val_ds = subject_disjoint_split(train_ds_full, val_fraction=0.25, seed=seed)
    n_train, n_val = len(train_ds), len(val_ds)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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
    # In-domain scoring (dev split).
    dev_scores, dev_labels, dev_atypes = _score_dataset(model, val_ds, batch_size, dev)
    in_eer = compute_eer(dev_scores, dev_labels)
    in_acc = (
        sum(int((s >= 0.5) == y) for s, y in zip(dev_scores, dev_labels)) / max(len(dev_scores), 1)
    )
    threshold, _ = threshold_at_apcer(dev_scores, dev_labels, dev_atypes, target_apcer)

    cross_eer: float | None = None
    cross_acc: float | None = None
    n_val_cross: int | None = None
    apcer_per_pai: dict[str, float] | None = None
    apcer_max: float | None = None
    bpcer: float | None = None
    acer: float | None = None
    if eval_root is not None:
        cross_ds = TinyPADDataset(eval_root)
        cross_scores, cross_labels, cross_atypes = _score_dataset(model, cross_ds, batch_size, dev)
        cross_eer = compute_eer(cross_scores, cross_labels)
        cross_acc = (
            sum(int((s >= 0.5) == y) for s, y in zip(cross_scores, cross_labels))
            / max(len(cross_scores), 1)
        )
        n_val_cross = len(cross_ds)
        apcer_per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(
            cross_scores, cross_labels, cross_atypes, threshold,
        )

    return {
        # Existing keys -- preserved.
        "eer_in_domain": in_eer,
        "val_accuracy_in_domain": in_acc,
        "n_train": n_train,
        "n_val_in_domain": n_val,
        "eer_cross_domain": cross_eer,
        "val_accuracy_cross_domain": cross_acc,
        "n_val_cross_domain": n_val_cross,
        # New additive keys.
        "threshold": float(threshold),
        "target_apcer": float(target_apcer),
        "apcer_cross_domain": apcer_max,
        "bpcer_cross_domain": bpcer,
        "acer_cross_domain": acer,
        "apcer_per_pai_cross_domain": apcer_per_pai,
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
