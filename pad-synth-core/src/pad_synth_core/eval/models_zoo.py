"""Model factories for the PAD scaling experiment.

Each factory returns an nn.Module mapping an RGB image batch (B, 3, H, W)
to logits (B, 2) — bonafide=0, attack=1. The factories are deliberately
small and explicit; they exist solely for the Spark capacity sweep.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def make_tiny_cnn() -> nn.Module:
    """The Phase-1 baseline (kept here for sweep symmetry)."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 2),
    )


def make_small_cnn() -> nn.Module:
    """~97k params; the mid-capacity tier."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 2),
    )


def make_resnet18() -> nn.Module:
    """torchvision ResNet18 from scratch; final fc -> Linear(512, 2)."""
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def make_resnet18_pretrained() -> nn.Module:
    """ResNet18 with ImageNet-pretrained weights; final fc -> Linear(512, 2).
    Same head-swap pattern as `make_resnet18`; only the weight init differs."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


FACTORIES = {
    "L1": make_tiny_cnn,
    "L2": make_small_cnn,
    "L3": make_resnet18,
    "L4": make_resnet18_pretrained,
}
