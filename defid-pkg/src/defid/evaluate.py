"""Train on a generated session set; evaluate continuous-auth EER and bot
accuracy in-domain, and optionally cross-domain on a second set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from defid.features import extract_features
from defid.models import LogisticBotClassifier, MahalanobisAuth
from defid.session import BehavioralSession


def _load(root: Path) -> dict[str, list[np.ndarray]]:
    by_label: dict[str, list[np.ndarray]] = {
        "genuine": [], "imposter": [], "bot": []
    }
    for line in (root / "manifest.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        s = BehavioralSession.model_validate_json(
            (root / row["payload_path"]).read_text()
        )
        by_label[row["label"]].append(extract_features(s))
    return {k: v for k, v in by_label.items()}


def _auth_eer(train: dict, test: dict) -> float:
    g_train = np.vstack(train["genuine"])
    m = MahalanobisAuth().fit(g_train)
    return m.eer(np.vstack(test["genuine"]), np.vstack(test["imposter"]))


def _bot_acc(train: dict, test: dict) -> float:
    Xtr = np.vstack(train["genuine"] + train["imposter"] + train["bot"])
    ytr = np.array(
        [0] * len(train["genuine"])
        + [0] * len(train["imposter"])
        + [1] * len(train["bot"])
    )
    clf = LogisticBotClassifier(seed=0).fit(Xtr, ytr)
    Xte = np.vstack(test["genuine"] + test["imposter"] + test["bot"])
    yte = np.array(
        [0] * len(test["genuine"])
        + [0] * len(test["imposter"])
        + [1] * len(test["bot"])
    )
    preds = (clf.predict_proba(Xte) >= 0.5).astype(int)
    return float((preds == yte).mean())


def evaluate(train_root: Path, eval_root: Path | None) -> dict[str, Any]:
    train = _load(Path(train_root))
    result: dict[str, Any] = {
        "auth_eer_in_domain": _auth_eer(train, train),
        "bot_accuracy_in_domain": _bot_acc(train, train),
        "auth_eer_cross_domain": None,
        "bot_accuracy_cross_domain": None,
    }
    if eval_root is not None:
        ev = _load(Path(eval_root))
        result["auth_eer_cross_domain"] = _auth_eer(train, ev)
        result["bot_accuracy_cross_domain"] = _bot_acc(train, ev)
    return result
