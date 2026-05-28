"""ISO 30107-3 PAD metrics + EER.

All functions are pure: `scores` is a list/array of attack-class probabilities
(P(attack)); `labels` is 0 (bona fide) or 1 (attack); `attack_types` is the
per-sample PAI species string for attack rows and None for bona fide rows.
The decision rule is `score >= threshold => classified attack`.

APCER per PAI species s = fraction of attacks of type s with score < threshold
(i.e. missed). APCER (overall) = max over PAI species (ISO worst-case).
BPCER = fraction of bona fide with score >= threshold. ACER = (APCER + BPCER)/2.

Attack rows whose `attack_type` is None are silently excluded from APCER
accounting -- callers must always set the PAI species on attack rows or the
metric will be deflated.

`threshold_at_apcer` scans candidate thresholds (the unique sample scores plus
sentinels just below the min and just above the max) and returns the highest
threshold whose overall APCER stays at or below `target_apcer`. APCER is
monotonically non-decreasing in the threshold, so 'highest threshold under
the budget' coincides with 'lowest BPCER under the budget' -- the best
operating point that respects the budget. The scan is O(N**2 * #PAI); fine
for evaluation sets up to a few thousand samples, do not call inside a
training loop.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_eer(scores: list[float], labels: list[int]) -> float:
    """Threshold-free Equal Error Rate. Numerically identical to the prior
    implementation in `baseline.py` (kept here as the canonical home).

    Degenerate-input contract: empty input returns 0.5 (no threshold sweep
    runs and the default holds). Single-class input (all bona fide or all
    attack) returns a number computed against a clamped fictitious
    denominator (`max(count, 1)`) -- not a meaningful EER. Callers should
    guard for non-empty, two-class inputs before trusting the result.
    """
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


def apcer_bpcer_acer(
    scores: Iterable[float],
    labels: Iterable[int],
    attack_types: Iterable[str | None],
    threshold: float,
) -> tuple[dict[str, float], float, float, float]:
    """Return (apcer_per_pai, apcer_max, bpcer, acer) at the given threshold.

    Bona fide rows (label 0) are ignored for APCER. Attack rows (label 1) with
    attack_type=None are silently skipped (defensive -- caller should always
    set attack_type on attack rows).
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.int64)
    types = list(attack_types)

    # Per-PAI APCER.
    pai_species = sorted({t for t, lab in zip(types, y) if lab == 1 and t is not None})
    apcer_per_pai: dict[str, float] = {}
    for pai in pai_species:
        mask = np.array([lab == 1 and t == pai for t, lab in zip(types, y)])
        n = int(mask.sum())
        if n == 0:
            continue
        missed = int((s[mask] < threshold).sum())
        apcer_per_pai[pai] = missed / n
    apcer_max = max(apcer_per_pai.values()) if apcer_per_pai else 0.0

    # BPCER.
    bona_mask = (y == 0)
    n_bona = int(bona_mask.sum())
    bpcer = float((s[bona_mask] >= threshold).sum()) / n_bona if n_bona else 0.0

    acer = (apcer_max + bpcer) / 2.0
    return apcer_per_pai, float(apcer_max), float(bpcer), float(acer)


def threshold_at_apcer(
    scores: Iterable[float],
    labels: Iterable[int],
    attack_types: Iterable[str | None],
    target_apcer: float = 0.05,
) -> tuple[float, float]:
    """Return (threshold, achieved_apcer) -- the highest threshold whose overall
    APCER does not exceed `target_apcer`. APCER is monotone non-decreasing in
    threshold, so this is also the threshold minimising BPCER under the budget.
    """
    if target_apcer < 0:
        raise ValueError(f"target_apcer must be non-negative, got {target_apcer}")
    s_arr = np.asarray(list(scores), dtype=np.float64)
    if s_arr.size == 0:
        return 0.0, 0.0
    # Candidate thresholds: every unique score plus sentinels just below min
    # and just above max so we can fully traverse the operating range. The
    # min-1 sentinel always achieves APCER = 0 (all attacks correctly
    # classified at "everything is an attack"), so for any non-negative
    # target_apcer at least one candidate satisfies the budget.
    cands = sorted(set(s_arr.tolist()))
    cands = [float(s_arr.min()) - 1.0] + cands + [float(s_arr.max()) + 1.0]
    types = list(attack_types)
    labels_list = list(labels)
    best_thr = cands[0]
    best_apcer = 0.0
    # cands is sorted ascending, so the last candidate that satisfies the
    # budget is automatically the highest -- no explicit tie-break needed.
    for t in cands:
        _, apcer_max, _, _ = apcer_bpcer_acer(s_arr.tolist(), labels_list, types, t)
        if apcer_max <= target_apcer:
            best_thr = float(t)
            best_apcer = float(apcer_max)
    return best_thr, best_apcer
