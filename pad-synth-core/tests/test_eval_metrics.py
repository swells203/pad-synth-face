import math

from pad_synth_core.eval.metrics import (
    apcer_bpcer_acer,
    compute_eer,
    threshold_at_apcer,
)


def test_compute_eer_matches_known_case():
    # Bonafide low scores, attacks high scores: EER ~ 0 at threshold 0.5.
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert compute_eer(scores, labels) == 0.0


def test_apcer_bpcer_acer_hand_computed():
    # 4 bonafide (label 0), 6 attacks: 3 of type 'print', 3 of type 'replay'.
    # At threshold 0.5: bona scores {0.2,0.3,0.6,0.7} -> 2 above (BPCER 2/4=0.5);
    # print scores {0.4,0.4,0.9} -> 2 below (APCER_print 2/3); replay {0.6,0.7,0.8}
    # -> 0 below (APCER_replay 0/3). APCER_max = 2/3, ACER = (2/3 + 1/2) / 2.
    scores       = [0.2, 0.3, 0.6, 0.7, 0.4, 0.4, 0.9, 0.6, 0.7, 0.8]
    labels       = [0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ]
    attack_types = [None,None,None,None,"print","print","print","replay","replay","replay"]
    per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(scores, labels, attack_types, 0.5)
    assert math.isclose(per_pai["print"], 2/3)
    assert math.isclose(per_pai["replay"], 0.0)
    assert math.isclose(apcer_max, 2/3)
    assert math.isclose(bpcer, 0.5)
    assert math.isclose(acer, (2/3 + 0.5) / 2.0)


def test_apcer_ignores_bonafide_rows_and_handles_missing_types():
    # All-bonafide eval: APCER must be 0, ACER = BPCER / 2.
    per_pai, apcer_max, bpcer, acer = apcer_bpcer_acer(
        scores=[0.1, 0.9], labels=[0, 0], attack_types=[None, None], threshold=0.5,
    )
    assert per_pai == {}
    assert apcer_max == 0.0
    assert bpcer == 0.5
    assert acer == 0.25


def test_threshold_at_apcer_respects_budget_and_prefers_higher_threshold():
    # Two PAI species; choose threshold so overall APCER <= 0.5, with the
    # higher-threshold tie-break (lower BPCER under the same budget).
    scores       = [0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.7]
    labels       = [0,   0,   0,   0,   1,   1,   1  ]
    attack_types = [None,None,None,None,"print","print","replay"]
    thr, achieved = threshold_at_apcer(scores, labels, attack_types, target_apcer=0.5)
    # Verify the constraint actually holds at the returned threshold.
    per_pai, apcer_max, _, _ = apcer_bpcer_acer(scores, labels, attack_types, thr)
    assert apcer_max <= 0.5 + 1e-9
    assert achieved == apcer_max
    # And no strictly higher candidate threshold (from the score set) would also satisfy it.
    for t in sorted(set(scores)):
        if t > thr:
            _, ap, _, _ = apcer_bpcer_acer(scores, labels, attack_types, t)
            assert ap > 0.5 + 1e-9, f"higher thr {t} also satisfied -- tie-break broken"


def test_threshold_at_apcer_trivially_low_with_attacks_only_above_budget():
    # All attacks have very high scores -> APCER stays 0 across most thresholds;
    # function must still return a finite threshold and achieved <= target.
    scores = [0.1, 0.2, 0.95, 0.96, 0.97]
    labels = [0,   0,   1,    1,    1   ]
    types  = [None,None,"print","print","print"]
    thr, achieved = threshold_at_apcer(scores, labels, types, target_apcer=0.05)
    assert achieved <= 0.05 + 1e-9
    assert math.isfinite(thr)


def test_threshold_at_apcer_rejects_negative_target():
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        threshold_at_apcer([0.1, 0.9], [0, 1], [None, "print"], target_apcer=-0.01)
