from pad_synth_core.orchestrator import WorkItem, enumerate_work_items


def test_enumerate_work_items_is_deterministic():
    a = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01", "02"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=2,
    )
    b = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01", "02"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=2,
    )
    assert a == b


def test_enumerate_respects_total_count():
    items = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01", "02"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=4,
    )
    assert len(items) == 3 * 4


def test_enumerate_balances_attack_weights():
    items = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=[f"{i:02d}" for i in range(50)],
        attack_weights={"print": 1.0, "replay": 3.0},
        samples_per_bonafide=4,
    )
    types = [it.attack_type for it in items]
    replay_pct = types.count("replay") / len(types)
    # With weights 1:3 and N=200, replay should be ~75% with noise tolerance.
    assert 0.65 < replay_pct < 0.85


def test_work_item_has_unique_sample_ids():
    items = enumerate_work_items(
        master_seed=42,
        modality="face",
        bonafide_ids=["00", "01"],
        attack_weights={"print": 1.0, "replay": 1.0},
        samples_per_bonafide=3,
    )
    ids = [it.sample_id for it in items]
    assert len(ids) == len(set(ids))
