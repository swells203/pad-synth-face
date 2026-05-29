"""End-to-end face generation pipeline."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

import pad_synth_core
import pad_synth_face
from pad_synth_core.manifest import (
    BonafideSource,
    ManifestWriter,
    SampleRecord,
)
from pad_synth_core.ontology import load_ontology
from pad_synth_core.orchestrator import enumerate_work_items
from pad_synth_core.provenance import (
    BonafideIngested,
    OntologyCitation,
    ProvenanceLedger,
)
from pad_synth_core import IMAGE_SHAPE
from pad_synth_core.qc.per_sample import check_image_basic
from pad_synth_core.rng import derive_sample_seed, sample_rng
from pad_synth_face.attacks.mask import MaskAttack
from pad_synth_face.attacks.print import PrintAttack
from pad_synth_face.attacks.replay import ReplayAttack
from pad_synth_face.bonafide import DigiFaceLoader
from pad_synth_face.sensor import MOBILE_FRONT_2024, WEBCAM_1080P, apply_sensor


_ATTACK_REGISTRY = {"print": PrintAttack, "replay": ReplayAttack, "mask": MaskAttack}
_SENSOR_REGISTRY = {
    "mobile-front-2024": MOBILE_FRONT_2024,
    "webcam-1080p": WEBCAM_1080P,
}


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _record_ontology_citations(
    ledger: ProvenanceLedger, attack_type: str, ontology_path: Path
) -> None:
    import yaml as _yaml

    raw = _yaml.safe_load(ontology_path.read_text())
    for axis, body in raw["axes"].items():
        prov = body["provenance"]
        ledger.record(
            OntologyCitation(
                attack_type=attack_type,
                axis=axis,
                paper=prov["paper"],
                doi=prov.get("doi"),
                url=prov.get("url"),
            )
        )


def _canonical_ontology_version(attack_modules: dict[str, Any]) -> str:
    """Pick one canonical ontology version to stamp on every sample record.

    Bonafide records have no attack ontology of their own, so the dataset
    borrows one attack's version. Prefer print (the dominant version-tracked
    component historically), then replay, then mask; otherwise fall back to
    the alphabetically-first attack present. Robust to mask-only configs that
    have no print attack.
    """
    for preferred in ("print", "replay", "mask"):
        if preferred in attack_modules:
            return attack_modules[preferred].ontology.version
    first = sorted(attack_modules)[0]
    return attack_modules[first].ontology.version


def run_pipeline(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_root = Path(cfg["run"]["output"])
    out_root.mkdir(parents=True, exist_ok=True)
    deterministic = bool(cfg["run"].get("deterministic", False))
    if deterministic:
        _set_global_determinism(cfg["run"]["seed"])

    _bonafide_cfg = cfg["bonafide"]
    if "identities_file" in _bonafide_cfg:
        _ids_path = Path(_bonafide_cfg["identities_file"])
        _restrict = [
            line.strip() for line in _ids_path.read_text().splitlines() if line.strip()
        ]
        loader = DigiFaceLoader(Path(_bonafide_cfg["root"]), restrict_to=_restrict)
    else:
        loader = DigiFaceLoader(Path(_bonafide_cfg["root"]))
    bonafide_ids = loader.list_identities()

    # Compute identity-disjoint splits. Default ratios if not in config.
    split_cfg = cfg["bonafide"].get("splits", {"train": 0.7, "dev": 0.15, "test": 0.15})
    train_ids, dev_ids, test_ids = loader.identity_disjoint_split(
        seed=cfg["run"]["seed"],
        ratios=(split_cfg["train"], split_cfg["dev"], split_cfg["test"]),
    )
    id_to_split: dict[str, str] = {}
    for tid in train_ids:
        id_to_split[tid] = "train"
    for did in dev_ids:
        id_to_split[did] = "dev"
    for sid in test_ids:
        id_to_split[sid] = "test"

    attack_weights = {k: float(v["weight"]) for k, v in cfg["attacks"].items()}
    attack_modules = {
        name: _ATTACK_REGISTRY[name](load_ontology(Path(spec["ontology"])))
        for name, spec in cfg["attacks"].items()
    }
    # Single canonical ontology_version for all sample records in this run.
    # Bonafide records have no attack ontology of their own, so they borrow
    # one attack's version via _canonical_ontology_version (priority:
    # print -> replay -> mask -> alphabetical), which is robust to configs
    # that omit any given attack (e.g. mask-only runs).
    _ontology_version = _canonical_ontology_version(attack_modules)
    sensor_preset = _SENSOR_REGISTRY[cfg["sensor_preset"]]

    items = enumerate_work_items(
        master_seed=cfg["run"]["seed"],
        modality="face",
        bonafide_ids=bonafide_ids,
        attack_weights=attack_weights,
        samples_per_bonafide=int(cfg["bonafide"]["samples_per_bonafide"]),
    )

    with ManifestWriter(out_root / "manifest.jsonl") as manifest, \
            ProvenanceLedger(out_root / "provenance.jsonl") as ledger:
        ledger.record(
            BonafideIngested(
                name="digiface_fixture",
                license="MIT",
                source_url=str(cfg["bonafide"]["root"]),
                sha256_of_index=hashlib.sha256(
                    "|".join(bonafide_ids).encode()
                ).hexdigest(),
            )
        )
        for name, spec in cfg["attacks"].items():
            _record_ontology_citations(ledger, name, Path(spec["ontology"]))

        generated = 0
        failed = 0
        skipped = 0

        # Emit samples_per_bonafide bonafide samples per identity, BEFORE the
        # attack loop. Each slot gets its own derived seed → distinct sensor
        # noise/WB/qf, so per-identity bonafide samples are visually distinct.
        # This is what gives the PAD detector enough negative-class diversity
        # to actually learn from; emitting one-per-identity collapses the
        # bonafide distribution to N points and produces a degenerate EER.
        samples_per_bonafide = int(cfg["bonafide"]["samples_per_bonafide"])
        existing = manifest.existing_sample_ids()
        bonafide_emitted = 0
        bonafide_failed = 0
        bonafide_counter = 0
        for bid in sorted(bonafide_ids):
            bsamples = loader.samples_for_identity(bid)
            if not bsamples:
                # All slots for this identity fail.
                bonafide_failed += samples_per_bonafide
                bonafide_counter += samples_per_bonafide
                continue
            for sub in range(samples_per_bonafide):
                bonafide_sid = f"face-bonafide-{bonafide_counter:08d}"
                if bonafide_sid in existing:
                    bonafide_counter += 1
                    continue
                bonafide_seed = derive_sample_seed(
                    cfg["run"]["seed"], "face", "bonafide", bonafide_counter
                )
                rng = sample_rng(bonafide_seed)
                arr = loader.load(bsamples[sub % len(bsamples)])
                sensored, sensor_params = apply_sensor(arr, sensor_preset, rng)
                qc = check_image_basic(sensored, IMAGE_SHAPE)
                if not qc.ok:
                    bonafide_failed += 1
                    bonafide_counter += 1
                    continue
                out_rel = f"face/bonafide/{bonafide_sid}.jpg"
                out_abs = out_root / out_rel
                out_abs.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(sensored).save(out_abs, format="JPEG", quality=92)
                sha = hashlib.sha256(out_abs.read_bytes()).hexdigest()
                rec = SampleRecord(
                    sample_id=bonafide_sid,
                    modality="face",
                    label="bonafide",
                    attack_type=None,
                    split=id_to_split[bid],
                    bonafide_source=BonafideSource(
                        dataset="digiface_fixture", id=bid, license="MIT"
                    ),
                    sensor_preset=sensor_preset.name,
                    sensor_params=sensor_params,
                    pipeline_version=f"pad-synth-face@{pad_synth_face.__version__}",
                    core_version=f"pad-synth-core@{pad_synth_core.__version__}",
                    ontology_version=_ontology_version,
                    seed=bonafide_seed,
                    output_path=out_rel,
                    output_sha256=sha,
                )
                manifest.append(rec)
                bonafide_emitted += 1
                bonafide_counter += 1

        # Re-read existing IDs after bonafide pass to cover both sets.
        existing = manifest.existing_sample_ids()

        for it in items:
            if it.sample_id in existing:
                skipped += 1
                continue
            rng = sample_rng(it.seed)
            sample_dir = out_root / "face" / it.attack_type
            sample_dir.mkdir(parents=True, exist_ok=True)

            bonafide_samples = loader.samples_for_identity(it.bonafide_id)
            if not bonafide_samples:
                failed += 1
                continue
            bonafide_arr = loader.load(bonafide_samples[0])

            module = attack_modules[it.attack_type]
            attack_params = module.sample_params(rng)
            attacked = module.simulate(bonafide_arr, attack_params, rng)
            sensored, sensor_params = apply_sensor(attacked, sensor_preset, rng)

            qc = check_image_basic(sensored, IMAGE_SHAPE)
            if not qc.ok:
                failed += 1
                continue

            out_path_rel = f"face/{it.attack_type}/{it.sample_id}.jpg"
            out_path_abs = out_root / out_path_rel
            Image.fromarray(sensored).save(out_path_abs, format="JPEG", quality=92)
            sha = hashlib.sha256(out_path_abs.read_bytes()).hexdigest()

            rec = SampleRecord(
                sample_id=it.sample_id,
                modality="face",
                label="attack",
                attack_type=it.attack_type,
                split=id_to_split[it.bonafide_id],
                bonafide_source=BonafideSource(
                    dataset="digiface_fixture",
                    id=it.bonafide_id,
                    license="MIT",
                ),
                attack_params=attack_params,
                sensor_preset=sensor_preset.name,
                sensor_params=sensor_params,
                pipeline_version=f"pad-synth-face@{pad_synth_face.__version__}",
                core_version=f"pad-synth-core@{pad_synth_core.__version__}",
                ontology_version=_ontology_version,
                seed=it.seed,
                output_path=out_path_rel,
                output_sha256=sha,
            )
            manifest.append(rec)
            generated += 1

    return {
        "samples_generated": generated,
        "samples_failed": failed,
        "samples_skipped_existing": skipped,
        "bonafide_emitted": bonafide_emitted,
        "bonafide_failed": bonafide_failed,
        "manifest_path": str(out_root / "manifest.jsonl"),
    }
