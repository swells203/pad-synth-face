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
from pad_synth_core.qc.per_sample import check_image_basic
from pad_synth_core.rng import sample_rng
from pad_synth_face.attacks.print import PrintAttack
from pad_synth_face.attacks.replay import ReplayAttack
from pad_synth_face.bonafide import DigiFaceLoader
from pad_synth_face.sensor import MOBILE_FRONT_2024, apply_sensor


_ATTACK_REGISTRY = {"print": PrintAttack, "replay": ReplayAttack}
_SENSOR_REGISTRY = {"mobile-front-2024": MOBILE_FRONT_2024}
_FIXED_IMAGE_SHAPE = (64, 64, 3)


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


def run_pipeline(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_root = Path(cfg["run"]["output"])
    out_root.mkdir(parents=True, exist_ok=True)
    deterministic = bool(cfg["run"].get("deterministic", False))
    if deterministic:
        _set_global_determinism(cfg["run"]["seed"])

    loader = DigiFaceLoader(Path(cfg["bonafide"]["root"]))
    bonafide_ids = loader.list_identities()

    attack_weights = {k: float(v["weight"]) for k, v in cfg["attacks"].items()}
    attack_modules = {
        name: _ATTACK_REGISTRY[name](load_ontology(Path(spec["ontology"])))
        for name, spec in cfg["attacks"].items()
    }
    sensor_preset = _SENSOR_REGISTRY[cfg["sensor_preset"]]

    items = enumerate_work_items(
        master_seed=cfg["run"]["seed"],
        modality="face",
        bonafide_ids=bonafide_ids,
        attack_weights=attack_weights,
        samples_per_bonafide=int(cfg["bonafide"]["samples_per_bonafide"]),
    )

    manifest = ManifestWriter(out_root / "manifest.jsonl")
    ledger = ProvenanceLedger(out_root / "provenance.jsonl")
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
    existing = manifest.existing_sample_ids()

    for it in items:
        if it.sample_id in existing:
            skipped += 1
            continue
        rng = sample_rng(it.seed)
        sample_dir = out_root / "face" / it.attack_type
        sample_dir.mkdir(parents=True, exist_ok=True)

        bonafide_samples = loader.samples_for_identity(it.bonafide_id)
        bonafide_arr = loader.load(bonafide_samples[0])

        module = attack_modules[it.attack_type]
        attack_params = module.sample_params(rng)
        attacked = module.simulate(bonafide_arr, attack_params, rng)
        sensored, sensor_params = apply_sensor(attacked, sensor_preset, rng)

        qc = check_image_basic(sensored, _FIXED_IMAGE_SHAPE)
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
            ontology_version="2026-05-11",
            seed=it.seed,
            output_path=out_path_rel,
            output_sha256=sha,
        )
        manifest.append(rec)
        generated += 1

    manifest.close()
    ledger.close()

    return {
        "samples_generated": generated,
        "samples_failed": failed,
        "samples_skipped_existing": skipped,
        "manifest_path": str(out_root / "manifest.jsonl"),
    }
