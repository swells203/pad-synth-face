"""Config-driven behavioral-session generation with manifest + provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from pad_synth_core.provenance import (
    BonafideIngested,
    OntologyCitation,
    ProvenanceLedger,
)
from defid.generator import generate_session
from defid.qc import check_session
from defid.session import SessionManifestWriter

_LABELS = ("genuine", "imposter", "bot")


def _record_citations(ledger: ProvenanceLedger, ontology_dir: Path) -> None:
    for fname in ("touch.yaml", "keystroke.yaml", "motion.yaml"):
        raw = yaml.safe_load((ontology_dir / fname).read_text())
        for axis, body in raw["axes"].items():
            prov = body["provenance"]
            ledger.record(
                OntologyCitation(
                    attack_type=raw["attack_type"],
                    axis=axis,
                    paper=prov["paper"],
                    doi=prov.get("doi"),
                    url=prov.get("url"),
                )
            )


def run_generation(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["run"]["output"])
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    ontology_dir = Path(cfg["ontology_dir"])
    domain = cfg.get("domain", "a")
    n_subjects = int(cfg["subjects"])
    n_sessions = int(cfg["sessions_per_subject"])
    master_seed = int(cfg["run"]["seed"])

    generated = failed = skipped = 0
    with SessionManifestWriter(out / "manifest.jsonl") as manifest, \
            ProvenanceLedger(out / "provenance.jsonl") as ledger:
        ledger.record(
            BonafideIngested(
                name="defid_synthetic",
                license="OWNED",
                source_url=str(ontology_dir),
                sha256_of_index=hashlib.sha256(
                    str(sorted(p.name for p in ontology_dir.glob("*.yaml"))).encode()
                ).hexdigest(),
            )
        )
        _record_citations(ledger, ontology_dir)
        existing = manifest.existing_ids()

        for subj in range(n_subjects):
            subject_id = f"subj-{subj:03d}"
            for sess in range(n_sessions):
                for label in _LABELS:
                    sid = f"{label}-{subject_id}-{sess}-{domain}"
                    if sid in existing:
                        skipped += 1
                        continue
                    s = generate_session(
                        label, subject_id,
                        seed=master_seed * 1000 + subj * n_sessions + sess,
                        ontology_dir=ontology_dir, domain=domain,
                    )
                    s.session_id = sid
                    qc = check_session(s)
                    if not qc.ok:
                        failed += 1
                        continue
                    rel = f"sessions/{sid}.json"
                    blob = s.model_dump_json()
                    (out / rel).write_text(blob)
                    sha = hashlib.sha256(blob.encode()).hexdigest()
                    manifest.append(sid, label, subject_id, rel, sha)
                    generated += 1

    return {"generated": generated, "failed": failed, "skipped": skipped}
