"""Append-only provenance ledger for dataset-level audit trail."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BonafideIngested(BaseModel):
    type: Literal["bonafide_dataset_ingested"] = "bonafide_dataset_ingested"
    name: str
    license: str
    source_url: str
    sha256_of_index: str
    ingested_at: datetime = Field(default_factory=_now)


class GeneratorRegistered(BaseModel):
    type: Literal["generator_registered"] = "generator_registered"
    name: str
    version: str
    license: str
    commercial_ok: bool
    model_hash: str
    registered_at: datetime = Field(default_factory=_now)


class OntologyCitation(BaseModel):
    type: Literal["ontology_citation"] = "ontology_citation"
    attack_type: str
    axis: str
    paper: str
    doi: str | None = None
    url: str | None = None


ProvenanceEvent = BonafideIngested | GeneratorRegistered | OntologyCitation


class ProvenanceLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def record(self, event: ProvenanceEvent) -> None:
        self._fh.write(event.model_dump_json() + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> "ProvenanceLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
