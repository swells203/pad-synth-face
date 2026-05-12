"""Sample manifest schema and append-only JSONL writer."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BonafideSource(BaseModel):
    dataset: str
    id: str
    license: str
    url: str | None = None


class GeneratorUsage(BaseModel):
    name: str
    version: str
    license: str
    commercial_ok: bool
    model_hash: str | None = None


class SampleRecord(BaseModel):
    sample_id: str
    modality: Literal["face", "voice"]
    label: Literal["bonafide", "attack"]
    attack_type: str | None
    bonafide_source: BonafideSource
    attack_params: dict[str, Any] = Field(default_factory=dict)
    sensor_preset: str | None = None
    sensor_params: dict[str, Any] = Field(default_factory=dict)
    generators_used: list[GeneratorUsage] = Field(default_factory=list)
    pipeline_version: str
    core_version: str
    ontology_version: str
    seed: int
    output_path: str
    output_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ManifestWriter:
    """Append-only JSONL writer for sample manifests.

    Reads existing sample_ids on open so callers can skip already-completed work.
    Caller is responsible for never instantiating two writers on the same path
    simultaneously.
    """

    def __init__(self, path: Path, fsync_every: int = 100) -> None:
        self.path = Path(path)
        self.fsync_every = fsync_every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing: set[str] = self._scan_existing()
        self._fh = self.path.open("a", encoding="utf-8")
        self._written_since_fsync = 0

    def _scan_existing(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["sample_id"])
        return ids

    def existing_sample_ids(self) -> set[str]:
        return set(self._existing)

    def append(self, record: SampleRecord) -> None:
        if record.sample_id in self._existing:
            return
        self._fh.write(record.model_dump_json() + "\n")
        self._existing.add(record.sample_id)
        self._written_since_fsync += 1
        if self._written_since_fsync >= self.fsync_every:
            self._fsync()

    def _fsync(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._written_since_fsync = 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fsync()
            self._fh.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
