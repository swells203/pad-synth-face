"""Behavioral session schema and an append-only JSONL session manifest.

The manifest stores per-session metadata + a pointer to the session payload
file and its sha256 (mirroring the PAD project's manifest/file split). The
full event arrays live in the payload file, not the manifest.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BehavioralSession(BaseModel):
    session_id: str
    label: Literal["genuine", "imposter", "bot"]
    subject_id: str
    touch: list[dict[str, Any]] = Field(default_factory=list)
    key: list[dict[str, Any]] = Field(default_factory=list)
    motion: list[dict[str, Any]] = Field(default_factory=list)
    ontology_version: str
    generator_version: str
    seed: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManifestWriter:
    """Append-only JSONL manifest. One row per session: id, label, subject,
    payload path, payload sha256. Tolerant of a partial trailing line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing = self._scan()
        self._fh = self.path.open("a", encoding="utf-8")

    def _scan(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["session_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a partial trailing line from a crash
        return ids

    def existing_ids(self) -> set[str]:
        return set(self._existing)

    def append(
        self,
        session_id: str,
        label: str,
        subject_id: str,
        payload_path: str,
        payload_sha256: str,
    ) -> None:
        if session_id in self._existing:
            return
        row = {
            "session_id": session_id,
            "label": label,
            "subject_id": subject_id,
            "payload_path": payload_path,
            "payload_sha256": payload_sha256,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._existing.add(session_id)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> "SessionManifestWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
