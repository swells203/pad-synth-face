"""Thin FastAPI shell over DemoService. One in-memory instance; trusted
LAN demo only — no auth, no persistence (by design, see spec §10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from defid_demo.adapter import RepPayload
from defid_demo.service import DemoService

app = FastAPI(title="DefinitiveID Live Demo")
_svc = DemoService()
_WEB = Path(__file__).parent / "web"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.get("/spectator")
def spectator() -> FileResponse:
    return FileResponse(_WEB / "spectator.html")


@app.post("/api/reset")
def reset() -> dict:
    _svc.reset()
    return {"ok": True}


@app.post("/api/enroll")
def enroll(payload: dict) -> dict:
    return _svc.enroll(RepPayload(pointer=payload.get("pointer", []),
                                  keys=payload.get("keys", [])))


@app.post("/api/calibrate")
def calibrate() -> dict:
    return _svc.calibrate()


@app.post("/api/attempt")
def attempt(payload: dict) -> dict:
    return _svc.attempt(RepPayload(pointer=payload.get("pointer", []),
                                   keys=payload.get("keys", [])))


@app.get("/api/state")
def state() -> dict:
    return _svc.state()
