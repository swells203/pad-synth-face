"""In-memory demo orchestrator. Pure (no web layer) so it is fully
unit-testable. Holds out the last HOLDOUT_REPS enrollment reps for
threshold calibration."""

from __future__ import annotations

import numpy as np

from defid_demo.adapter import RepPayload, payload_to_session
from defid_demo.demo_auth import DemoAuth
from defid_demo.qc import check_rep
from defid_demo.windows import FEATURE_SUBSET, extract_windows

MIN_FIT_REPS = 4      # reps used to fit (after holdout)
HOLDOUT_REPS = 2      # last enrollment reps reserved for calibration


class DemoService:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._reps: list[np.ndarray] = []  # one (k,9) array per enroll rep
        self._auth: DemoAuth | None = None
        self._attempts: list[dict] = []

    def _windows(self, payload: RepPayload):
        touch, key = payload_to_session(payload)
        qc = check_rep(touch, key)
        if not qc.ok:
            return None, qc.reason
        return extract_windows(touch, key), None

    def enroll(self, payload: RepPayload) -> dict:
        W, reason = self._windows(payload)
        if W is None:
            return {"ok": False, "reason": reason}
        self._reps.append(W)
        return {"ok": True, "enroll_reps": len(self._reps)}

    def calibrate(self) -> dict:
        if len(self._reps) < MIN_FIT_REPS + HOLDOUT_REPS:
            need = MIN_FIT_REPS + HOLDOUT_REPS - len(self._reps)
            return {"ok": False,
                    "reason": f"need {need} more enrollment reps"}
        fit_reps = self._reps[:-HOLDOUT_REPS]
        hold_reps = self._reps[-HOLDOUT_REPS:]
        Xfit = np.vstack(fit_reps)
        Xhold = np.vstack(hold_reps)
        self._auth = DemoAuth().fit_named(Xfit, list(FEATURE_SUBSET))
        thr = self._auth.calibrate(Xhold)
        return {
            "ok": True,
            "threshold": thr,
            "kept": len(self._auth.kept_idx),
            "dropped": self._auth.dropped_names,
        }

    def attempt(self, payload: RepPayload) -> dict:
        if self._auth is None or self._auth.threshold is None:
            return {"ok": False, "reason": "not calibrated"}
        W, reason = self._windows(payload)
        if W is None:
            return {"ok": False, "reason": reason}
        result = self._auth.classify(W)
        result["ok"] = True
        feat_mean = W.mean(axis=0)
        result["feature_values"] = {
            n: float(v) for n, v in zip(FEATURE_SUBSET, feat_mean)
        }
        self._attempts.append(result)
        return result

    def state(self) -> dict:
        return {
            "enroll_reps": len(self._reps),
            "calibrated": self._auth is not None
            and self._auth.threshold is not None,
            "threshold": None if self._auth is None else self._auth.threshold,
            "attempts": self._attempts,
            "features": list(FEATURE_SUBSET),
        }
