% DefinitiveID — Live Partner Demo Design
% Behavioral-biometrics in-browser enroll-and-impostor demonstration
% 2026-05-15

---

## 1. Purpose and audience

A live, in-person demonstration of the DefinitiveID behavioral-biometrics
mechanic for **technical potential partners**.

The demo has two jobs, in priority order:

- **A — visceral proof the signal is real.** Enroll one person live; a
  different person performs the same gesture and passphrase; the system
  visibly rejects them. The spine of the demo is this moment.
- **B — methodology credibility.** A skeptical engineer must be able to
  watch the evidence accumulate (impostor distances stacking above the
  decision threshold), inspect the live feature vector and Mahalanobis
  distance, and hear an honest account of why it works and what is and is
  not being claimed.

This is a separate deliverable from the offline synthetic PoC (merged to
`main`). The demo *cites* the PoC's cross-domain number; it does not
recompute it.

## 2. The honesty claim (load-bearing — partners will probe it)

The demo is **live-enrolled**: the model is fit on the subject's own real
captured behavior, then tested on the same real distribution. The
synthetic→real transfer gap is therefore *not* exercised or claimed by the
demo; it is quantified separately by the offline cross-domain PoC.

The exact, defensible statement to make to partners:

> "The scorer is the same `MahalanobisAuth` class validated in our offline
> cross-domain PoC. The features use the same formulas (touch
> speed/curvature/jitter/inter-touch interval; key dwell/flight),
> restricted to the well-conditioned touch+keystroke subset and computed
> per window. It is live-enrolled — fit on the subject's own real captures
> — so no synthetic→real transfer is being claimed here; that gap is
> quantified separately in the offline cross-domain number."

Constraint: the existing `defid` package (`defid-pkg/`, import `defid`)
and its 102-test suite **must not be modified**. The demo's
windowed-subset extractor and shrinkage are new code that call into or
mirror `defid`, never edits to it.

## 3. Architecture

**Mobile web client + laptop-local Python service that reuses `defid`.**

The phone browser captures behavioral events and POSTs them as JSON to a
FastAPI service running on a laptop on the same LAN. The service runs the
actual validated `defid` scorer. No tunnel and no HTTPS are required:
touch and keystroke events are not secure-context-gated. (An all-in-browser
JavaScript reimplementation was rejected — it would fork the validated math
and destroy the section-2 claim.)

Room visibility: the phone screen is mirrored (AirPlay / screen-share) or a
read-only spectator web view of the same live session is projected, so the
audience watches the Dashboard while one person plays impostor.

## 4. New package: `defid-demo-pkg` (import `defid_demo`)

Matches repo convention (`defid-pkg/` → `defid`, `pad-synth-core/` →
`pad_synth_core`). uv workspace member; pytest with `--import-mode=importlib`;
no `tests/__init__.py`; no `conftest.py`.

Modules:

- **`adapter.py`** — converts browser event JSON into `defid`
  session-shaped dicts: `touch` items `{t, x, y}` (seconds, CSS px),
  `key` items `{t, phase}` where `phase ∈ {"down","up"}`. Timing and
  coordinates only; key *content* is never transmitted from the browser.
  Accepts PointerEvents; falls back to TouchEvents.
- **`windows.py`** — slices one captured rep into windows and computes the
  feature subset per window using the same formulas as `defid.features`.
- **`demo_auth.py`** — composes `defid.models.MahalanobisAuth` with
  covariance shrinkage and empirical threshold calibration. Wraps; does not
  modify `defid`.
- **`qc.py`** — rejects degenerate reps and signals a redo.
- **`app.py`** — FastAPI app: `enroll`, `calibrate`, `attempt`, `reset`
  endpoints; in-memory session state.
- **`web/`** — static mobile capture client + Dashboard UI, plus a
  read-only spectator view.

### 4.1 Feature subset (concrete)

The working subset is the 9 well-conditioned touch+keystroke features:

```
touch_speed_mean, touch_speed_std, touch_curvature_mean,
touch_jitter, inter_touch_interval_mean,
key_dwell_mean, key_dwell_std, key_flight_mean, key_flight_std
```

Excluded and why: the 4 motion features (`accel_mag_mean`,
`tremor_std`, `motion_touch_coupling`, `touch_without_motion_ratio`) —
motion is off the critical path; `key_paste_ratio` — degenerate (~0 for
all subjects) in a live typed-passphrase capture.

Conditioning guard: before fitting, drop any subset feature whose value is
constant across all enrollment windows (zero-variance column), and record
which were dropped (surfaced in the inspect panel for honesty).

### 4.2 Windowing (concrete)

Touch is high-rate (a swipe is many points); keystroke is one summary per
rep (one passphrase). A **window** is a contiguous slice of the rep's touch
stream paired with that rep's keystroke summary. Each rep is split into
`K = 5` overlapping touch sub-windows (50% overlap); each produces one
feature row: the 5 touch features computed on the sub-window ⊕ the rep's 4
keystroke features. Touch columns vary within a rep; keystroke columns vary
across reps; shrinkage (4.3) absorbs the resulting structured correlation.

### 4.3 `demo_auth` (concrete)

- Shrinkage: `cov_shrunk = (1 − α)·cov + α·diag(cov)`, default `α = 0.10`,
  applied before the existing `MahalanobisAuth` `reg=1e-3` diagonal
  loading. `α` is a constructor argument.
- Threshold calibration: hold out the last 2 enrollment reps' windows.
  `threshold = mean(d_holdout) + 3·std(d_holdout)`; if fewer than 4
  hold-out windows exist, fall back to `max(d_holdout) × 1.10`.
- Attempt verdict: an attempt yields `K` window distances. Compute
  `frac_above = fraction of windows with distance > threshold`. Verdict is
  **REJECT** if `frac_above ≥ 0.5`, else **ACCEPT**.

## 5. Data flow

Capture rep → POST → `adapter` → `qc` (redo prompt if bad) → `windows`.
During enrollment, accumulate windows. On `calibrate`: `demo_auth` fits
shrunk `MahalanobisAuth` on enrollment windows and sets the threshold from
held-out genuine distances. Each `attempt`: score its windows, aggregate to
a verdict, return verdict + distances + the per-feature vector. The
Dashboard renders the verdict and gauge on the left and the running
attempt-history scatter (vs. the threshold line) and live feature table on
the right.

## 6. Demo screen — Dashboard layout

Selected layout: **Dashboard**. Left column: live swipe-trace canvas, the
ACCEPT/REJECT verdict, a gauge showing distance vs. threshold. Right
column, always visible: the attempt-history scatter (genuine windows below
the threshold line, impostor windows above) and the live feature table for
the most recent attempt. The accumulating evidence on the right is what
answers the "signal, not luck?" skeptic in real time.

## 7. Demo choreography (the rigor protocol)

1. **Pre-flight** — LAN reachability check screen (phone confirms it can
   reach the service).
2. **Enroll** — enrollee performs 6–8 reps of {guided swipe through
   on-screen targets + type a fixed *public* passphrase}. Progress bar.
   The last 2 reps are held out for calibration.
3. **Calibrate** — threshold computed from held-out genuine distances and
   displayed on screen.
4. **Genuine confirm (false-reject control)** — enrollee performs one
   fresh attempt; it must ACCEPT. Demonstrates the system is not simply
   rejecting everyone.
5. **Impostor round** — at least 3 different people perform the same task;
   each REJECTs; their window distances stack visibly above the threshold
   line on the scatter.
6. **Optional motion bonus** — only if a tunnel is already on hand; never
   blocks steps 1–5.

## 8. Error handling

- Degenerate capture (a tap not a swipe; too few touch points; empty
  typing) → `qc` rejects the rep with a specific redo prompt.
- Poor covariance conditioning after the constant-column drop → the
  service returns a "need N more enrollment reps" state rather than a
  garbage metric; the UI asks for more reps.
- Phone cannot reach the service → caught at pre-flight with a clear
  remediation message (same Wi-Fi, correct LAN URL).
- Browser without PointerEvents → `adapter` falls back to TouchEvents.

## 9. Testing

- **Unit:** `adapter` (recorded browser-event JSON fixtures → expected
  session dicts), `windows` (golden feature vectors), `demo_auth`
  (deterministic shrinkage, threshold calibration, verdict aggregation),
  `qc` (degenerate-capture fixtures).
- **Integration:** a headless full-flow test from recorded event fixtures
  (enroll → calibrate → genuine confirm → impostor round) asserting
  genuine ACCEPT and impostor REJECT separation — the demo without a
  browser. Fully deterministic; no test requires a real human.
- **Manual:** the on-device dry-run following the section-7 protocol.

## 10. Explicit non-goals (YAGNI)

- No accounts, database, or persistence — in-memory session state, reset
  per demo via the `reset` endpoint.
- No synthetic training in the demo loop — live-enroll only.
- No motion on the critical path.
- No native mobile app — this is a mobile *web* client (the native app,
  prior milestone M3, remains a separately deferred plan).
- No production security hardening — trusted-LAN demonstration only.
- The demo does not recompute the offline cross-domain number; it cites
  the PoC's.

## 11. Success criteria

- Enrollment completes in well under a minute on a phone over LAN.
- In a dry run: the enrollee's genuine confirm ACCEPTs and ≥3 distinct
  impostors REJECT, with visible distance separation on the scatter.
- The inspect panel shows the live feature vector, the Mahalanobis
  distance, the threshold, and any dropped constant columns.
- The full automated suite (existing 102 `defid`/PAD tests + new
  `defid_demo` tests) passes; the existing `defid` package is unmodified.
