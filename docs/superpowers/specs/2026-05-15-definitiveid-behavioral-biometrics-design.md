% DefinitiveID — Behavioral Biometrics for Continuous Identity Assurance
% Architecture, Design, Requirements, and Proof-of-Concept Implementation Plan
% 2026-05-15

---

# 1. Executive Summary

**DefinitiveID** is a behavioral-biometrics client that turns an ordinary smartphone (and, later, a paired wearable such as Apple Watch) into a continuous identity sensor. Instead of asking *"did this person present the right credential at login?"* it continuously asks *"is the person using this session still the legitimate owner, and is this a human at all?"*

The product vision targets the capability surface of a system like BioCatch: continuous authentication, bot/automation detection, account-takeover (ATO) signaling, and — with a wearable and richer data — remote-access-trojan (RAT) and social-engineering/scam-under-duress detection.

This document is deliberately split into two horizons:

- **The target architecture** — the full vision (phone + wearable, embeddable SDK, server-side scoring, all four threat models).
- **The proof-of-concept (PoC)** — a fast, basic, two-person build that proves the core mechanic: a synthetic-data-trained model that distinguishes the enrolled user from an imposter and flags non-human (bot) interaction, running on-device in a standalone phone demo app.

The PoC deliberately reuses the engineering spine already built for the synthetic Presentation Attack Detection (PAD) project in this repository — the deterministic generation pipeline, literature-cited parameter ontology, manifest/provenance ledger, and cross-domain evaluation harness. That reuse is the single biggest reason a credible PoC is achievable quickly with two people: the hard, slow infrastructure already exists and is modality-agnostic.

---

# 2. The Fraud Problem in Identity Verification

## 2.1 Credentials are no longer proof of identity

Identity verification historically rested on something the user *knows* (password), *has* (OTP, device), or *is* (a one-time biometric check). Every one of these has been systematically degraded:

- **Knowledge factors** are commoditized. Mass credential breaches and infostealer malware have made valid username/password pairs a cheap, liquid commodity. A correct password is now weak evidence of identity.
- **Possession factors** are phishable. Real-time phishing kits and OTP-relay/adversary-in-the-middle toolkits defeat SMS and even app-based one-time codes by proxying them in real time.
- **One-time biometrics** are increasingly spoofable. This is precisely the problem the companion PAD project in this repository addresses — print, replay, mask, and deepfake presentation attacks against face-verification systems.

The structural weakness is shared by all three: **they are point-in-time checks.** Verification happens once, at the door, and the entire subsequent session inherits that trust. An attacker who clears the door — by phishing, by malware, by social-engineering the victim into doing it for them — owns the session.

## 2.2 The fraud categories this enables

| Fraud category | Mechanism | Why point-in-time auth fails |
|---|---|---|
| **Account takeover (ATO)** | Attacker logs in with stolen/phished credentials | Credentials are valid; the door check passes |
| **Bot / automation fraud** | Scripted clients perform credential stuffing, scraping, fraudulent transactions at scale | Automation can replay a valid credential perfectly |
| **Remote access trojan (RAT)** | Malware lets a remote attacker drive the victim's own authenticated device | The legitimate device and session are used; nothing to "verify" |
| **Authorized push payment (APP) / social-engineering scams** | Victim is manipulated (often by phone) into authorizing a transfer themselves | The legitimate user, on their own device, performs the action under coercion |
| **Synthetic identity / new-account fraud** | Fabricated or stitched identities open accounts | The identity data is internally consistent enough to pass static checks |

Industry fraud reporting (FTC consumer fraud reports, FBI IC3 annual reports, UK Finance and European payment-fraud aggregators) consistently describes the same trend: ATO and authorized-scam fraud are growing faster than traditional card fraud, and synthetic-identity fraud is repeatedly cited as the fastest-growing financial crime category. Exact dollar figures vary widely by source and methodology and are deliberately not quoted as precise numbers here; the directional consensus is what matters for this design — **the fraud has moved from "stolen card" to "compromised or coerced session."**

## 2.3 Why behavioral biometrics

Behavioral biometrics shifts the question from *the credential* to *the behavior of whoever holds the session*. It has four properties that directly counter the failure mode above:

1. **Continuous, not point-in-time.** It scores the whole session, not just the login moment. A session that starts legitimate and is hijacked mid-stream (RAT, hand-off, coercion) shows a behavioral discontinuity.
2. **Passive and frictionless.** It observes how the user already interacts. There is no extra prompt, no extra step, no user-perceived authentication. Friction is the enemy of adoption; behavioral biometrics adds none.
3. **Replay-resistant.** A captured password can be replayed perfectly. A captured behavioral session cannot be re-performed by a different hand or a script without introducing detectable artifacts.
4. **Hard to socially engineer.** A victim can be talked into reading out an OTP. A victim cannot be talked into changing their involuntary motor patterns — and *can* exhibit detectable stress/hesitation signatures when acting under duress (especially with wearable physiological signals).

Behavioral biometrics does not replace credentials or one-time biometrics. It is the **continuous assurance layer** that catches the attacks that survive the front door.

---

# 3. Behavioral Biometrics — Phone Signals

A modern smartphone exposes four families of behavioral signal usable by an SDK embedded in a host application. **Critical platform constraint, stated up front:** a third-party SDK observes behavior *within the host application's own surfaces only.* It cannot perform system-wide keystroke logging or observe other apps. On iOS in particular there is no legitimate system-wide keylogger; keystroke timing is available only for text fields inside the host app. This constraint is not a limitation to apologize for — it is exactly how a legitimate solution (and BioCatch itself) operates: embedded in the relying party's app, scoring interactions with that app.

## 3.1 Touch dynamics

Per-gesture kinematics of taps, swipes, and scrolls within the host app's views:

- Swipe velocity and acceleration profiles
- Trajectory curvature and straightness
- Touch-down pressure / contact area (where the platform exposes it)
- Tap dwell time and inter-tap intervals
- Multi-touch geometry

**Catches:** imposter (different motor signature), bots (perfectly straight or perfectly repeated trajectories, zero natural jitter), automation frameworks (injected events lack realistic kinematics).

## 3.2 Keystroke dynamics

Timing of text entry inside the host app's input fields:

- Dwell time (key-down to key-up)
- Flight time (key-up to next key-down)
- Per-digraph/trigraph rhythm
- Correction and pause patterns

**Catches:** imposter (typing rhythm is highly individual), bots/paste (instantaneous "typing" or paste events have no human rhythm), scripted form-fill, and — relevant to scam detection — the difference between *recalling your own data fluently* and *transcribing dictated data hesitantly*.

## 3.3 Device motion

Accelerometer and gyroscope sampled while the user interacts:

- Device orientation and micro-tremor while holding
- Motion coupled to touch events (a real tap perturbs the device; an injected event does not)
- Pick-up / put-down transitions, gait while walking and using the phone

**Catches:** bots and emulators (no physical device motion, or unrealistically static motion), RAT/remote control (interaction occurs with no corresponding device motion — a strong tell), and contributes to liveness.

## 3.4 Navigation and session cadence

Higher-level interaction structure within the app:

- Screen-to-screen navigation paths and timing
- Dwell time per screen, hesitation before sensitive actions
- Session rhythm and time-of-day patterns

**Catches:** ATO (a different person navigates a familiar app differently), scripted fraud flows (machine-like efficiency, no exploratory behavior), and scam/duress (atypical hesitation or atypical directness toward a high-value action).

## 3.5 What each signal does and does not give you

| Signal | Strongest for | Weak/unavailable for |
|---|---|---|
| Touch dynamics | Continuous auth, bot detection | Requires touch-heavy UI; sparse in form-only flows |
| Keystroke dynamics | Continuous auth, paste/bot, scam-transcription | Only host-app fields; sparse if little typing |
| Device motion | Bot/emulator, RAT, liveness | Stationary legitimate use looks similar to some bots |
| Navigation cadence | ATO, scripted-flow, scam-directness | Needs enough session history; noisier per-event |

No single signal is sufficient. The design fuses them.

---

# 4. Adding the Wearable (Target Vision — Milestone 2)

A paired wearable (Apple Watch via HealthKit + Core Motion; Wear OS analogues) adds a physiological and second-vantage-point channel that the phone alone cannot provide. This is **out of scope for the PoC** and **in scope for the target architecture.**

## 4.1 New signals the wearable contributes

- **Heart rate and heart-rate variability (HRV).** Accessed through the platform health framework with explicit user permission. HRV is a recognized correlate of acute stress. A user authorizing a routine transfer and a user authorizing a transfer while being coached by a scammer on the phone are physiologically different.
- **Wrist motion and gait.** Independent inertial sensing on the wrist.
- **Liveness and presence.** A live wrist signal indicates a real, present human, not an emulator farm.

## 4.2 The high-value capability: phone↔wearable coherence

The single most valuable wearable feature is **cross-device motion coherence.** When a real person taps their phone, the hand holding/operating it moves, and — if the watch is on that wrist or the other hand braces the phone — there is a correlated micro-motion signature across the two devices. Under **remote control (RAT)**, the phone receives interaction events with **no corresponding wrist motion**, because the human is not the one driving the device. This phone-says-active / wrist-says-still divergence is a powerful, hard-to-forge RAT signal that neither device produces alone.

Similarly, **scam/duress detection** is materially strengthened: phone-side hesitation (navigation cadence, keystroke transcription pattern) correlated with wrist-side physiological stress (elevated HR, suppressed HRV) is a far stronger joint signal than either side alone.

## 4.3 Honest constraints

- Wearable physiological data is sensitive health-adjacent personal data; permission, purpose limitation, and minimization are mandatory (Section 6.6).
- Not all users own a paired wearable; the wearable channel must be strictly *additive* — the system degrades gracefully to phone-only.
- Real-time streaming of wearable sensors has platform and battery constraints; the design assumes windowed/batched features, not continuous high-rate streaming.

---

# 5. Architecture

## 5.1 Target architecture (the vision)

```
 Host mobile app
   └── DefinitiveID SDK (collection only)
         • signal collectors: touch, keystroke, motion, navigation
         • on-device windowing + privacy filter (data minimization)
         • batched, encrypted telemetry upload
                    │
                    ▼
 Telemetry ingestion service  ──►  Feature service
   • schema validation               • windowed feature extraction
   • per-tenant isolation            • enrollment profile store (per user)
                                      │
                                      ▼
                              Scoring service
                                • continuous-auth model (user vs imposter)
                                • bot/automation classifier
                                • (vision) ATO / RAT / scam models
                                      │
                                      ▼
                              Risk decision API
                                • risk score + reason codes
                                • policy thresholds per relying party
                                      │
                                      ▼
                    Host app / fraud platform (step-up, block, review)
```

Key properties: the SDK **collects but does not decide**; scoring is server-side so models can be updated without shipping a new app build; every component is independently testable; per-tenant data isolation is a first-class concern.

## 5.2 PoC architecture (the fast first slice)

```
 Standalone demo app (single phone app, no backend)
   ├── signal collectors: touch, keystroke, motion  (navigation stubbed)
   ├── on-device feature extraction (windowed)
   ├── bundled models (trained offline on synthetic data):
   │     • continuous-auth model (enrolled user vs imposter)
   │     • bot/automation classifier
   └── on-device risk readout (score + reason codes shown in the demo UI)

 Offline (developer machine, reusing pad-synth-core spine):
   ├── behavioral synthetic-data generator (deterministic, ontology-driven)
   ├── manifest + provenance ledger
   ├── model training
   └── cross-domain evaluation (synthetic → Touchalytics sanity check)
```

The PoC has **no server**: collection and scoring are on-device, models are trained offline and bundled. This is a deliberate first slice of the target architecture, not a throwaway — the SDK collectors, feature extraction, and model interfaces are the same components that later move behind the SDK/server split.

## 5.3 Component responsibilities

| Component | Responsibility | PoC? |
|---|---|---|
| Signal collectors | Capture raw touch/keystroke/motion events within the app | Yes |
| Privacy filter | Drop/transform content; keep only behavioral features | Yes |
| Windowing | Segment event streams into fixed analysis windows | Yes |
| Feature extractor | Window → fixed-length feature vector | Yes |
| Enrollment store | Per-user behavioral profile (PoC: single local profile) | Yes (local) |
| Continuous-auth model | Score: is this the enrolled user? | Yes |
| Bot classifier | Score: is this human or automation? | Yes |
| Synthetic data generator | Produce labeled behavioral sessions (reuses pad-synth-core) | Yes (offline) |
| Telemetry/ingestion/scoring services | Server-side target architecture | No (vision) |
| ATO / RAT / scam models | Advanced threat models | No (vision) |
| Wearable channel | Apple Watch / Wear OS signals | No (vision) |

---

# 6. Design

## 6.1 Signal collection

Each collector emits timestamped, typed events into a common in-memory ring buffer. Event schema (illustrative):

- `touch`: `{t, phase(down|move|up), x, y, pressure?, area?, view_id}`
- `key`: `{t, phase(down|up), field_id}` — **key identity is never recorded**, only timing and field identity
- `motion`: `{t, ax, ay, az, gx, gy, gz}` sampled at a fixed rate (e.g., 50 Hz) during interaction
- `nav`: `{t, screen_id, transition}` (stubbed in PoC)

The privacy filter is applied at the collector boundary: typed content is never buffered; only behavioral metadata is. This is a design invariant, not a configuration option.

## 6.2 Windowing and feature extraction

Event streams are segmented into overlapping fixed-duration windows (e.g., 5–10 s, 50% overlap). Each window is reduced to a fixed-length feature vector:

- Touch: velocity/acceleration statistics, curvature, dwell/flight distributions, jitter entropy
- Keystroke: dwell/flight means and dispersions, rhythm regularity, paste-event ratio
- Motion: orientation statistics, micro-tremor spectral features, motion-coupled-to-touch correlation
- Cross-signal: touch-event-without-motion ratio (bot/RAT tell), input-without-jitter ratio (automation tell)

The feature schema is versioned and recorded in the manifest, exactly as ontology versions are recorded in the PAD pipeline.

## 6.3 Enrollment vs. continuous verification

- **Enrollment:** the first N legitimate windows establish a per-user profile (PoC: a single on-device profile created in a guided enrollment screen).
- **Continuous verification:** each subsequent window is scored against the profile; scores are smoothed over a sliding horizon to produce a session risk trajectory rather than a noisy per-window flag.

## 6.4 Model design

- **Continuous auth (user vs imposter):** framed as metric learning / one-class scoring. The model learns an embedding in which the enrolled user's windows cluster; a window's distance to the enrolled profile is the imposter score. This avoids needing per-deployment retraining and degrades gracefully with little enrollment data. PoC baseline: a small embedding network with a one-class/distance head — analogous in spirit to the PAD project's "start with a small baseline, escalate only if it plateaus" principle.
- **Bot/automation detection:** a separate binary classifier on the regularity/jitter/motion-coupling features. Bots are separable from humans on these features without per-user modeling.

The two models are independent and fused at the decision layer (a window can be "the right user but a bot replaying them," or "a human but the wrong person").

## 6.5 Synthetic data generation (the speed multiplier)

The PoC trains on **synthetic behavioral sessions** generated by reusing the existing `pad-synth-core` spine. The mapping is direct:

| PAD project concept | DefinitiveID analogue |
|---|---|
| Attack-parameter ontology (literature-cited YAML) | Behavioral-parameter ontology: touch/keystroke/motion parameter ranges with citations to the behavioral-biometrics literature (HMOG, Touchalytics, keystroke-dynamics studies) |
| Deterministic seed derivation | Same module, unchanged |
| Manifest + provenance ledger | Same schema; records which behavioral ontology + generator version produced each session |
| Per-sample QC | Behavioral-plausibility QC (e.g., velocities within human range, no NaN) |
| Cross-domain eval (Set A → Set B) | Synthetic → Touchalytics sanity check (does a synthetic-trained model do better than chance on a small real public dataset?) |

The generator produces three labeled session classes for the PoC: **genuine** (a parameterized "owner" profile with natural jitter), **imposter** (different motor parameters), and **bot** (degenerate jitter, machine timing, no motion coupling). Literature-cited parameter ranges keep the synthesis defensible and prevent the model from learning a degenerate synthetic fingerprint — the same discipline that protected the PAD work.

## 6.6 Privacy and consent design

- **Data minimization by construction:** content is never buffered; only behavioral features leave the collector boundary.
- **On-device where possible:** the PoC scores entirely on-device; nothing leaves the phone.
- **Explicit consent and purpose limitation:** behavioral biometrics is personal data (and, with a wearable, health-adjacent). The target architecture requires informed consent, a stated purpose, retention limits, and per-tenant isolation — the same regulatory hygiene flagged for face data in the PAD project.
- **Defensive use only:** DefinitiveID is a fraud-prevention control. It is not, and must not be repurposed as, a covert tracking or deanonymization tool. This constraint is stated in the spec so it cannot be silently dropped.

---

# 7. Requirements

## 7.1 Functional requirements

- **FR1** — The SDK/app shall collect touch, keystroke-timing, and motion events within the host app's surfaces, applying the content-drop privacy filter at the collector boundary.
- **FR2** — The system shall segment events into fixed, overlapping windows and produce a versioned fixed-length feature vector per window.
- **FR3** — The system shall support an enrollment mode that builds a per-user behavioral profile.
- **FR4** — The system shall produce a continuous per-window imposter score against the enrolled profile and a smoothed session risk trajectory.
- **FR5** — The system shall produce a per-window bot/automation score independent of the user profile.
- **FR6** — The system shall expose a combined risk readout with human-readable reason codes.
- **FR7** — The offline pipeline shall generate labeled synthetic behavioral sessions deterministically, recording ontology and generator versions in a manifest + provenance ledger.
- **FR8** — The offline pipeline shall report a synthetic→real cross-domain evaluation metric.

## 7.2 Non-functional requirements

- **NFR1 (Latency):** per-window scoring on-device shall complete within one window interval (no backlog).
- **NFR2 (Battery/footprint):** motion sampling and inference shall be duty-cycled to interaction periods; bundled models shall be small enough for on-device inference on a mid-range phone.
- **NFR3 (Privacy):** no typed content shall ever be persisted or transmitted; this is verified by test, not assumed.
- **NFR4 (Reproducibility):** synthetic generation shall be byte-deterministic from `(config, seed)` and guarded by a golden test, mirroring the PAD pipeline's determinism contract.
- **NFR5 (Graceful degradation):** absence of any one signal (e.g., no typing in a session) shall not break scoring; the wearable channel shall be strictly additive.
- **NFR6 (Portability):** the offline pipeline shall run without specialized hardware; on-device inference shall not require a GPU.

## 7.3 Explicitly out of scope for the PoC

- Server-side telemetry/ingestion/scoring services
- ATO, RAT, and social-engineering/scam models
- Wearable (Apple Watch / Wear OS) integration
- Real-user data collection
- Production SDK packaging, multi-tenant isolation, model-update channel
- Navigation-cadence collector (stubbed)

These are documented in the target architecture and roadmap so the PoC is a deliberate first slice, not an architectural dead end.

---

# 8. Implementation Plan (Proof of Concept)

Two engineers. Milestones are independently demonstrable and ordered so that something works end-to-end as early as possible.

## 8.1 Milestones

| ID | Milestone | Deliverable | Reuses from `pad-synth-core` |
|---|---|---|---|
| **M0** | Behavioral synthetic-data generator | Deterministic generator producing genuine/imposter/bot labeled sessions; behavioral ontology YAML with cited ranges; manifest + provenance | RNG, manifest, provenance, ontology loader, QC, determinism golden pattern |
| **M1** | Feature extraction + continuous-auth model | Windowing + feature extractor; metric/one-class model; offline training; in-domain EER reported | eval/baseline patterns, cross-domain eval harness |
| **M2** | Bot/automation detector | Binary classifier on regularity features; fused decision layer | eval harness |
| **M3** | Standalone phone demo app | iOS (or Android) app: collectors → on-device feature extraction → bundled models → live risk readout UI; guided enrollment | — (new mobile surface) |
| **M4** | Cross-domain sanity + report | Synthetic→Touchalytics evaluation; metrics report; decision on whether the synthetic signal is real | cross-domain eval harness, decisions-report pattern |

## 8.2 Two-person split

- **Engineer A (data + models):** M0, M1, M2, M4 — owns the offline pipeline and the reuse of `pad-synth-core`.
- **Engineer B (client):** M3 — owns the mobile app, collectors, on-device inference integration; consumes the model artifacts and feature schema from Engineer A.
- Shared contract: the **feature schema** (versioned) and the **bundled model interface**. Agree these in M1 so M3 can proceed in parallel against a stub.

## 8.3 Sequencing

M0 → M1 unlock the data and the core model. M2 is additive. M3 can start as soon as the feature schema is fixed (mid-M1) and integrates real models when M1/M2 land. M4 is the go/no-go: does a synthetic-trained model beat chance on real public data? That is the PoC's equivalent of the PAD project's cross-domain EER gate.

## 8.4 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Synthetic behavior is too "clean" → model learns a synthetic fingerprint, fails on real data | High | Literature-cited parameter ranges + jitter modeling; M4 cross-domain check is the explicit gate (same discipline that protected the PAD project) |
| Platform limits on signal access (esp. iOS keystroke/motion) reduce signal | Medium | Design assumes host-app-only signals from the start; touch + motion alone are sufficient for the PoC's continuous-auth + bot scope |
| 2-person bandwidth vs. mobile-app surface | Medium | M3 scoped to a single demo app, on-device, no backend; navigation collector stubbed |
| Privacy/regulatory exposure | Medium | On-device-only PoC, content-drop filter as a tested invariant, defensive-use-only constraint stated in the spec |
| Over-reach into RAT/scam | Medium | Explicitly deferred; documented as vision with their data dependencies stated |

---

# 9. Roadmap Beyond the PoC

1. **SDK + server split.** Promote the on-device collectors into a distributable SDK; move scoring server-side for model-update agility and multi-tenant isolation.
2. **Wearable channel (Milestone 2 of the vision).** Apple Watch / Wear OS: HR/HRV, wrist motion, and the high-value phone↔watch motion-coherence RAT signal.
3. **ATO as a derived signal.** Operationalize continuous-auth divergence into an account-takeover decision with policy thresholds.
4. **RAT and scam/duress models.** Data-gated; require fraud-labeled sessions. Pursue via (a) synthetic generation of remote-control and duress signatures, and (b) partnerships for labeled data — the same "synthetic-first, validate against real" arc as the PAD project.
5. **Real-data validation.** Beyond the Touchalytics sanity check: HMOG, HuMIdb, BB-MAS, UMDAA-02 for continuous-auth realism; this is the equivalent of the PAD project's deferred real-PAD-dataset integration.
6. **Metric alignment for external comparison.** Report APCER/BPCER-style operating points and standard continuous-auth metrics so results can be positioned against published behavioral-biometrics literature.

---

# 10. Appendix

## 10.1 Signal / feature quick reference

| Family | Raw signal | Example features | Primary threat value |
|---|---|---|---|
| Touch | down/move/up, x/y, pressure, area | velocity/accel stats, curvature, dwell/flight, jitter entropy | Continuous auth, bot |
| Keystroke | key down/up timing (no content) | dwell/flight stats, rhythm regularity, paste ratio | Continuous auth, bot, scam-transcription |
| Motion | accel/gyro @ ~50 Hz during interaction | orientation stats, micro-tremor spectrum, motion-coupled-to-touch | Bot/emulator, RAT, liveness |
| Navigation | screen transitions, dwell | path entropy, hesitation-before-sensitive-action | ATO, scripted-flow, scam-directness |
| Wearable (vision) | HR, HRV, wrist motion | stress correlates, gait, phone↔watch coherence | RAT, scam/duress, liveness |

## 10.2 Public datasets (for real-data validation)

- **HMOG** (Sitová et al.) — hand movement, orientation, grasp; phone, touch + motion, ~100 subjects.
- **Touchalytics** (Frank et al.) — touchscreen swipe dynamics; the PoC's M4 sanity-check set.
- **HuMIdb** (Acien et al.) — multimodal mobile behavior.
- **BB-MAS** (Belman et al.) — multi-device behavioral biometrics.
- **UMDAA-02** (Mahbub et al.) — continuous mobile authentication.

All are research-scale, not fraud-labeled — they validate continuous-auth realism, not ATO/RAT/scam. Eval-only use, mirroring the PAD project's posture toward research-license datasets.

## 10.3 Glossary

- **Continuous authentication:** ongoing, passive verification that the session's operator is still the enrolled user.
- **ATO:** account takeover — valid credentials used by the wrong person.
- **RAT:** remote access trojan — malware giving a remote attacker control of the victim's authenticated device.
- **APP fraud / scam-under-duress:** the legitimate user, manipulated, performs the fraudulent action themselves.
- **EER:** equal error rate — operating point where false-accept and false-reject rates are equal; the project's headline model metric, carried over from the PAD work.
- **Cross-domain evaluation:** train on one distribution, evaluate on a deliberately different one; the project's discipline for detecting overfitting to a synthetic distribution.
