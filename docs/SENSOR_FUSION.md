# Sensor Fusion

> Design reference for CREBAIN's multi-target tracking and sensor-fusion subsystems —
> the math, the data contracts, the tuning knobs, and the known limitations.

CREBAIN fuses detections from heterogeneous sensors (visual, thermal, acoustic,
radar, lidar, RF) into a small set of persistent **tracks** — each an estimate of a
target's 3D position, velocity, classification, uncertainty, and threat level. This
document explains how that works, why it is built the way it is, and where the
edges are.

It is written to be read alongside the code. Primary sources:

| File | Role |
|------|------|
| `src-tauri/src/sensor_fusion.rs` | Native multi-modal tracking engine (KF/EKF/UKF/PF/IMM) |
| `src/detection/AdvancedSensorFusion.ts` | TypeScript bridge to the native engine (Tauri IPC + response validation) |
| `src/ros/useROSSensors.ts` | Converts ROS sensor messages into fusion measurements |
| `src/detection/SensorFusion.ts` | Browser-only multi-camera correlation + triangulation |
| `src/detection/types.ts` | Shared detection / track / threat types |
| `src/components/SensorFusionPanel.tsx` | Operator-facing track list and filter selector |

---

## Table of Contents

- [Two fusion subsystems](#two-fusion-subsystems)
- [The estimation pipeline](#the-estimation-pipeline)
- [Coordinate frames and the measurement contract](#coordinate-frames-and-the-measurement-contract)
- [Filter algorithms](#filter-algorithms)
- [Data association and gating](#data-association-and-gating)
- [Multi-sensor fusion semantics](#multi-sensor-fusion-semantics)
- [Track lifecycle](#track-lifecycle)
- [Threat assessment](#threat-assessment)
- [Multi-camera triangulation](#multi-camera-triangulation)
- [Configuration and tuning](#configuration-and-tuning)
- [Validation and metrics](#validation-and-metrics)
- [Known limitations and roadmap](#known-limitations-and-roadmap)
- [References](#references)

---

## Two fusion subsystems

CREBAIN contains **two independent fusion engines** that serve different sensor
geometries. They are not competing implementations — they sit at different points
in the pipeline.

```mermaid
graph LR
    subgraph Browser["Browser (TypeScript)"]
        Cams["Multi-camera<br/>detections + intrinsics"] --> TSF["SensorFusion.ts<br/>correlate → triangulate → track"]
        TSF --> TSTracks["FusedTrack[]<br/>(THREE.Vector3)"]
    end

    subgraph Native["Native engine (Rust, via Tauri)"]
        ROS["ROS sensor topics<br/>(thermal/acoustic/radar/lidar)"] --> Bridge["useROSSensors.ts<br/>+ AdvancedSensorFusion.ts"]
        Visual["CoreML / YOLO<br/>visual detections"] --> Bridge
        Bridge -->|"invoke('fusion_process')"| Engine["MultiSensorFusion<br/>predict → associate → update → lifecycle"]
        Engine --> Out["TrackOutput[]"]
        Out --> Panel["SensorFusionPanel"]
    end
```

### 1. Native multi-modal engine — `sensor_fusion.rs`

The primary engine. It maintains a recursive Bayesian estimate per target using a
selectable filter (Kalman, Extended Kalman, Unscented Kalman, Particle, or IMM),
associates incoming measurements to tracks with a Mahalanobis gate, and manages the
full track lifecycle. It runs in Rust for performance and numerical control, and is
reached from the UI over Tauri IPC. This is the engine the **Sensor Fusion panel**
displays and the one this document is mostly about.

### 2. Browser multi-camera engine — `SensorFusion.ts`

A self-contained TypeScript engine for the special case of **several calibrated
RGB cameras observing the same scene**. It correlates 2D detections across cameras,
triangulates a 3D position from the back-projected rays, and runs a lightweight
track manager. It exists so the 3D viewer can show fused camera tracks without a
round trip to the native backend.

The remainder of this document treats the native engine as the default and calls
out the browser engine explicitly in [Multi-camera triangulation](#multi-camera-triangulation).

---

## The estimation pipeline

Each call to `fusion_process(measurements, timestamp_ms)` runs one cycle of a
standard recursive multi-target tracker:

```mermaid
flowchart TD
    A["measurements[]<br/>(this frame)"] --> B["1. PREDICT<br/>advance every track to now<br/>x' = F·x,  P' = F·P·Fᵀ + Q·dt"]
    B --> C["2. ASSOCIATE<br/>gate by Mahalanobis distance,<br/>pick nearest track per measurement"]
    C --> D["3. UPDATE<br/>fuse associated measurements,<br/>correct state + covariance"]
    C --> E["4. INITIATE<br/>spawn Tentative track from<br/>each unassociated measurement"]
    D --> F["5. LIFECYCLE<br/>age, confirm, coast,<br/>delete missed tracks"]
    E --> F
    F --> G["TrackOutput[]"]
```

1. **Predict** — every existing track is advanced from its last update to the
   current frame time using the constant-velocity motion model. `dt` is computed
   from the actual frame timestamps (clamped to ≤ 1 s) so the prediction stays
   correct under a variable frame rate.
2. **Associate** — see [Data association and gating](#data-association-and-gating).
3. **Update** — associated measurements correct the track's state and shrink its
   covariance through the selected filter.
4. **Initiate** — measurements that matched no track seed a new `Tentative` track
   (subject to `MAX_FUSION_TRACKS`).
5. **Lifecycle** — tracks are aged, promoted, coasted, or deleted (see
   [Track lifecycle](#track-lifecycle)).

The **state vector** is 6-dimensional: `x = [px, py, pz, vx, vy, vz]` (position in
meters, velocity in m/s, common world frame). The engine **observes position only**
(the measurement matrix is `H = [I₃ | 0₃]`); velocity is inferred by the filter
from the position sequence.

---

## Coordinate frames and the measurement contract

> **This is the single most important contract to get right.** A coordinate-frame
> mismatch silently corrupts every track for the affected modality.

A `SensorMeasurement` carries a `position`, an optional Cartesian `velocity` seed,
and a diagonal measurement-noise `covariance`. The frame of `position` and
`covariance` is **selected by modality**:

| Modality | `position` frame | `covariance` units | Notes |
|----------|------------------|--------------------|-------|
| `radar` | **polar** `[range_m, azimuth_rad, elevation_rad]` | `[m², rad², rad²]` | Native radar geometry; consumed directly by the EKF polar model |
| `lidar` | Cartesian `[x, y, z]` m | `[m², m², m²]` | A precise 3D centroid — **not** a polar sensor |
| `visual` | Cartesian `[x, y, z]` m | `[m², m², m²]` | From triangulation / projection |
| `thermal` | Cartesian `[x, y, z]` m | `[m², m², m²]` | |
| `acoustic` | Cartesian `[x, y, z]` m | `[m², m², m²]` | DOA + range estimate, converted on the producer side |

`velocity`, when present, is **always Cartesian** `[vx, vy, vz]` and is only used to
seed a new track's initial velocity. Radar producers project the scalar radial
velocity onto the line of sight before sending it.

The frame is interpreted in Rust by two helpers:

- `measurement_position_cartesian()` — used for association and track initiation.
  For `radar` it converts polar → Cartesian; for everything else it passes
  `[x, y, z]` through.
- `measurement_position_polar()` — returns `Some([range, az, el])` **only for
  radar**, feeding the EKF polar update; `None` otherwise.

> **Historical note.** Radar/lidar measurements were previously converted
> spherical→Cartesian on the TypeScript side *and* re-interpreted as polar in Rust,
> a double conversion that corrupted both modalities. The contract above (radar
> stays polar end-to-end; lidar is Cartesian) is the corrected design — see the
> regression tests `radar_measurement_creates_cartesian_track_from_polar_input` and
> `lidar_centroid_is_treated_as_cartesian_not_polar`.

**Producers must emit the frame that matches their modality.** Two consequences
worth internalizing:

- A sensor that natively reports angles (radar) should keep them as angles so the
  EKF can model the (curved) polar error correctly. Converting to Cartesian first
  and then trusting a diagonal Cartesian covariance throws away the real error
  shape.
- A sensor that natively reports a metric 3D point (lidar) must **not** be routed
  through a polar conversion — doing so fabricates angular error and discards its
  main advantage.

---

## Filter algorithms

All five filters are instances of **recursive Bayesian estimation**: they keep a
Gaussian belief `(x, P)` and alternate a *predict* (time update) with an *update*
(measurement update). They differ only in how they handle nonlinearity and
multi-modality.

```text
PREDICT:   x' = F·x                 P' = F·P·Fᵀ + Q
UPDATE:    y  = z − H·x'            (innovation)
           S  = H·P'·Hᵀ + R         (innovation covariance)
           K  = P'·Hᵀ·S⁻¹           (Kalman gain)
           x  = x' + K·y            P = (I − K·H)·P'  [Joseph form below]
```

`F` is the constant-velocity transition (`position += velocity·dt`); `Q` is the
process noise (un-modeled acceleration / maneuvers); `R` is the per-sensor
measurement noise.

| Filter | Selector | Best for | Cost | How it handles nonlinearity |
|--------|----------|----------|------|------------------------------|
| **Kalman (KF)** | `Kalman` | Linear, Cartesian, constant-velocity targets | Lowest | N/A — assumes linear models |
| **Extended Kalman (EKF)** | `ExtendedKalman` *(default)* | Radar / polar measurements with mild nonlinearity | Low | First-order Jacobian linearization about the estimate |
| **Unscented Kalman (UKF)** | `UnscentedKalman` | Stronger nonlinearity than EKF tolerates | Medium | Deterministic sigma points through the true model (derivative-free) |
| **Particle (PF)** | `Particle` | Non-Gaussian / multi-modal posteriors | High (`O(N)` per track) | Weighted Monte-Carlo sample set |
| **IMM** | `IMM` | Maneuvering targets switching dynamics | Medium | Bank of models mixed by a Markov chain |

### Numerical hardening

The covariance update uses the **Joseph stabilized form** in the KF and EKF paths:

```text
P = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ
```

This is algebraically equal to `(I − K·H)·P` for the optimal gain, but is a sum of
two symmetric positive-semidefinite terms, so it **preserves symmetry and PSD under
finite-precision arithmetic** — the property that keeps the UKF's Cholesky
decomposition from failing and the Mahalanobis gate well-defined. The UKF
additionally symmetrizes `P` after each update. As defense-in-depth, the sigma-point
generator retains a diagonal fallback when Cholesky still fails, and `TrackOutput`
clamps any negative covariance diagonal before reporting uncertainty.

### EKF polar model (radar)

For radar, the measurement function maps Cartesian state to polar observation:

```text
h(x) = [ √(x²+y²+z²),  atan2(y, x),  asin(z / range) ]   →  [range, azimuth, elevation]
```

The EKF linearizes `h` with its analytic Jacobian, guards the range singularity at
the origin (`range = max(√(x²+y²+z²), 1e-6)`), and wraps the azimuth innovation to
`(−π, π]` so a measurement near the ±π boundary does not produce a spurious 2π jump.

### IMM details

The IMM runs a bank of two constant-velocity Kalman filters with different process
noise (a low-Q "cruise" model and a high-Q "maneuver" model), mixes their estimates
each cycle via a fixed Markov transition matrix, updates the mode probabilities from
each model's Gaussian innovation likelihood, and outputs a moment-matched combined
estimate. The likelihood uses the correct 3-D multivariate-Gaussian normalizer
`√((2π)³·det S)`.

---

## Data association and gating

Before any filter update, the engine decides **which measurement updates which
track**. This is a two-stage process.

### Gating

For a track with predicted position `Hx` and innovation covariance `S = HPHᵀ + R`,
a candidate measurement `z` is gated by its Mahalanobis distance to that track.
Folding the measurement noise `R` into `S` (rather than gating on the track
covariance alone) is what makes this a proper Mahalanobis gate: it keeps the gate
from collapsing once `P` shrinks below `R` for a confident track, which would
otherwise reject ~1σ-valid measurements and spawn duplicate tracks.

A measurement is admissible when the **squared** Mahalanobis distance
`d² = (z − Hx)ᵀ S⁻¹ (z − Hx)` is below `association_threshold`. Because `d²` is
χ²-distributed with degrees of freedom equal to the **measurement** dimension (3 for
position), the threshold is a χ²(3) quantile: the default `11.345` is the 99 % gate
(`9.348` ≈ 97.5 %, `7.815` ≈ 95 %). Raising it admits more candidates (more clutter,
fewer missed associations); lowering it tightens the gate.

If `S` is singular (degenerate covariance — rare in practice since `Q, R > 0`), the
gate falls back to a Euclidean distance normalized by a nominal per-axis sigma so it
stays on the same unitless scale as the Mahalanobis branch.

The gate runs in the **Cartesian** position frame, so each measurement's noise `R`
must be supplied in that frame. Cartesian modalities use their diagonal covariance
directly; radar's noise is polar `[m², rad², rad²]`, so it is propagated into
Cartesian via the polar→Cartesian Jacobian
(`R_cart = J⁻¹ R J⁻ᵀ`, `J = ∂(range,az,el)/∂(x,y,z)`) before being folded into `S`.
Skipping this conversion would add `rad²` to `m²` and badly under-estimate
cross-range uncertainty — an angular 1σ at range `R` spans ≈ `R·σ_angle` in
cross-range, not `σ_angle` metres.

### Assignment

CREBAIN uses **greedy nearest-neighbour** assignment: each measurement is assigned to
its single closest gated track. This is fast and adequate for sparse, well-separated
targets, but it does **not** enforce a one-measurement-to-one-track constraint and is
order-dependent — in dense or crossing scenarios an early greedy pick can steal a
measurement that optimally belonged to another track. A global assignment
(Hungarian / auction) is the standard upgrade and is on the roadmap.

---

## Multi-sensor fusion semantics

When several measurements (e.g. radar + thermal) associate to one track in a single
frame, the engine fuses them into one effective position by a **confidence-weighted
average** and applies a single update, plus a small per-extra-modality confidence
boost.

This is pragmatic and cheap, but it is worth being explicit about what it is and is
not:

- **What it is:** a heuristic that pulls the track toward whichever detector reported
  higher confidence, with a multi-sensor confidence bump for display/threat logic.
- **What it is not:** statistically optimal fusion. The optimal combination is
  *information-form* (inverse-covariance) weighting — `x̂ = (ΣCᵢ⁻¹)⁻¹ ΣCᵢ⁻¹xᵢ` — which
  weights each sensor by its actual precision. Detector confidence is not the same as
  measurement precision: a lidar centroid (centimeter-accurate) and an acoustic
  bearing (tens-of-meters-accurate) should not be averaged with equal geometric
  weight just because their confidences are similar.

The principled alternative is to apply each associated measurement **sequentially**
through its own measurement model and covariance `R`, rather than pre-averaging. This
is tracked in [Known limitations](#known-limitations-and-roadmap).

---

## Track lifecycle

A track moves through four states. The transitions below reflect the actual Rust
implementation in `update_track` and `handle_missed_detections` (defaults:
`min_confirmation_hits = 3`, `max_missed_detections = 5`).

```mermaid
stateDiagram-v2
    [*] --> Tentative: unassociated measurement<br/>(new track)

    Tentative --> Confirmed: hit count (age) ≥ 3
    Tentative --> Coasting: 2 consecutive misses
    Tentative --> Lost: 5 consecutive misses

    Confirmed --> Coasting: 2 consecutive misses
    Coasting --> Confirmed: re-associated (if age ≥ 3)
    Coasting --> Lost: 5 consecutive misses

    Lost --> [*]: removed from the track table
```

- **Tentative** → **Confirmed** once the track's hit counter (`age`, incremented on
  each association) reaches `min_confirmation_hits`.
- Any live track → **Coasting** once it accumulates `≥ 2` consecutive missed frames.
  A coasting track keeps being *predicted* forward by its motion model (the
  covariance grows), which is what bridges short occlusions and sensor dropouts.
- Any live track → **Lost** at `max_missed_detections` consecutive misses, at which
  point it is **removed** from the table (and its per-track PF/IMM filter state is
  freed).
- A coasting track that is re-associated resets its miss counter and returns to
  Confirmed (if it had already reached the confirmation hit count).

At **birth**, a track is created by *single-point initiation*: its position is the
measurement and its velocity is unknown. The velocity block of the initial covariance
is therefore seeded with a deliberately **wide** prior
(`INITIAL_VELOCITY_VARIANCE_M2_S2`, σ_v ≈ 20 m/s), not a tight one. A position-only
measurement (radar without Doppler, lidar, visual) carries no velocity information, so
an over-confident birth velocity would make the constant-velocity prediction reject
the next frame of a genuinely moving target through the (correctly tightened) χ²(3)
gate — fragmenting one target into duplicate tracks. The wide prior only ever eases
the *first* post-birth association; returns that are far in absolute terms are still
gated out.

> **Implementation note.** Confirmation keys off a monotonic hit counter, not a true
> sliding-window *M-of-N* rule. A real M-of-N ("M hits in the last N opportunities")
> would let an intermittently-dropping track lose evidence toward confirmation. This
> is a known simplification — see the roadmap.

---

## Threat assessment

Each track is assigned an integer **threat level from 1 to 4** (there is no level 0).
A single canonical formula is implemented identically in Rust
(`calculate_threat_level`) and TypeScript (`getThreatLevel`, `src/detection/types.ts`):

| Class | Level |
|-------|-------|
| drone / uav | 4 if confidence > 0.8, 3 if > 0.5, else 2 |
| aircraft / helicopter | 2 |
| bird | 1 |
| unknown / unrecognized | 3 if confidence > 0.7, else 2 |

Drone threat is **graduated** by confidence: a low-confidence (≤ 0.5) single-sensor
drone hypothesis stays "guarded" (2) to avoid flooding the operator with amber from
uncorroborated returns, rising to "elevated" (3) and then "severe" (4) as confidence
(e.g. from multi-sensor corroboration) grows. A confidently-tracked but unidentified
object is treated as "elevated" (3). All thresholds are strict `>`. Crucially, the
**same raw label is bucketed identically on both engines**: the Rust
`map_to_detection_class` mirrors the TypeScript `mapToDetectionClass`, so a label is
resolved to its canonical class once, the same way, before the shared formula runs.
Colors: `1` green (minimal), `2` blue (guarded), `3` amber (elevated), `4` red
(severe).

> A compound or unrecognized label (e.g. `"fpv-drone"`) resolves to `unknown` on
> both engines — not `drone` — because the shared mapping is exact-match. Extending
> the recognized vocabulary is a `mapToDetectionClass`/`map_to_detection_class`
> enhancement that automatically keeps both engines in step.

---

## Multi-camera triangulation

The browser engine (`SensorFusion.ts`) turns 2D detections from multiple cameras into
3D tracks. A bounding-box center from a single camera is **not** a 3D point — it is a
*bearing*, a ray from the camera center through the back-projected pixel. A 3D
position requires triangulating rays from two or more viewpoints.

```mermaid
flowchart LR
    D1["cam A bbox"] --> R1["ray A"]
    D2["cam B bbox"] --> R2["ray B"]
    R1 --> T["least-squares<br/>ray intersection"]
    R2 --> T
    T --> P["3D position<br/>+ residual error"]
```

The solver minimizes the summed squared perpendicular distance from a point to each
ray using the projector `(I − ddᵀ)`, giving the normal equations

```text
( Σᵢ (I − dᵢdᵢᵀ) ) · X = Σᵢ (I − dᵢdᵢᵀ) · oᵢ
```

where `dᵢ` is each ray's unit direction and `oᵢ` its origin (the camera center). The
implementation:

- derives each ray from the bbox center using camera FOV + aspect (a calibrated
  intrinsic/extrinsic path is supported by the types but not yet populated);
- refuses two rays from the same camera (no parallax → biases the solve);
- falls back to an assumed fixed range along each ray when the rays are
  near-parallel and the linear system is ill-conditioned.

> **Conditioning caveat.** When the baseline is small or the target is distant, the
> depth direction becomes nearly unobservable and triangulation error grows roughly
> as `range² / baseline`. The fixed-range fallback is a placeholder, not a
> measurement; treat triangulated depth from near-parallel rays as weakly determined.

> **Correlation caveat.** Cross-camera correlation currently matches on class,
> confidence, and timestamp only — it has no geometric (epipolar) gate, so two
> different same-class drones seen by two cameras can be merged into one phantom
> triangulation. Epipolar / reprojection gating is on the roadmap.

---

## Configuration and tuning

The native engine is configured through `FusionConfig`
(`fusion_init` / `fusion_set_config`):

| Parameter | Default | Meaning | Tuning guidance |
|-----------|---------|---------|-----------------|
| `algorithm` | `ExtendedKalman` | Filter family | EKF for radar; KF for clean Cartesian; UKF for strong nonlinearity; IMM for maneuvering; PF only when genuinely multi-modal |
| `process_noise` (Q) | `1.0` | Un-modeled dynamics / maneuver intensity | ↑ to track agile targets (snappier, noisier); ↓ to smooth steady targets (laggier, risk of divergence on a maneuver) |
| `measurement_noise` (R) | `2.0` | Default sensor uncertainty | Overridden per-measurement by each modality's `covariance` |
| `association_threshold` | `11.345` | χ²(3) gate on the **squared** Mahalanobis distance (≈99%) | ↑ admits more candidates (more clutter, fewer missed associations); ↓ tightens the gate |
| `max_missed_detections` | `5` | Consecutive misses before a track is deleted | ↑ to ride through longer occlusions; ↓ to drop stale tracks faster |
| `min_confirmation_hits` | `3` | Hits before Tentative → Confirmed | ↑ to suppress false tracks from clutter; ↓ for faster confirmation |
| `particle_count` | `100` | Particles per track (PF only) | ↑ accuracy at `O(N)` cost; the default is a real-time compromise |

Per-modality measurement covariances are set by the producers in `useROSSensors.ts`
and reflect realistic sensor characteristics: lidar tightest (`[0.1, 0.1, 0.1]` m²),
radar good in range / coarse in angle (`[0.5 m², (1°)², (1.5°)²]`), thermal moderate
(`[2, 2, 2]` m²), acoustic loosest (`[10, 10, 10]` m²).

The **fusion rate** (`fusionRateHz`, default 10 Hz, clamped to 1–60 Hz) is set on the
ROS hook; measurements are buffered between cycles, with a backpressure guard that
drops the oldest measurements if a topic floods.

---

## Validation and metrics

The engine ships with Rust unit and multi-frame scenario tests
(`cargo test sensor_fusion`) covering the predict/update math, the track lifecycle,
polar radar integration, the lidar Cartesian contract, algorithm switching, and
Joseph-form covariance stability. The browser engine and ROS bridge have Vitest
coverage (`bun run test:run`).

For deeper tracker validation — recommended before any accuracy claim against real
hardware — the standard tools are:

- **NEES / NIS** (normalized estimation / innovation error squared): test whether the
  reported covariance is *honest*. A consistent filter yields NEES averaging the state
  dimension and NIS averaging the measurement dimension, both within χ² bounds. A
  chronically small NIS is the signature of an overconfident filter (e.g. from
  confidence-weighted fusion dropping cross-correlations).
- **OSPA / GOSPA**: a principled set-distance between estimated and true target sets,
  decomposable into localization error plus missed/false-target (cardinality) error.
  GOSPA is preferred because the components separate cleanly.
- **CLEAR-MOT (MOTA / MOTP)** and **ID-switch / track-purity** counts: identity-aware
  accuracy. Report ID switches alongside MOTA — MOTA is detection-dominated and hides
  the label swaps that would misdirect an interceptor.

These should be run as Monte-Carlo sweeps over scripted ground-truth scenarios that
vary clutter, detection probability, occlusion duration, maneuvers, and crossing
targets — not single happy-path runs.

---

## Known limitations and roadmap

These are deliberate simplifications, surfaced here rather than hidden. None of them
crash the engine; they bound its accuracy and consistency. Ordered roughly by impact.

| # | Limitation | Impact | Direction |
|---|-----------|--------|-----------|
| 1 | **Greedy nearest-neighbour association** with no one-to-one constraint | Mis-assignment and duplicate/coalesced tracks in dense or crossing scenes | Global assignment (Hungarian / auction) over the gated cost matrix |
| 2 | **Confidence-weighted fusion**, not information-form | Biased fused position; uncertainty not reflective of true precision | Sequential per-sensor updates with each modality's own `R`; covariance-intersection for correlated track fusion |
| 3 | **Age-based confirmation**, not sliding-window M-of-N | Intermittently-dropping tracks can over-promote | True M-of-N (and optionally a score-based SPRT) confirmation |
| 4 | **KF/UKF updates ignore per-measurement `R`** (only the EKF polar path uses it) | Sensor-specific accuracy underused on the non-polar path | Thread the per-measurement covariance into every filter's update |
| 5 | **Per-measurement timestamps unused**; one global `dt` per frame | Asynchronous sensors mis-timed; no out-of-sequence handling | Predict each track to each measurement's own time; OOSM buffering/retrodiction |
| 6 | **No CA/CT motion models in the IMM** (two CV models with different Q) | Cannot represent turns / sustained acceleration | Add coordinated-turn and constant-acceleration models |
| 7 | **No geometric (epipolar) cross-camera gate** in the browser engine | Phantom triangulations from different same-class targets | Epipolar + reprojection-error gating; populate camera intrinsics/extrinsics |
| 8 | **Diagonal-only covariances** on the measurement boundary | Drops off-diagonal correlation from coordinate conversion / Doppler | Full 3×3 measurement covariances |

### Resuming this work

The limitations above are specified in implementation-ready detail — with code
sketches, exact parameters, and test lists — in
[`SENSOR_FUSION_AGENT_SPECS.md`](SENSOR_FUSION_AGENT_SPECS.md). Recommended order,
because the Rust association / update / lifecycle items touch overlapping code in
`sensor_fusion.rs`:

1. **Shared gate helper.** Factor a `gated_sq_mahalanobis(track, meas) -> Option<f64>`
   (the squared-Mahalanobis gate, already χ²-calibrated and Cartesian-frame correct)
   so the assignment rewrite builds on one definition.
2. **Information-form fusion** (rows 2, 4). Apply each associated measurement
   sequentially with its own `R` instead of confidence-averaging; derive
   `track.confidence` afterward rather than using it as a fusion weight.
3. **Sliding-window M-of-N confirmation + covariance-volume deletion** (row 3).
4. **Global assignment** (row 1). Cluster co-located same-class measurements, then
   solve a 1:1 track↔cluster assignment with a dependency-free Kuhn–Munkres. A pure
   1:1 *measurement*↔track Hungarian would break multi-sensor fusion — N co-located
   returns from N sensors must all reach the one track.
5. **CV + Coordinated-Turn IMM** (row 6). The current IMM is two CV models; add a CT
   model so it can represent turns. Self-contained to `IMMFilter`.
6. **Per-measurement timestamps** (row 5). Predict each track to each measurement's
   own time within a batch. Full out-of-sequence-measurement (OOSM) retrodiction is a
   larger feature — **defer and document** (no recovered spec exists for it).
7. **Geometric cross-camera gate** (row 7). Isolated to the browser
   `SensorFusion.ts` — lowest-risk, a good first item.

Deferred (document, don't force): full 3×3 measurement covariances across the
TS↔Rust boundary (row 8) and full OOSM (the cross-batch half of item 6).

> **Gotchas for whoever picks this up.** Radar measurement `R` is polar
> (`[m², rad², rad²]`); the association gate already converts it to Cartesian, but
> `create_track` still seeds the *initial position* covariance from it verbatim, so
> for radar the birth position covariance is polar units treated as Cartesian — fold
> the Jacobian conversion in when addressing full covariances. When a gate seems too
> tight, fix the *scenario realism* (e.g. the single-point birth velocity prior), not
> the threshold. The browser `triangulatePosition` minimizes algebraic, not
> perpendicular, distance and lacks a behind-camera (cheirality) check.

---

## References

The design and the improvements above are grounded in standard multi-target tracking
and estimation literature.

**Estimation / Kalman family**
- Kalman filter — predict/update, Joseph-form covariance, numerical stability: <https://en.wikipedia.org/wiki/Kalman_filter>
- Extended Kalman filter — Jacobian linearization and divergence modes: <https://en.wikipedia.org/wiki/Extended_Kalman_filter>
- Unscented transform — sigma points, derivative-free accuracy: <https://en.wikipedia.org/wiki/Unscented_transform>
- R. Labbe, *Kalman and Bayesian Filters in Python* — CV model, discretized process noise: <https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python>

**Particle filter / IMM**
- Particle filter — SIR/bootstrap weights, degeneracy, resampling: <https://en.wikipedia.org/wiki/Particle_filter>
- Genovese, *The Interacting Multiple Model Algorithm* — JHU/APL Technical Digest: <https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V22-N04/22-04-Genovese.pdf>

**Data association**
- Mahalanobis distance — χ² gating and confidence ellipsoids: <https://en.wikipedia.org/wiki/Mahalanobis_distance>
- Radar tracker — gating, NN vs GNN vs JPDA/MHT, M-of-N, coasting: <https://en.wikipedia.org/wiki/Radar_tracker>
- Hungarian algorithm — optimal one-to-one assignment: <https://en.wikipedia.org/wiki/Hungarian_algorithm>
- Joint Probabilistic Data Association Filter: <https://en.wikipedia.org/wiki/Joint_Probabilistic_Data_Association_Filter>

**Multi-sensor fusion**
- Covariance intersection — conservative fusion of correlated estimates: <https://en.wikipedia.org/wiki/Covariance_intersection>
- Inverse-variance (information-form) weighting: <https://en.wikipedia.org/wiki/Inverse-variance_weighting>
- Out-of-sequence measurement handling (MathWorks): <https://www.mathworks.com/help/fusion/ug/introduciton-to-out-of-sequence-measurement-handling.html>

**Lifecycle / metrics**
- Track algorithm — tentative/confirmed, M-of-N: <https://en.wikipedia.org/wiki/Track_algorithm>
- Rahmathullah, García-Fernández, Svensson — *Generalized Optimal Sub-Pattern Assignment (GOSPA)*: <https://arxiv.org/abs/1601.05585>
- Bernardin & Stiefelhagen — *CLEAR MOT metrics*: <https://link.springer.com/content/pdf/10.1155/2008/246309.pdf>

**Multi-view geometry**
- Triangulation (computer vision) — midpoint / DLT, degeneracy: <https://en.wikipedia.org/wiki/Triangulation_(computer_vision)>
- Epipolar geometry — fundamental matrix, correspondence test: <https://en.wikipedia.org/wiki/Epipolar_geometry>
- Hartley & Zisserman, *Multiple View Geometry*, Ch. 8: <https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook1/HZepipolar.pdf>
