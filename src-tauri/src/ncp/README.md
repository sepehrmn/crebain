# `src/ncp` — Engram neuro-control (NCP), Rust client

CREBAIN's **native Rust + Zenoh** client for the Neuro-Control Protocol (NCP) —
the high-performance peer to the TypeScript WebSocket client in
[`../../../src/neuro/`](../../../src/neuro). It lets CREBAIN ask **Engram**
(Paper2Brain) for a neural simulation and/or be steered as a controller, for
**perception, action, both, or neither**, over the recommended decoupled Zenoh
bus.

It uses the canonical NCP SDK (`ncp-core` + `ncp-zenoh`) from the sibling
**`Paper2Brain/ncp`** workspace, so the wire is identical across the Rust, Python
and TS peers. Spec: `Paper2Brain/NEURO_CONTROL_PROTOCOL.md`.

## Opt-in (feature-gated)

This module is behind the **`ncp` Cargo feature** (off by default) so the default
CREBAIN build and the command-contract test are unchanged:

```bash
cargo check  --features ncp --manifest-path src-tauri/Cargo.toml
cargo test   --features ncp --lib ncp --manifest-path src-tauri/Cargo.toml
```

It requires the sibling `Paper2Brain/ncp` workspace (the path dependency in
`src-tauri/Cargo.toml`). For a standalone build, switch that to a git/crates.io
dependency — a one-line change.

## What it provides

- **Project mapping (CREBAIN-specific, stays here):** `sensor_frame_from_pose`
  (pose + body velocity → NCP `SensorFrame`), `velocity_from_command` (NCP
  `CommandFrame` → `TwistStampedData` for `/mavros/<ns>/setpoint_velocity/cmd_vel`,
  failing safe to zero on `hold`/`estop`), and `observation_scalar` (a population
  observation → a scalar feature).
- **`NcpBridge`** — a Zenoh-backed client: `connect`, `open_feature_neuron` /
  `step_feature_neuron` / `close` (perception/sim service via control-plane RPC),
  `publish_sensor` (perception plane), `subscribe_commands` (action plane → MAVROS).
- **Tauri commands** (`ncp_connect`, `ncp_open_feature_neuron`,
  `ncp_step_feature_neuron`, `ncp_close`) — ready to register.

## Exposing it to the frontend (one deliberate step)

The commands compile but are **not** registered by default (so the
`generate_handler!` command-contract test stays green). To turn them on, in
`src-tauri/src/lib.rs::run()`:

```rust
// after `tauri::Builder::default()`:
#[cfg(feature = "ncp")] let builder = builder.manage(crate::ncp::NcpHandle::default());
// and add to the generate_handler![] list:
//   ncp_connect, ncp_open_feature_neuron, ncp_step_feature_neuron, ncp_close,
```

then add the matching entries to the frontend command registry. Until then, the
TS WebSocket client (`src/neuro`) remains the shipped path.

## Boundary

Returned `V_m`/spikes are raw simulation outputs (`calibrated_posterior=false`,
`is_simulation_output=true`), never a validated reproduction; a neuro-controller
is a control artifact, not a scientific claim. Engram holds no CREBAIN-specific
topic knowledge — the mapping lives here, in CREBAIN.
