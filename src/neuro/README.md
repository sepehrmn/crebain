# `src/neuro` — Engram neuro-cybernetic (NCP) client

Lets CREBAIN ask **Engram** (Paper2Brain) for a neural simulation and read back
membrane potential / spikes / population rate — for **perception, action, both,
or neither** (the rest stays classic ML in CREBAIN). Self-contained and
**non-invasive**: this directory adds no dependencies and touches no existing
CREBAIN code. Protocol spec lives in the published NCP repo
(`NEURO_CYBERNETIC_PROTOCOL.md` in https://github.com/sepehrmn/NCP); the
matching JSON Schemas live in the Paper2Brain repo (`backend/neurocontrol/`).

## Use

```ts
import { NeuroSimClient, WebSocketNeuroSim } from './neuro'

const transport = new WebSocketNeuroSim('ws://127.0.0.1:28471/api/neurocontrol/ws')
const engram = new NeuroSimClient(transport.send)

// e.g. a per-UAV "feature neuron": drive it from a detection score, read its spikes
await engram.open(
  'uav3-percept',
  { kind: 'builtin', ref: 'iaf_psc_alpha', population_sizes: { feat: 1 } },
  [{ port: 'spk', target: 'feat', observable: 'spikes' }],
  [{ port: 'drive', target: 'feat', kind: 'current_pA' }],
)
const obs = await engram.step('uav3-percept', { drive: { data: [500.0], unit: 'pA' } }, 50.0)
const spikeCount = obs.records.spk.times.length // feed into CREBAIN's logic
await engram.close('uav3-percept')
```

## Wiring (your choice; both are non-invasive)

- **WebSocket** (`ws.ts`) — point at Engram's `/api/neurocontrol/ws`. Simplest;
  works from the Tauri webview.
- **Zenoh** — for a fully **decoupled** bus, implement `Send` over CREBAIN's
  existing `ZenohBridge` (query the `engram/ncp/rpc` key; subscribe to
  `engram/ncp/session/{id}/observation`). Engram recommends Zenoh as the default
  decoupled transport precisely to avoid binding CREBAIN to a server address.
- **Native Rust + Zenoh** (recommended for performance) — a Rust NCP client now
  lives at `src-tauri/src/ncp/` (behind the `ncp` Cargo feature), built on the
  canonical NCP SDK. It speaks the queryable RPC + the perception/action pub/sub
  planes with proper QoS, and maps pose/velocity ↔ NCP frames in Rust. This TS
  client remains the zero-dependency path (browser/Tauri-webview); the Rust client
  is the high-performance path. See `src-tauri/src/ncp/README.md`.

## Action (Engram as the brain)

For closed-loop control, Engram emits NCP `command_frame`s and **CREBAIN maps them
to its actuators** (e.g. publish a decoded `velocity_setpoint` to
`/mavros/<ns>/setpoint_velocity/cmd_vel` via the existing ROSBridge). Engram holds
no CREBAIN-specific topic knowledge — the mapping lives here, in CREBAIN.

## Boundary

Returned `V_m`/spikes are **raw simulation outputs of a specified model**
(`calibrated_posterior=false`, `is_simulation_output=true`), never a validated
reproduction; a neuro-controller is a control artifact, not a scientific claim.
