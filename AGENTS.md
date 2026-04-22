# AGENTS.md - CREBAIN Development Guide

## Build Commands

```bash
# Frontend development
bun run dev              # Start Vite dev server
bun run build            # Typecheck + build for production
bun run typecheck        # TypeScript type checking only

# Tauri (full app)
bun run tauri:dev        # Development mode with hot reload
bun run tauri:build      # Production build

# Testing
bun run test             # Run tests in watch mode
bun run test:run         # Run tests once
bun run test:coverage    # Run tests with coverage
bun run test:benchmark   # Run detector benchmarks

# Rust backend
cd src-tauri && cargo check    # Type check Rust code
cd src-tauri && cargo build    # Build Rust backend
cd src-tauri && cargo clippy   # Lint Rust code
```

## Code Style

### TypeScript/React
- Use functional components with hooks
- Prefer `useMemo` and `useCallback` for expensive computations
- Use `useRef` for mutable values that don't trigger re-renders
- Avoid `console.log` in production code
- Use named constants for magic numbers
- Always clean up effects (intervals, subscriptions, event listeners)

### Rust
- Run `cargo clippy` before committing
- Use `log::info/warn/error` instead of `println!`
- Validate all external inputs (paths, user data)
- Use `spawn_blocking` for CPU-intensive operations in async contexts

## Architecture Notes

### Frontend (`src/`)
- `components/` - React UI components
- `hooks/` - Custom React hooks
- `ros/` - ROS bridge and Gazebo integration
- `detection/` - ML detection types and sensor fusion
- `physics/` - Drone physics simulation
- `simulation/` - Interception system

### Backend (`src-tauri/`)
- `inference/` - ML abstraction layer (CoreML, ONNX)
- `transport/` - Zenoh low-latency transport
- `sensor_fusion.rs` - Kalman/Particle/IMM filters

## Performance Guidelines

- Use `CircularBuffer` for high-frequency position data
- Prefer squared distance comparisons (avoid `sqrt()`)
- Use `ImageBitmap` for GPU-accelerated image decoding
- Memoize derived state to prevent unnecessary recomputes
- Keep camera feed updates at ~12 FPS (83ms interval)

## Testing

Test files use Vitest. Place tests in `__tests__/` directories or use `.test.ts` suffix.

```typescript
import { describe, it, expect } from 'vitest'
```
