export interface SystemInfo {
  platform: string
  arch: string
  coremlAvailable: boolean
  onnxAvailable: boolean
  backend: string
  mode: string
  onnxDetector?: unknown
  sensorFusion?: unknown
}

export type BackendHealth = 'ready' | 'unavailable' | 'unknown'

export interface LatencyStats {
  mean: number
  p50: number
  p95: number
  p99: number
  min: number
  max: number
  fps: number
}

const UNKNOWN_SYSTEM_INFO: SystemInfo = {
  platform: 'unknown',
  arch: 'unknown',
  coremlAvailable: false,
  onnxAvailable: false,
  backend: 'Unknown',
  mode: 'unknown',
}

function readString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.length > 0 ? value : fallback
}

function readBoolean(value: unknown): boolean {
  return typeof value === 'boolean' ? value : false
}

export function normalizeSystemInfo(value: unknown): SystemInfo {
  if (!value || typeof value !== 'object') return UNKNOWN_SYSTEM_INFO

  const record = value as Record<string, unknown>

  return {
    platform: readString(record.platform, UNKNOWN_SYSTEM_INFO.platform),
    arch: readString(record.arch, UNKNOWN_SYSTEM_INFO.arch),
    coremlAvailable: readBoolean(record.coremlAvailable),
    onnxAvailable: readBoolean(record.onnxAvailable),
    backend: readString(record.backend, UNKNOWN_SYSTEM_INFO.backend),
    mode: readString(record.mode, UNKNOWN_SYSTEM_INFO.mode),
    onnxDetector: record.onnxDetector,
    sensorFusion: record.sensorFusion,
  }
}

export function getBackendHealth(info: Pick<SystemInfo, 'backend' | 'coremlAvailable' | 'onnxAvailable'>): BackendHealth {
  const backend = info.backend.toLowerCase()

  if (backend.includes('no backend') || backend === 'unknown') return 'unavailable'
  if (info.coremlAvailable || info.onnxAvailable) return 'ready'
  if (backend.includes('coreml') || backend.includes('onnx') || backend.includes('cuda') || backend.includes('tensorrt')) {
    return 'ready'
  }

  return 'unknown'
}

export function summarizeSystemInfo(info: SystemInfo) {
  return {
    platform: info.platform,
    arch: info.arch,
    backend: info.backend,
    mode: info.mode,
    backendHealth: getBackendHealth(info),
    fusionReady: info.sensorFusion != null,
  }
}

export function calculateLatencyStats(times: number[]): LatencyStats {
  if (times.length === 0) {
    throw new Error('Cannot calculate latency stats for an empty sample')
  }

  const mean = times.reduce((a, b) => a + b, 0) / times.length
  const sorted = [...times].sort((a, b) => a - b)
  const percentile = (value: number) => sorted[Math.min(sorted.length - 1, Math.floor(times.length * value))]

  return {
    mean,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    min: sorted[0],
    max: sorted[sorted.length - 1],
    fps: 1000 / mean,
  }
}
