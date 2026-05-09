/**
 * CREBAIN CoreML Diagnostics Hook
 * Provides CoreML test and benchmark functionality
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { invoke } from '@tauri-apps/api/core'
import type { CoreMLDetectionResult } from '../detection/types'
import { calculateLatencyStats } from '../lib/diagnostics'
import { TAURI_COMMANDS } from '../lib/tauriCommands'

type NativeDetectionResult = CoreMLDetectionResult & { backend?: string }

export interface CoreMLDiagnosticsConfig {
  onMessage?: (level: 'info' | 'success' | 'error', text: string) => void
  onDetectionComplete?: (result: {
    inferenceTimeMs: number
    preprocessTimeMs?: number
    postprocessTimeMs?: number
    detectionCount: number
  }) => void
  maxBenchmarkIterations?: number
}

export interface CoreMLDiagnosticsReturn {
  isTesting: boolean
  isBenchmarking: boolean
  benchmarkProgress: number
  runTest: () => Promise<void>
  runBenchmark: (iterations?: number) => Promise<void>
  cancelBenchmark: () => void
}

const TEST_IMAGE_SIZE = 640
const DEFAULT_BENCHMARK_ITERATIONS = 100
const MIN_BENCHMARK_ITERATIONS = 1
const MAX_BENCHMARK_ITERATIONS = 1000
const BENCHMARK_WARMUP_ITERATIONS = 5
const DEFAULT_CONFIDENCE_THRESHOLD = 0.25
const DEFAULT_MAX_DETECTIONS = 100

function normalizeBenchmarkIterations(iterations: number | undefined, maxIterations: number | undefined): number {
  const normalizedMax = Number.isFinite(maxIterations)
    ? Math.max(MIN_BENCHMARK_ITERATIONS, Math.floor(maxIterations!))
    : MAX_BENCHMARK_ITERATIONS
  const requested = Number.isFinite(iterations)
    ? Math.floor(iterations!)
    : DEFAULT_BENCHMARK_ITERATIONS

  return Math.min(
    normalizedMax,
    Math.max(MIN_BENCHMARK_ITERATIONS, requested)
  )
}

function generateTestImage(): { imageData: Uint8Array; width: number; height: number } {
  const width = TEST_IMAGE_SIZE
  const height = TEST_IMAGE_SIZE
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new Error('Failed to get 2D context for diagnostics image')
  }

  // Create a gradient background
  const gradient = ctx.createLinearGradient(0, 0, width, height)
  gradient.addColorStop(0, '#1a1a2e')
  gradient.addColorStop(0.5, '#16213e')
  gradient.addColorStop(1, '#0f3460')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, width, height)

  // Draw grid pattern
  ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)'
  ctx.lineWidth = 1
  for (let x = 0; x < width; x += 40) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
    ctx.stroke()
  }
  for (let y = 0; y < height; y += 40) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }

  // Draw some shapes to simulate objects
  const shapes = [
    { x: 100, y: 100, size: 80, color: '#ff6b6b' },
    { x: 300, y: 200, size: 60, color: '#4ecdc4' },
    { x: 500, y: 400, size: 100, color: '#ffe66d' },
    { x: 200, y: 450, size: 50, color: '#95e1d3' },
  ]

  for (const shape of shapes) {
    ctx.fillStyle = shape.color
    ctx.beginPath()
    ctx.arc(shape.x, shape.y, shape.size / 2, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
    ctx.lineWidth = 2
    ctx.stroke()
  }

  // Draw text
  ctx.fillStyle = '#00ff88'
  ctx.font = '24px monospace'
  ctx.fillText('CREBAIN DIAGNOSTICS', 180, 320)
  ctx.font = '16px monospace'
  ctx.fillStyle = '#888'
  ctx.fillText(`${width}x${height} Test Pattern`, 230, 350)

  const imageDataObj = ctx.getImageData(0, 0, width, height)
  return {
    imageData: new Uint8Array(
      imageDataObj.data.buffer,
      imageDataObj.data.byteOffset,
      imageDataObj.data.byteLength
    ),
    width,
    height,
  }
}

export function useCoreMLDiagnostics(
  config: CoreMLDiagnosticsConfig = {}
): CoreMLDiagnosticsReturn {
  const { onMessage, onDetectionComplete, maxBenchmarkIterations } = config

  const [isTesting, setIsTesting] = useState(false)
  const [isBenchmarking, setIsBenchmarking] = useState(false)
  const [benchmarkProgress, setBenchmarkProgress] = useState(0)
  const mountedRef = useRef(true)
  const abortRef = useRef(false)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      abortRef.current = true
    }
  }, [])

  const runTest = useCallback(async () => {
    if (isTesting || isBenchmarking) return

    setIsTesting(true)
    onMessage?.('info', 'DETECTOR TEST: Generating test image...')

    try {
      const { imageData, width, height } = generateTestImage()

      onMessage?.('info', 'DETECTOR TEST: Running inference...')

      const result = await invoke<NativeDetectionResult>(TAURI_COMMANDS.detection.nativeRaw, {
        rgbaData: Array.from(imageData),
        width,
        height,
        confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
        maxDetections: DEFAULT_MAX_DETECTIONS,
      })

      if (result.success) {
        const backendText = result.backend ? ` (${result.backend})` : ''
        onMessage?.(
          'success',
          `DETECTOR TEST: ${result.detections.length} detections in ${result.inferenceTimeMs.toFixed(1)}ms${backendText}`
        )
        onDetectionComplete?.({
          inferenceTimeMs: result.inferenceTimeMs,
          preprocessTimeMs: result.preprocessTimeMs ?? undefined,
          postprocessTimeMs: result.postprocessTimeMs ?? undefined,
          detectionCount: result.detections.length,
        })
      } else {
        onMessage?.('error', `DETECTOR TEST FAILED: ${result.error}`)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      onMessage?.('error', `DETECTOR TEST ERROR: ${message}`)
    } finally {
      if (mountedRef.current) setIsTesting(false)
    }
  }, [isTesting, isBenchmarking, onMessage, onDetectionComplete])

  const runBenchmark = useCallback(async (iterations = DEFAULT_BENCHMARK_ITERATIONS) => {
    if (isTesting || isBenchmarking) return
    const runIterations = normalizeBenchmarkIterations(iterations, maxBenchmarkIterations)

    setIsBenchmarking(true)
    setBenchmarkProgress(0)
    abortRef.current = false

    onMessage?.('info', `BENCHMARK: Starting ${runIterations} iterations...`)

    try {
      const { imageData, width, height } = generateTestImage()
      const rgbaArray = Array.from(imageData)

      // Warm-up runs
      onMessage?.('info', `BENCHMARK: Warm-up phase (${BENCHMARK_WARMUP_ITERATIONS} runs)...`)
      for (let i = 0; i < BENCHMARK_WARMUP_ITERATIONS; i++) {
        if (abortRef.current) break
        await invoke<NativeDetectionResult>(TAURI_COMMANDS.detection.nativeRaw, {
          rgbaData: rgbaArray,
          width,
          height,
          confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
          maxDetections: DEFAULT_MAX_DETECTIONS,
        })
      }

      if (abortRef.current) {
        onMessage?.('info', 'BENCHMARK: Aborted')
        return
      }

      // Benchmark runs
      const times: number[] = []
      for (let i = 0; i < runIterations; i++) {
        if (abortRef.current) break

        const result = await invoke<NativeDetectionResult>(TAURI_COMMANDS.detection.nativeRaw, {
          rgbaData: rgbaArray,
          width,
          height,
          confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
          maxDetections: DEFAULT_MAX_DETECTIONS,
        })

        if (result.success && Number.isFinite(result.inferenceTimeMs) && result.inferenceTimeMs >= 0) {
          times.push(result.inferenceTimeMs)
        }

        if (mountedRef.current) {
          setBenchmarkProgress(Math.round(((i + 1) / runIterations) * 100))
        }
      }

      if (abortRef.current) {
        onMessage?.('info', 'BENCHMARK: Aborted')
        return
      }

      if (times.length === 0) {
        onMessage?.('error', 'BENCHMARK: No successful runs')
        return
      }

      // Calculate statistics
      const stats = calculateLatencyStats(times)

      onMessage?.(
        'success',
        `BENCHMARK COMPLETE: mean=${stats.mean.toFixed(1)}ms, p50=${stats.p50.toFixed(1)}ms, p95=${stats.p95.toFixed(1)}ms, p99=${stats.p99.toFixed(1)}ms, min=${stats.min.toFixed(1)}ms, max=${stats.max.toFixed(1)}ms, ~${stats.fps.toFixed(0)} FPS`
      )

      onDetectionComplete?.({
        inferenceTimeMs: stats.mean,
        detectionCount: times.length,
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      onMessage?.('error', `BENCHMARK ERROR: ${message}`)
    } finally {
      if (mountedRef.current) {
        setIsBenchmarking(false)
        setBenchmarkProgress(0)
      }
    }
  }, [isTesting, isBenchmarking, maxBenchmarkIterations, onMessage, onDetectionComplete])

  const cancelBenchmark = useCallback(() => {
    if (!isBenchmarking) return
    abortRef.current = true
    onMessage?.('info', 'BENCHMARK: Abort requested')
  }, [isBenchmarking, onMessage])

  return {
    isTesting,
    isBenchmarking,
    benchmarkProgress,
    runTest,
    runBenchmark,
    cancelBenchmark,
  }
}

export default useCoreMLDiagnostics
