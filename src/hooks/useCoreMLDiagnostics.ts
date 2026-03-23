/**
 * CREBAIN CoreML Diagnostics Hook
 * Provides CoreML test and benchmark functionality
 */

import { useState, useCallback, useRef } from 'react'
import { invoke } from '@tauri-apps/api/core'
import type { CoreMLDetectionResult } from '../detection/types'

type NativeDetectionResult = CoreMLDetectionResult & { backend?: string }

export interface CoreMLDiagnosticsConfig {
  onMessage?: (level: 'info' | 'success' | 'error', text: string) => void
  onDetectionComplete?: (result: {
    inferenceTimeMs: number
    preprocessTimeMs?: number
    postprocessTimeMs?: number
    detectionCount: number
  }) => void
}

export interface CoreMLDiagnosticsReturn {
  isTesting: boolean
  isBenchmarking: boolean
  benchmarkProgress: number
  runTest: () => Promise<void>
  runBenchmark: (iterations?: number) => Promise<void>
}

const TEST_IMAGE_SIZE = 640
const DEFAULT_BENCHMARK_ITERATIONS = 100

function generateTestImage(): { imageData: Uint8Array; width: number; height: number } {
  const width = TEST_IMAGE_SIZE
  const height = TEST_IMAGE_SIZE
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!

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
    imageData: new Uint8Array(imageDataObj.data.buffer),
    width,
    height,
  }
}

export function useCoreMLDiagnostics(
  config: CoreMLDiagnosticsConfig = {}
): CoreMLDiagnosticsReturn {
  const { onMessage, onDetectionComplete } = config

  const [isTesting, setIsTesting] = useState(false)
  const [isBenchmarking, setIsBenchmarking] = useState(false)
  const [benchmarkProgress, setBenchmarkProgress] = useState(0)
  const abortRef = useRef(false)

  const runTest = useCallback(async () => {
    if (isTesting || isBenchmarking) return

    setIsTesting(true)
    onMessage?.('info', 'DETECTOR TEST: Generating test image...')

    try {
      const { imageData, width, height } = generateTestImage()

      onMessage?.('info', 'DETECTOR TEST: Running inference...')

      const result = await invoke<NativeDetectionResult>('detect_native_raw', {
        rgbaData: Array.from(imageData),
        width,
        height,
        confidenceThreshold: 0.25,
        maxDetections: 100,
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
      setIsTesting(false)
    }
  }, [isTesting, isBenchmarking, onMessage, onDetectionComplete])

  const runBenchmark = useCallback(async (iterations = DEFAULT_BENCHMARK_ITERATIONS) => {
    if (isTesting || isBenchmarking) return

    setIsBenchmarking(true)
    setBenchmarkProgress(0)
    abortRef.current = false

    onMessage?.('info', `BENCHMARK: Starting ${iterations} iterations...`)

    try {
      const { imageData, width, height } = generateTestImage()
      const rgbaArray = Array.from(imageData)

      // Warm-up runs
      onMessage?.('info', 'BENCHMARK: Warm-up phase (5 runs)...')
      for (let i = 0; i < 5; i++) {
        if (abortRef.current) break
        await invoke<NativeDetectionResult>('detect_native_raw', {
          rgbaData: rgbaArray,
          width,
          height,
          confidenceThreshold: 0.25,
          maxDetections: 100,
        })
      }

      if (abortRef.current) {
        onMessage?.('info', 'BENCHMARK: Aborted')
        return
      }

      // Benchmark runs
      const times: number[] = []
      for (let i = 0; i < iterations; i++) {
        if (abortRef.current) break

        const result = await invoke<NativeDetectionResult>('detect_native_raw', {
          rgbaData: rgbaArray,
          width,
          height,
          confidenceThreshold: 0.25,
          maxDetections: 100,
        })

        if (result.success) {
          times.push(result.inferenceTimeMs)
        }

        setBenchmarkProgress(Math.round(((i + 1) / iterations) * 100))
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
      const mean = times.reduce((a, b) => a + b, 0) / times.length
      const sorted = [...times].sort((a, b) => a - b)
      const p50 = sorted[Math.floor(times.length * 0.5)]
      const p95 = sorted[Math.floor(times.length * 0.95)]
      const p99 = sorted[Math.floor(times.length * 0.99)]
      const min = sorted[0]
      const max = sorted[sorted.length - 1]
      const fps = 1000 / mean

      onMessage?.(
        'success',
        `BENCHMARK COMPLETE: mean=${mean.toFixed(1)}ms, p50=${p50.toFixed(1)}ms, p95=${p95.toFixed(1)}ms, p99=${p99.toFixed(1)}ms, min=${min.toFixed(1)}ms, max=${max.toFixed(1)}ms, ~${fps.toFixed(0)} FPS`
      )

      onDetectionComplete?.({
        inferenceTimeMs: mean,
        detectionCount: iterations,
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      onMessage?.('error', `BENCHMARK ERROR: ${message}`)
    } finally {
      setIsBenchmarking(false)
      setBenchmarkProgress(0)
    }
  }, [isTesting, isBenchmarking, onMessage, onDetectionComplete])

  return {
    isTesting,
    isBenchmarking,
    benchmarkProgress,
    runTest,
    runBenchmark,
  }
}

export default useCoreMLDiagnostics
