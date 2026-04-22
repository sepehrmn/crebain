/**
 * CREBAIN CoreML Detection Hook
 * Adaptive Response & Awareness System (ARAS)
 *
 * Legacy hook for native detection via Tauri commands.
 *
 * Despite the name, this now uses the backend-side `detect_native_raw` command
 * so it works cross-platform (CoreML on macOS, ONNX/TensorRT/CUDA on Linux).
 */

import { useCallback, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import type { Detection, CoreMLDetection, CoreMLDetectionResult } from '../detection/types'
import {
  mapToDetectionClass,
  getThreatLevel,
  DEFAULT_CONFIDENCE_THRESHOLD,
  DEFAULT_IOU_THRESHOLD,
  DEFAULT_MAX_DETECTIONS,
} from '../detection/types'
import { detectionLogger as log } from '../lib/logger'

type NativeDetectionResult = CoreMLDetectionResult & { backend?: string }

interface SystemInfo {
  platform: string
  arch: string
  coremlAvailable: boolean
  backend: string
}

interface CoreMLDetectionState {
  isReady: boolean
  isLoading: boolean
  error: string | null
  detections: Detection[]
  inferenceTime: number
  preprocessTime: number
  postprocessTime: number
  backend: string
}

interface UseCoreMLDetectionReturn extends CoreMLDetectionState {
  detect: (imageData: ImageData) => Promise<Detection[]>
  detectFromCanvas: (canvas: HTMLCanvasElement | OffscreenCanvas) => Promise<Detection[]>
  getSystemInfo: () => Promise<SystemInfo>
}

// Convert CoreML detection (bbox is in pixel coordinates) to our Detection format
function convertDetection(coremlDet: CoreMLDetection, frameWidth: number, frameHeight: number): Detection {
  const detClass = mapToDetectionClass(coremlDet.classLabel)
  const threatLevel = getThreatLevel(detClass, coremlDet.confidence)
  
  return {
    id: coremlDet.id,
    class: detClass,
    confidence: coremlDet.confidence,
    bbox: [
      coremlDet.bbox.x1,
      coremlDet.bbox.y1,
      coremlDet.bbox.x2,
      coremlDet.bbox.y2,
    ],
    timestamp: coremlDet.timestamp,
    threatLevel,
    frameWidth,
    frameHeight,
  }
}

function imageDataToRGBA(imageData: ImageData): Uint8Array {
  return new Uint8Array(
    imageData.data.buffer,
    imageData.data.byteOffset,
    imageData.data.byteLength
  )
}

function canvasToImageData(canvas: HTMLCanvasElement | OffscreenCanvas): ImageData {
  if (canvas instanceof HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Failed to get 2D context')
    return ctx.getImageData(0, 0, canvas.width, canvas.height)
  }

  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | null
  if (!ctx) throw new Error('Failed to get 2D context')
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

/**
 * Hook for using CoreML detection via Tauri
 */
export function useCoreMLDetection(): UseCoreMLDetectionReturn {
  const [state, setState] = useState<CoreMLDetectionState>({
    isReady: true, // Backend selects/initializes the best available detector
    isLoading: false,
    error: null,
    detections: [],
    inferenceTime: 0,
    preprocessTime: 0,
    postprocessTime: 0,
    backend: 'Native Detector',
  })

  /**
   * Get system information
   */
  const getSystemInfo = useCallback(async (): Promise<SystemInfo> => {
    try {
      const info = await invoke<SystemInfo>('get_system_info')
      return info
    } catch (error) {
      log.error('Failed to get system info', { error })
      return {
        platform: 'unknown',
        arch: 'unknown',
        coremlAvailable: false,
        backend: 'Unknown',
      }
    }
  }, [])

  /**
   * Run detection on ImageData
   */
  const detect = useCallback(async (imageData: ImageData): Promise<Detection[]> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }))

    try {
      const rgbaData = imageDataToRGBA(imageData)

      const result = await invoke<NativeDetectionResult>('detect_native_raw', {
        rgbaData: Array.from(rgbaData),
        width: imageData.width,
        height: imageData.height,
        confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
        iouThreshold: DEFAULT_IOU_THRESHOLD, // retained for API parity (backend may ignore)
        maxDetections: DEFAULT_MAX_DETECTIONS
      })

      if (!result.success) {
        throw new Error(result.error || 'Detection failed')
      }

      const detections = result.detections.map(det => convertDetection(det, imageData.width, imageData.height))

      setState(prev => ({
        ...prev,
        isLoading: false,
        detections,
        inferenceTime: result.inferenceTimeMs,
        preprocessTime: result.preprocessTimeMs ?? 0,
        postprocessTime: result.postprocessTimeMs ?? 0,
        backend: result.backend ?? prev.backend,
      }))

      return detections
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: message,
        detections: [],
      }))
      throw error
    }
  }, [])

  /**
   * Run detection on a canvas element
   */
  const detectFromCanvas = useCallback(async (
    canvas: HTMLCanvasElement | OffscreenCanvas
  ): Promise<Detection[]> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }))

    try {
      const imageData = canvasToImageData(canvas)
      const rgbaData = imageDataToRGBA(imageData)

      const result = await invoke<NativeDetectionResult>('detect_native_raw', {
        rgbaData: Array.from(rgbaData),
        width: imageData.width,
        height: imageData.height,
        confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
        iouThreshold: DEFAULT_IOU_THRESHOLD, // retained for API parity (backend may ignore)
        maxDetections: DEFAULT_MAX_DETECTIONS
      })

      if (!result.success) {
        throw new Error(result.error || 'Detection failed')
      }

      const detections = result.detections.map(det => convertDetection(det, imageData.width, imageData.height))

      setState(prev => ({
        ...prev,
        isLoading: false,
        detections,
        inferenceTime: result.inferenceTimeMs,
        preprocessTime: result.preprocessTimeMs ?? 0,
        postprocessTime: result.postprocessTimeMs ?? 0,
        backend: result.backend ?? prev.backend,
      }))

      return detections
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: message,
        detections: [],
      }))
      throw error
    }
  }, [])

  return {
    ...state,
    detect,
    detectFromCanvas,
    getSystemInfo,
  }
}

export default useCoreMLDetection
