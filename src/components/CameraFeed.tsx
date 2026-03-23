/**
 * CREBAIN Camera Feed Component
 * Adaptive Response & Awareness System (ARAS)
 *
 * Individual camera feed with detection overlay rendering
 */

import { useRef, useEffect, useCallback, useState, useMemo } from 'react'
import * as THREE from 'three'
import type { SurveillanceCamera, Detection, FusedTrack, DetectionClass } from '../detection/types'
import type { RendererWithAsync } from './viewer/types'
import { useDetection } from '../hooks/useDetection'
import { DEFAULT_OVERLAY_STYLE, THREAT_LEVEL_COLORS } from '../detection/types'

interface CameraFeedProps {
  camera: SurveillanceCamera
  renderer: RendererWithAsync | null
  scene: THREE.Scene | null
  width: number
  height: number
  isSelected?: boolean
  onClick?: () => void
  detectionEnabled?: boolean
  onDetections?: (detections: Detection[]) => void
  fusedTracks?: FusedTrack[]
}

export function CameraFeed({
  camera,
  renderer,
  scene,
  width,
  height,
  isSelected = false,
  onClick,
  detectionEnabled = true,
  onDetections,
  fusedTracks = [],
}: CameraFeedProps) {
  const feedCanvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const renderTargetRef = useRef<THREE.WebGLRenderTarget | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const animationFrameRef = useRef<number>(0)
  const lastDetectionTimeRef = useRef<number>(0)

  const [localDetections, setLocalDetections] = useState<Detection[]>([])
  const [inferenceTime, setInferenceTime] = useState<number>(0)

  // Pre-allocate buffers outside render loop for performance
  const pixelBufferRef = useRef<Uint8Array | null>(null)
  const imageDataRef = useRef<ImageData | null>(null)
  // Reusable OffscreenCanvas for vertical flip
  const flipCanvasRef = useRef<OffscreenCanvas | null>(null)
  const flipCtxRef = useRef<OffscreenCanvasRenderingContext2D | null>(null)

  const detection = useDetection({
    autoInit: detectionEnabled && camera.isActive,
    config: {
      confidenceThreshold: 0.3,
      iouThreshold: 0.45,
      maxDetections: 20,
      useWebGPU: true,
    },
  })

  useEffect(() => {
    if (!camera.isActive) return

    cameraRef.current = new THREE.PerspectiveCamera(
      camera.fov,
      width / height,
      0.1,
      1000
    )
    cameraRef.current.position.copy(camera.position)
    cameraRef.current.lookAt(camera.target)

    cameraRef.current.rotation.y += THREE.MathUtils.degToRad(camera.pan)
    cameraRef.current.rotation.x += THREE.MathUtils.degToRad(camera.tilt)
    cameraRef.current.zoom = camera.zoom
    cameraRef.current.updateProjectionMatrix()

    return () => {
      cameraRef.current = null
    }
  }, [camera, width, height])

  useEffect(() => {
    if (!renderer || width <= 0 || height <= 0) return

    renderTargetRef.current = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    })

    return () => {
      renderTargetRef.current?.dispose()
      renderTargetRef.current = null
    }
  }, [renderer, width, height])

  useEffect(() => {
    if (!camera.isActive || !renderer || !scene || !cameraRef.current || !renderTargetRef.current) {
      return
    }

    const feedCanvas = feedCanvasRef.current
    const feedCtx = feedCanvas?.getContext('2d')
    if (!feedCanvas || !feedCtx) return

    if (!pixelBufferRef.current || pixelBufferRef.current.length !== width * height * 4) {
      pixelBufferRef.current = new Uint8Array(width * height * 4)
    }
    const pixelBuffer = pixelBufferRef.current

    if (!flipCanvasRef.current || flipCanvasRef.current.width !== width || flipCanvasRef.current.height !== height) {
      flipCanvasRef.current = new OffscreenCanvas(width, height)
      flipCtxRef.current = flipCanvasRef.current.getContext('2d')
    }
    const flipCanvas = flipCanvasRef.current
    const flipCtx = flipCtxRef.current
    if (!flipCtx) return

    const render = async () => {
      if (!cameraRef.current || !renderTargetRef.current || !renderer || !scene) return

      const currentRenderTarget = renderer.getRenderTarget()
      renderer.setRenderTarget(renderTargetRef.current)

      if (renderer.renderAsync) {
        await renderer.renderAsync(scene, cameraRef.current)
      } else {
        renderer.render(scene, cameraRef.current)
      }
      
      renderer.setRenderTarget(currentRenderTarget)

      renderer.readRenderTargetPixels(
        renderTargetRef.current,
        0, 0, width, height,
        pixelBuffer
      )
      if (!imageDataRef.current || imageDataRef.current.width !== width || imageDataRef.current.height !== height) {
        imageDataRef.current = new ImageData(
          new Uint8ClampedArray(pixelBuffer.buffer as ArrayBuffer, pixelBuffer.byteOffset, pixelBuffer.length),
          width,
          height
        )
      }

      const imageData = imageDataRef.current!
      flipCtx.putImageData(imageData, 0, 0)

      feedCtx.save()
      feedCtx.scale(1, -1)
      feedCtx.drawImage(flipCanvas, 0, -height)
      feedCtx.restore()

      const now = performance.now()
      if (detectionEnabled && detection.isReady && now - lastDetectionTimeRef.current > 200) {
        lastDetectionTimeRef.current = now
        runDetection(imageData)
      }

      animationFrameRef.current = requestAnimationFrame(render)
    }

    render()

    return () => {
      cancelAnimationFrame(animationFrameRef.current)
    }
  }, [camera.isActive, renderer, scene, width, height, detectionEnabled, detection.isReady])

  const runDetection = useCallback(async (imageData: ImageData) => {
    if (!detection.isReady) return

    try {
      const detections = await detection.detect(imageData)
      setLocalDetections(detections)
      setInferenceTime(detection.inferenceTime)
      onDetections?.(detections)
    } catch {
    }
  }, [detection, onDetections])

  useEffect(() => {
    const overlayCanvas = overlayCanvasRef.current
    const ctx = overlayCanvas?.getContext('2d')
    if (!overlayCanvas || !ctx || width <= 0 || height <= 0) return

    ctx.clearRect(0, 0, width, height)

    localDetections.forEach(det => {
      drawDetection(ctx, det, width, height)
    })

    fusedTracks.forEach(track => {
      drawFusedTrackIndicator(ctx, track, width, height)
    })
  }, [localDetections, fusedTracks, width, height])

  const statusColor = useMemo(() => {
    if (!camera.isActive) return '#666'
    if (localDetections.some(d => d.class === 'drone' && d.confidence > 0.7)) return '#8b4a4a'
    if (localDetections.length > 0) return '#a08040'
    return '#3a6b4a'
  }, [camera.isActive, localDetections])

  const highestThreat = useMemo(() => {
    if (localDetections.length === 0) return null
    return localDetections.reduce((max, det) =>
      (det.threatLevel ?? 0) > (max.threatLevel ?? 0) ? det : max
    )
  }, [localDetections])

  if (width <= 0 || height <= 0) return null

  return (
    <div
      className="camera-feed"
      onClick={onClick}
      style={{
        position: 'relative',
        width: `${width}px`,
        height: `${height}px`,
        backgroundColor: '#0a0a0a',
        border: `1px solid ${isSelected ? '#4a6a8a' : '#333'}`,
        overflow: 'hidden',
        cursor: 'pointer',
      }}
    >
      <canvas
        ref={feedCanvasRef}
        width={width}
        height={height}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
        }}
      />

      <canvas
        ref={overlayCanvasRef}
        width={width}
        height={height}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '4px 8px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          fontFamily: "'Roboto Mono', monospace",
          fontSize: '10px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: statusColor,
              boxShadow: camera.isActive ? `0 0 4px ${statusColor}` : 'none',
            }}
          />
          <span style={{ color: '#e0e0e0', fontWeight: 500 }}>{camera.name}</span>
          {camera.isRecording && (
            <span style={{ color: '#8b4a4a', fontWeight: 700 }}>REC</span>
          )}
        </div>
        <div style={{ color: '#888' }}>
          {camera.isActive
            ? `${inferenceTime.toFixed(0)}ms`
            : 'INAKTIV'
          }
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '4px 8px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          fontFamily: "'Roboto Mono', monospace",
          fontSize: '10px',
        }}
      >
        <div style={{ color: '#888' }}>
          DETEK: <span style={{ color: '#e0e0e0' }}>{localDetections.length}</span>
        </div>
        {highestThreat && (
          <div
            style={{
              color: THREAT_LEVEL_COLORS[highestThreat.threatLevel ?? 1],
              fontWeight: 500,
            }}
          >
            {highestThreat.class.toUpperCase()} {(highestThreat.confidence * 100).toFixed(0)}%
          </div>
        )}
        {!highestThreat && camera.isActive && (
          <div style={{ color: '#3a6b4a' }}>KLAR</div>
        )}
      </div>

      {!camera.isActive && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
          }}
        >
          <span
            style={{
              color: '#666',
              fontFamily: "'Roboto Mono', monospace",
              fontSize: '12px',
            }}
          >
            KEIN SIGNAL
          </span>
        </div>
      )}
    </div>
  )
}

function drawDetection(
  ctx: CanvasRenderingContext2D,
  detection: Detection,
  _canvasWidth: number,
  _canvasHeight: number
): void {
  const [x1, y1, x2, y2] = detection.bbox
  const color = DEFAULT_OVERLAY_STYLE.boxColor[detection.class]

  ctx.strokeStyle = color
  ctx.lineWidth = DEFAULT_OVERLAY_STYLE.boxWidth
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

  const bracketSize = Math.min(15, (x2 - x1) / 4, (y2 - y1) / 4)

  ctx.beginPath()
  ctx.moveTo(x1, y1 + bracketSize)
  ctx.lineTo(x1, y1)
  ctx.lineTo(x1 + bracketSize, y1)
  ctx.moveTo(x2 - bracketSize, y1)
  ctx.lineTo(x2, y1)
  ctx.lineTo(x2, y1 + bracketSize)
  ctx.moveTo(x2, y2 - bracketSize)
  ctx.lineTo(x2, y2)
  ctx.lineTo(x2 - bracketSize, y2)
  ctx.moveTo(x1 + bracketSize, y2)
  ctx.lineTo(x1, y2)
  ctx.lineTo(x1, y2 - bracketSize)
  ctx.stroke()

  const label = `${detection.class.toUpperCase()} ${(detection.confidence * 100).toFixed(0)}%`
  ctx.font = '10px "Roboto Mono", monospace'
  const labelWidth = ctx.measureText(label).width + 8
  const labelHeight = 14

  if (DEFAULT_OVERLAY_STYLE.labelBackground) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(x1, y1 - labelHeight - 2, labelWidth, labelHeight)
  }

  ctx.fillStyle = color
  ctx.fillText(label, x1 + 4, y1 - 4)

  if (detection.trackId && DEFAULT_OVERLAY_STYLE.showTrackId) {
    const trackLabel = detection.trackId
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(x1, y2 + 2, ctx.measureText(trackLabel).width + 8, labelHeight)
    ctx.fillStyle = '#4a6a8a'
    ctx.fillText(trackLabel, x1 + 4, y2 + 12)
  }
}

function drawFusedTrackIndicator(
  ctx: CanvasRenderingContext2D,
  track: FusedTrack,
  _canvasWidth: number,
  _canvasHeight: number
): void {
  const color = THREAT_LEVEL_COLORS[track.threatLevel]

  if (track.lastDetection?.bbox) {
    const [x1, y1] = track.lastDetection.bbox

    const size = 6
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.moveTo(x1 - size - 4, y1 + size)
    ctx.lineTo(x1 - size - 4 - size, y1)
    ctx.lineTo(x1 - size - 4, y1 - size)
    ctx.lineTo(x1 - size - 4 + size, y1)
    ctx.closePath()
    ctx.fill()
  }
}

export function getDetectionClassName(detClass: DetectionClass): string {
  const names: Record<DetectionClass, string> = {
    drone: 'DROHNE',
    bird: 'VOGEL',
    aircraft: 'FLUGZEUG',
    helicopter: 'HELIKOPTER',
    unknown: 'UNBEKANNT',
  }
  return names[detClass] || 'UNBEKANNT'
}
