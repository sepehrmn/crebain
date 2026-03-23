/**
 * CREBAIN Camera Grid Component
 * Adaptive Response & Awareness System (ARAS)
 *
 * 2x2 grid layout for multi-camera surveillance display
 */

import { useRef, useEffect, useCallback, useState } from 'react'
import * as THREE from 'three'
import { CameraFeed } from './CameraFeed'
import type { SurveillanceCamera, Detection, FusedTrack } from '../detection/types'
import type { RendererWithAsync } from './viewer/types'

interface CameraGridProps {
  cameras: SurveillanceCamera[]
  renderer: RendererWithAsync | null
  scene: THREE.Scene | null
  onCameraSelect?: (cameraId: string) => void
  selectedCameraId?: string
  fusedTracks?: FusedTrack[]
  detectionEnabled?: boolean
  onDetection?: (cameraId: string, detections: Detection[]) => void
}

export function CameraGrid({
  cameras,
  renderer,
  scene,
  onCameraSelect,
  selectedCameraId,
  fusedTracks = [],
  detectionEnabled = true,
  onDetection,
}: CameraGridProps) {
  const gridRef = useRef<HTMLDivElement>(null)
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 })

  // Calculate grid dimensions
  useEffect(() => {
    const updateSize = () => {
      if (gridRef.current) {
        setGridSize({
          width: gridRef.current.clientWidth,
          height: gridRef.current.clientHeight,
        })
      }
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Calculate individual feed size
  const feedWidth = Math.floor(gridSize.width / 2) - 4
  const feedHeight = Math.floor(gridSize.height / 2) - 4

  const handleCameraClick = useCallback((cameraId: string) => {
    onCameraSelect?.(cameraId)
  }, [onCameraSelect])

  const handleDetections = useCallback((cameraId: string, detections: Detection[]) => {
    onDetection?.(cameraId, detections)
  }, [onDetection])

  // Ensure we always have exactly 4 camera slots
  const displayCameras = [...cameras]
  while (displayCameras.length < 4) {
    displayCameras.push(createPlaceholderCamera(displayCameras.length))
  }

  return (
    <div
      ref={gridRef}
      className="camera-grid"
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gridTemplateRows: 'repeat(2, 1fr)',
        gap: '4px',
        width: '100%',
        height: '100%',
        backgroundColor: '#1a1a1a',
        padding: '4px',
      }}
    >
      {displayCameras.slice(0, 4).map((camera) => (
        <CameraFeed
          key={camera.id}
          camera={camera}
          renderer={renderer}
          scene={scene}
          width={feedWidth}
          height={feedHeight}
          isSelected={camera.id === selectedCameraId}
          onClick={() => handleCameraClick(camera.id)}
          detectionEnabled={detectionEnabled && camera.isActive}
          onDetections={(dets) => handleDetections(camera.id, dets)}
          fusedTracks={fusedTracks.filter(t =>
            t.contributingCameras.includes(camera.id)
          )}
        />
      ))}
    </div>
  )
}

/**
 * Create placeholder camera for empty grid slots
 * Returns a minimal SurveillanceCamera that matches the interface
 */
function createPlaceholderCamera(index: number): SurveillanceCamera {
  const position = new THREE.Vector3(0, 0, 0)
  const target = new THREE.Vector3(0, 0, -1)
  return {
    id: `placeholder-${index}`,
    name: `KAMERA ${String(index + 1).padStart(3, '0')}`,
    position,
    target,
    fov: 60,
    aspectRatio: 16 / 9,
    isActive: false,
    isRecording: false,
    detections: [],
    lastInferenceTime: 0,
    inferenceLatency: 0,
    trackingEnabled: false,
    fusionWeight: 1.0,
    pan: 0,
    tilt: 0,
    zoom: 1.0,
  }
}

/**
 * Camera grid status summary
 */
interface GridStatusProps {
  cameras: SurveillanceCamera[]
  totalDetections: number
  activeTracks: number
  fusionEnabled: boolean
}

export function CameraGridStatus({
  cameras,
  totalDetections,
  activeTracks,
  fusionEnabled,
}: GridStatusProps) {
  const activeCameras = cameras.filter(c => c.isActive).length
  const recordingCameras = cameras.filter(c => c.isRecording).length

  return (
    <div
      className="camera-grid-status"
      style={{
        display: 'flex',
        gap: '16px',
        padding: '8px 12px',
        backgroundColor: '#1e1e1e',
        borderBottom: '1px solid #333',
        fontFamily: "'Roboto Mono', monospace",
        fontSize: '11px',
      }}
    >
      <StatusItem
        label="KAMERAS"
        value={`${activeCameras}/${cameras.length}`}
        status={activeCameras === cameras.length ? 'good' : 'warning'}
      />
      <StatusItem
        label="AUFZ."
        value={recordingCameras.toString()}
        status={recordingCameras > 0 ? 'recording' : 'inactive'}
      />
      <StatusItem
        label="DETEK."
        value={totalDetections.toString()}
        status={totalDetections > 0 ? 'active' : 'inactive'}
      />
      <StatusItem
        label="TRACKS"
        value={activeTracks.toString()}
        status={activeTracks > 0 ? 'tracking' : 'inactive'}
      />
      <StatusItem
        label="FUSION"
        value={fusionEnabled ? 'AN' : 'AUS'}
        status={fusionEnabled ? 'good' : 'inactive'}
      />
    </div>
  )
}

interface StatusItemProps {
  label: string
  value: string
  status: 'good' | 'warning' | 'error' | 'recording' | 'active' | 'tracking' | 'inactive'
}

function StatusItem({ label, value, status }: StatusItemProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'good':
      case 'active':
      case 'tracking':
        return '#3a6b4a'
      case 'warning':
        return '#a08040'
      case 'error':
        return '#8b4a4a'
      case 'recording':
        return '#8b4a4a'
      case 'inactive':
      default:
        return '#666'
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: getStatusColor(),
          boxShadow: status !== 'inactive' ? `0 0 4px ${getStatusColor()}` : 'none',
        }}
      />
      <span style={{ color: '#888' }}>{label}:</span>
      <span style={{ color: '#e0e0e0', fontWeight: 500 }}>{value}</span>
    </div>
  )
}
