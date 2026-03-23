/**
 * CREBAIN Multi-Sensor Fusion Panel
 * Adaptive Response & Awareness System (ARAS)
 *
 * Real-time display of fused tracks from multiple sensor modalities
 */

import { useState, useEffect, useCallback, useMemo, memo } from 'react'
import { useDraggablePanel } from '../hooks/useDraggablePanel'
import { PANEL_POSITIONS } from './BasePanel'
import type {
  FusedTrack,
  FusionStats,
  FilterAlgorithm,
  SensorModality,
} from '../detection/AdvancedSensorFusion'
import {
  getThreatColor,
  getTrackStateColor,
  formatAlgorithmName,
  formatModality,
  getAlgorithms,
  setFusionConfig,
} from '../detection/AdvancedSensorFusion'
import { fusionLogger as log } from '../lib/logger'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface SensorFusionPanelProps {
  tracks: FusedTrack[]
  stats: FusionStats | null
  sensorStatus: Record<SensorModality, boolean>
  isExpanded?: boolean
  onToggleExpand?: () => void
  onSelectTrack?: (trackId: string) => void
  selectedTrackId?: string | null
}

interface AlgorithmOption {
  id: FilterAlgorithm
  name: string
  description: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// SENSOR ICONS (inline SVG for tactical display)
// ═══════════════════════════════════════════════════════════════════════════════

const SensorIcon = ({ modality, active }: { modality: SensorModality; active: boolean }) => {
  const color = active ? '#3a6b4a' : '#404040'
  const size = 12

  switch (modality) {
    case 'visual':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M2 12s4-8 10-8 10 8 10 8-4 8-10 8-10-8-10-8z" />
        </svg>
      )
    case 'thermal':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
        </svg>
      )
    case 'acoustic':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" />
        </svg>
      )
    case 'radar':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 2v10l7 7" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      )
    case 'lidar':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M2 12h2M20 12h2M12 2v2M12 20v2" />
          <circle cx="12" cy="12" r="4" />
          <circle cx="12" cy="12" r="8" strokeDasharray="2 2" />
        </svg>
      )
    case 'radiofrequency':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01" />
        </svg>
      )
    default:
      return null
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function SensorFusionPanel({
  tracks,
  stats,
  sensorStatus,
  isExpanded = true,
  onToggleExpand,
  onSelectTrack,
  selectedTrackId,
}: SensorFusionPanelProps) {
  const [algorithms, setAlgorithms] = useState<AlgorithmOption[]>([])
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<FilterAlgorithm>('ExtendedKalman')
  const [showSettings, setShowSettings] = useState(false)

  // Use combined draggable panel hook
  // Use centralized position from PANEL_POSITIONS
  const fusionPosition = PANEL_POSITIONS.sensorFusion
  const { panelStyle, handleMouseDown, handleHeaderClick, elementRef } = useDraggablePanel({
    initialPosition: fusionPosition.initialPosition,
    snapDistance: fusionPosition.snapDistance,
    edgePadding: fusionPosition.edgePadding,
    side: fusionPosition.side,
    onHeaderClick: onToggleExpand,
  })

  // Load available algorithms
  useEffect(() => {
    getAlgorithms()
      .then((algos) => setAlgorithms(algos as AlgorithmOption[]))
      .catch((err) => log.error('Failed to load algorithms', { error: err }))
  }, [])

  // Handle algorithm change
  const handleAlgorithmChange = useCallback(async (algorithm: FilterAlgorithm) => {
    setSelectedAlgorithm(algorithm)
    try {
      await setFusionConfig({
        algorithm,
        process_noise: 1.0,
        measurement_noise: 2.0,
        association_threshold: 10.0,
        max_missed_detections: 5,
        min_confirmation_hits: 3,
        particle_count: 100,
      })
    } catch (error) {
      log.error('Failed to set algorithm', { error })
    }
  }, [])

  // Sort tracks by threat level (memoized to prevent unnecessary re-renders)
  const sortedTracks = useMemo(
    () => [...tracks].sort((a, b) => b.threat_level - a.threat_level),
    [tracks]
  )

  // Count sensors (memoized)
  const { activeSensors, totalSensors } = useMemo(() => ({
    activeSensors: Object.values(sensorStatus).filter(Boolean).length,
    totalSensors: Object.keys(sensorStatus).length,
  }), [sensorStatus])

  // Max threat level (memoized)
  const maxThreatLevel = useMemo(
    () => Math.max(...tracks.map((t) => t.threat_level), 0),
    [tracks]
  )

  if (!isExpanded) {
    return (
      <div
        ref={elementRef}
        className="absolute top-0 right-3 z-40"
        style={panelStyle}
        onMouseDown={handleMouseDown}
      >
        <button
          data-drag-handle
          onClick={handleHeaderClick}
          className="px-3 py-2 bg-[#0c0c0c] border border-[#252525] text-[1.25em] hover:border-[#404040] cursor-grab select-none"
        >
          <div className="flex items-center gap-2">
            <span className="text-[#606060]">FUSION</span>
            <span className={tracks.length > 0 ? 'text-[#3a6b4a]' : 'text-[#404040]'}>
              {tracks.length} TRK
            </span>
            <span className="text-[#505050]">|</span>
            <span className="text-[#606060]">{formatAlgorithmName(selectedAlgorithm)}</span>
          </div>
        </button>
      </div>
    )
  }

  return (
    <div
      ref={elementRef}
      className="absolute top-0 right-3 w-72 z-40"
      style={panelStyle}
      onMouseDown={handleMouseDown}
    >
      <div className="bg-[#0c0c0c] border border-[#1a1a1a]">
        {/* Header - Drag Handle */}
        <div
          data-drag-handle
          className="h-8 border-b border-[#1a1a1a] flex items-center justify-between px-3 bg-[#101010] cursor-grab select-none"
          onClick={handleHeaderClick}
        >
          <div className="flex items-center gap-2">
            <span className="text-[1.25em] text-[#909090] tracking-[0.2em]">SENSOR FUSION</span>
            <span className="text-[1.125em] px-1.5 py-0.5 bg-[#1a1a1a] border border-[#252525] text-[#707070]">
              {formatAlgorithmName(stats?.algorithm ?? selectedAlgorithm)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings) }}
              className="text-[1.125em] text-[#505050] hover:text-[#909090]"
            >
              ⚙
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onToggleExpand?.() }}
              className="text-[1.125em] text-[#505050] hover:text-[#909090]"
            >
              ─
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="border-b border-[#1a1a1a] p-2 bg-[#0a0a0a]">
            <div className="text-[1.25em] text-[#505050] tracking-wider mb-2">FILTER ALGORITHMUS</div>
            <div className="grid grid-cols-5 gap-1">
              {algorithms.map((algo) => (
                <button
                  key={algo.id}
                  onClick={() => handleAlgorithmChange(algo.id)}
                  title={algo.description}
                  className={`py-1.5 text-[1.125em] border transition-all ${
                    selectedAlgorithm === algo.id
                      ? 'bg-[#1a2a1a] border-[#3a6b4a] text-[#6a9a7a]'
                      : 'bg-[#0c0c0c] border-[#1a1a1a] text-[#505050] hover:border-[#303030]'
                  }`}
                >
                  {formatAlgorithmName(algo.id)}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Sensor Status Bar */}
        <div className="h-7 border-b border-[#1a1a1a] flex items-center justify-between px-3 bg-[#0a0a0a]">
          <div className="flex items-center gap-3">
            {(Object.entries(sensorStatus) as [SensorModality, boolean][]).map(([modality, active]) => (
              <div
                key={modality}
                className="flex items-center gap-1"
                title={`${modality.toUpperCase()}: ${active ? 'AKTIV' : 'INAKTIV'}`}
              >
                <SensorIcon modality={modality} active={active} />
                <span className={`text-[1.25em] ${active ? 'text-[#6a9a7a]' : 'text-[#404040]'}`}>
                  {formatModality(modality)}
                </span>
              </div>
            ))}
          </div>
          <div className="text-[1.125em] text-[#505050]">
            {activeSensors}/{totalSensors}
          </div>
        </div>

        {/* Stats Bar */}
        {stats && (
          <div className="h-6 border-b border-[#1a1a1a] flex items-center justify-between px-3 text-[1.25em] text-[#505050]">
            <span>FRAME: {stats.frame_count}</span>
            <span className="text-[#3a6b4a]">KONF: {stats.confirmed_tracks}</span>
            <span className="text-[#a08040]">TENT: {stats.tentative_tracks}</span>
            <span>MULTI: {stats.multi_sensor_tracks}</span>
          </div>
        )}

        {/* Track List */}
        <div className="max-h-80 overflow-y-auto">
          {sortedTracks.length === 0 ? (
            <div className="p-4 text-center text-[1.25em] text-[#404040]">
              KEINE AKTIVEN TRACKS
            </div>
          ) : (
            <div className="divide-y divide-[#1a1a1a]">
              {sortedTracks.map((track) => (
                <TrackRow
                  key={track.id}
                  track={track}
                  isSelected={selectedTrackId === track.id}
                  onClick={() => onSelectTrack?.(track.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="h-6 border-t border-[#1a1a1a] flex items-center justify-between px-3 text-[1.25em] text-[#404040]">
          <span>TRACKS: {tracks.length}</span>
          <span>
            BEDROHUNG:{' '}
            <span
              style={{
                color: getThreatColor(maxThreatLevel),
              }}
            >
              {maxThreatLevel}
            </span>
          </span>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRACK ROW COMPONENT (memoized for performance)
// ═══════════════════════════════════════════════════════════════════════════════

interface TrackRowProps {
  track: FusedTrack
  isSelected: boolean
  onClick: () => void
}

const TrackRow = memo(function TrackRow({ track, isSelected, onClick }: TrackRowProps) {
  const threatColor = getThreatColor(track.threat_level)
  const stateColor = getTrackStateColor(track.state)

  // Format position
  const pos = track.position
  const posStr = `${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)}`

  // Format velocity magnitude
  const vel = track.velocity
  const speed = Math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

  // Format uncertainty (RMS of position uncertainty)
  const unc = track.position_uncertainty
  const rmsUnc = Math.sqrt((unc[0] ** 2 + unc[1] ** 2 + unc[2] ** 2) / 3)

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 transition-all ${
        isSelected
          ? 'bg-[#141a14] border-l-2 border-[#3a6b4a]'
          : 'hover:bg-[#0e0e0e] border-l-2 border-transparent'
      }`}
    >
      {/* Row 1: ID, Class, Threat */}
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2"
            style={{ backgroundColor: threatColor }}
          />
          <span className="text-[1.25em] text-[#b0b0b0] font-medium">
            {track.id}
          </span>
          <span className="text-[1.125em] text-[#707070]">
            {track.class_label.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="text-[1.25em] px-1 py-0.5 border"
            style={{
              borderColor: stateColor,
              color: stateColor,
            }}
          >
            {track.state === 'Confirmed' ? 'KONF' : 
             track.state === 'Tentative' ? 'TENT' :
             track.state === 'Coasting' ? 'COAST' : 'LOST'}
          </span>
          <span
            className="text-[1.25em] font-bold"
            style={{ color: threatColor }}
          >
            T{track.threat_level}
          </span>
        </div>
      </div>

      {/* Row 2: Position & Speed */}
      <div className="flex items-center justify-between text-[1.125em] text-[#606060] mb-1">
        <span>POS: {posStr}</span>
        <span>SPD: {speed.toFixed(1)} m/s</span>
      </div>

      {/* Row 3: Sensors & Confidence */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          {track.sensor_sources.map((modality) => (
            <span
              key={modality}
              className="text-[1.25em] px-1 py-0.5 bg-[#1a1a1a] border border-[#252525] text-[#707070]"
            >
              {formatModality(modality)}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-2 text-[1.25em]">
          <span className="text-[#505050]">
            σ: {rmsUnc.toFixed(1)}m
          </span>
          <span style={{ color: getThreatColor(track.confidence > 0.7 ? 3 : track.confidence > 0.4 ? 2 : 1) }}>
            {(track.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </button>
  )
})
