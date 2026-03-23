/**
 * CREBAIN Advanced Sensor Fusion Frontend
 * Adaptive Response & Awareness System (ARAS)
 *
 * TypeScript interface to the native Rust sensor fusion backend
 * Supports: Kalman, EKF, UKF, Particle Filter, IMM
 */

import { invoke } from '@tauri-apps/api/core'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/** Sensor modality types */
export type SensorModality =
  | 'visual'
  | 'thermal'
  | 'acoustic'
  | 'radar'
  | 'lidar'
  | 'radiofrequency'

/** Filter algorithm selection */
export type FilterAlgorithm =
  | 'Kalman'
  | 'ExtendedKalman'
  | 'UnscentedKalman'
  | 'Particle'
  | 'IMM'

/** Track state labels */
export type TrackStateLabel = 'Tentative' | 'Confirmed' | 'Coasting' | 'Lost'

/** Raw sensor measurement from any modality */
export interface SensorMeasurement {
  sensor_id: string
  modality: SensorModality
  timestamp_ms: number
  /** Position in sensor frame [x, y, z] or [azimuth, elevation, range] */
  position: [number, number, number]
  /** Velocity if available [vx, vy, vz] */
  velocity?: [number, number, number]
  /** Measurement covariance (diagonal elements) */
  covariance: [number, number, number]
  /** Detection confidence [0, 1] */
  confidence: number
  /** Classification label */
  class_label: string
  /** Additional sensor-specific data */
  metadata: Record<string, number>
}

/** Thermal-specific measurement */
export interface ThermalMeasurement extends SensorMeasurement {
  modality: 'thermal'
  /** Temperature in Kelvin */
  temperature_k: number
  /** Thermal signature area in m² */
  signature_area: number
  /** Emissivity estimate */
  emissivity: number
}

/** Acoustic-specific measurement */
export interface AcousticMeasurement extends SensorMeasurement {
  modality: 'acoustic'
  /** Sound pressure level in dB */
  spl_db: number
  /** Dominant frequency in Hz */
  frequency_hz: number
  /** Direction of arrival [azimuth, elevation] in radians */
  doa: [number, number]
  /** Doppler shift in Hz */
  doppler_hz?: number
}

/** Fused track output from backend */
export interface FusedTrack {
  id: string
  position: [number, number, number]
  velocity: [number, number, number]
  position_uncertainty: [number, number, number]
  velocity_uncertainty: [number, number, number]
  class_label: string
  confidence: number
  sensor_sources: SensorModality[]
  last_update_ms: number
  age: number
  state: TrackStateLabel
  threat_level: number
}

/** Fusion engine configuration */
export interface FusionConfig {
  algorithm: FilterAlgorithm
  process_noise: number
  measurement_noise: number
  association_threshold: number
  max_missed_detections: number
  min_confirmation_hits: number
  particle_count: number
}

/** Fusion statistics */
export interface FusionStats {
  total_tracks: number
  confirmed_tracks: number
  tentative_tracks: number
  coasting_tracks: number
  multi_sensor_tracks: number
  algorithm: FilterAlgorithm
  frame_count: number
}

/** Algorithm info */
export interface AlgorithmInfo {
  id: FilterAlgorithm
  name: string
  description: string
}

/** Modality info */
export interface ModalityInfo {
  id: SensorModality
  name: string
  icon: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Initialize the sensor fusion engine
 */
export async function initFusion(config?: Partial<FusionConfig>): Promise<void> {
  const fullConfig: FusionConfig | undefined = config
    ? {
        algorithm: config.algorithm ?? 'ExtendedKalman',
        process_noise: config.process_noise ?? 1.0,
        measurement_noise: config.measurement_noise ?? 2.0,
        association_threshold: config.association_threshold ?? 10.0,
        max_missed_detections: config.max_missed_detections ?? 5,
        min_confirmation_hits: config.min_confirmation_hits ?? 3,
        particle_count: config.particle_count ?? 100,
      }
    : undefined

  await invoke('fusion_init', { config: fullConfig })
}

/**
 * Process sensor measurements and get fused tracks
 */
export async function processMeasurements(
  measurements: SensorMeasurement[],
  timestampMs?: number
): Promise<FusedTrack[]> {
  const ts = timestampMs ?? Date.now()
  return invoke<FusedTrack[]>('fusion_process', {
    measurements,
    timestampMs: ts,
  })
}

/**
 * Get current tracks without processing new measurements
 */
export async function getTracks(): Promise<FusedTrack[]> {
  return invoke<FusedTrack[]>('fusion_get_tracks')
}

/**
 * Get fusion statistics
 */
export async function getFusionStats(): Promise<FusionStats> {
  return invoke<FusionStats>('fusion_get_stats')
}

/**
 * Update fusion configuration
 */
export async function setFusionConfig(config: FusionConfig): Promise<void> {
  await invoke('fusion_set_config', { config })
}

/**
 * Clear all tracks
 */
export async function clearTracks(): Promise<void> {
  await invoke('fusion_clear')
}

/**
 * Get available filter algorithms
 */
export async function getAlgorithms(): Promise<AlgorithmInfo[]> {
  return invoke<AlgorithmInfo[]>('fusion_get_algorithms')
}

/**
 * Get available sensor modalities
 */
export async function getModalities(): Promise<ModalityInfo[]> {
  return invoke<ModalityInfo[]>('fusion_get_modalities')
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a visual camera measurement from detection
 */
export function createVisualMeasurement(
  sensorId: string,
  position: [number, number, number],
  confidence: number,
  classLabel: string,
  covariance: [number, number, number] = [1, 1, 1]
): SensorMeasurement {
  return {
    sensor_id: sensorId,
    modality: 'visual',
    timestamp_ms: Date.now(),
    position,
    covariance,
    confidence,
    class_label: classLabel,
    metadata: {},
  }
}

/**
 * Create a thermal measurement
 */
export function createThermalMeasurement(
  sensorId: string,
  position: [number, number, number],
  confidence: number,
  classLabel: string,
  temperatureK: number,
  signatureArea: number = 0.5,
  emissivity: number = 0.9
): ThermalMeasurement {
  return {
    sensor_id: sensorId,
    modality: 'thermal',
    timestamp_ms: Date.now(),
    position,
    covariance: [2, 2, 2], // Thermal typically has higher uncertainty
    confidence,
    class_label: classLabel,
    metadata: {},
    temperature_k: temperatureK,
    signature_area: signatureArea,
    emissivity,
  }
}

/**
 * Create an acoustic measurement
 */
export function createAcousticMeasurement(
  sensorId: string,
  doa: [number, number], // [azimuth, elevation] in radians
  confidence: number,
  classLabel: string,
  splDb: number,
  frequencyHz: number,
  dopplerHz?: number
): AcousticMeasurement {
  // Convert DOA to approximate Cartesian (assume 50m range for initial estimate)
  const range = 50
  const x = range * Math.cos(doa[1]) * Math.cos(doa[0])
  const y = range * Math.cos(doa[1]) * Math.sin(doa[0])
  const z = range * Math.sin(doa[1])

  return {
    sensor_id: sensorId,
    modality: 'acoustic',
    timestamp_ms: Date.now(),
    position: [x, y, z],
    covariance: [10, 10, 10], // Acoustic has high position uncertainty
    confidence,
    class_label: classLabel,
    metadata: {},
    spl_db: splDb,
    frequency_hz: frequencyHz,
    doa,
    doppler_hz: dopplerHz,
  }
}

/**
 * Get threat color based on level
 */
export function getThreatColor(level: number): string {
  switch (level) {
    case 1:
      return '#3a6b4a' // Green - low
    case 2:
      return '#6a8a4a' // Yellow-green - moderate
    case 3:
      return '#a08040' // Amber - elevated
    case 4:
      return '#8b4a4a' // Red - critical
    default:
      return '#606060' // Gray - unknown
  }
}

/**
 * Get track state color
 */
export function getTrackStateColor(state: TrackStateLabel): string {
  switch (state) {
    case 'Confirmed':
      return '#3a6b4a'
    case 'Tentative':
      return '#a08040'
    case 'Coasting':
      return '#6a6a6a'
    case 'Lost':
      return '#4a4a4a'
    default:
      return '#606060'
  }
}

/**
 * Format algorithm name for display
 */
export function formatAlgorithmName(algorithm: FilterAlgorithm): string {
  switch (algorithm) {
    case 'Kalman':
      return 'KF'
    case 'ExtendedKalman':
      return 'EKF'
    case 'UnscentedKalman':
      return 'UKF'
    case 'Particle':
      return 'PF'
    case 'IMM':
      return 'IMM'
    default:
      return algorithm
  }
}

/**
 * Format sensor modality for display
 */
export function formatModality(modality: SensorModality): string {
  const modalityMap: Record<SensorModality, string> = {
    visual: 'VIS',
    thermal: 'IR',
    acoustic: 'ACO',
    radar: 'RAD',
    lidar: 'LID',
    radiofrequency: 'RF',
  }
  return modalityMap[modality] ?? modality.slice(0, 3).toUpperCase()
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEFAULT EXPORT
// ═══════════════════════════════════════════════════════════════════════════════

const AdvancedSensorFusion = {
  // Core API
  initFusion,
  processMeasurements,
  getTracks,
  getFusionStats,
  setFusionConfig,
  clearTracks,
  getAlgorithms,
  getModalities,

  // Measurement creators
  createVisualMeasurement,
  createThermalMeasurement,
  createAcousticMeasurement,

  // Helpers
  getThreatColor,
  getTrackStateColor,
  formatAlgorithmName,
  formatModality,
}

export default AdvancedSensorFusion
