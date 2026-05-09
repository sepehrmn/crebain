import { beforeEach, describe, expect, it, vi } from 'vitest'

const invokeMock = vi.hoisted(() => vi.fn())

vi.mock('@tauri-apps/api/core', () => ({
  invoke: invokeMock,
}))

import {
  clearTracks,
  getAlgorithms,
  getFusionStats,
  getModalities,
  getTracks,
  initFusion,
  processMeasurements,
  setFusionConfig,
  type FusionConfig,
  type SensorMeasurement,
} from '../AdvancedSensorFusion'

describe('AdvancedSensorFusion IPC', () => {
  beforeEach(() => {
    invokeMock.mockReset()
  })

  it('routes fusion initialization through the centralized command name', async () => {
    invokeMock.mockResolvedValue(undefined)

    await initFusion({ algorithm: 'IMM' })

    expect(invokeMock).toHaveBeenCalledWith('fusion_init', {
      config: {
        algorithm: 'IMM',
        process_noise: 1,
        measurement_noise: 2,
        association_threshold: 10,
        max_missed_detections: 5,
        min_confirmation_hits: 3,
        particle_count: 100,
      },
    })
  })

  it('routes measurement processing with explicit timestamps', async () => {
    invokeMock.mockResolvedValue([])
    const measurement: SensorMeasurement = {
      sensor_id: 'camera-1',
      modality: 'visual',
      timestamp_ms: 123,
      position: [1, 2, 3],
      covariance: [1, 1, 1],
      confidence: 0.9,
      class_label: 'drone',
      metadata: {},
    }

    await processMeasurements([measurement], 456)

    expect(invokeMock).toHaveBeenCalledWith('fusion_process', {
      measurements: [measurement],
      timestampMs: 456,
    })
  })

  it('routes query and mutation commands', async () => {
    invokeMock.mockResolvedValue(undefined)
    const config: FusionConfig = {
      algorithm: 'ExtendedKalman',
      process_noise: 1,
      measurement_noise: 2,
      association_threshold: 10,
      max_missed_detections: 5,
      min_confirmation_hits: 3,
      particle_count: 100,
    }

    await getTracks()
    await getFusionStats()
    await setFusionConfig(config)
    await clearTracks()
    await getAlgorithms()
    await getModalities()

    expect(invokeMock.mock.calls.map(call => call[0])).toEqual([
      'fusion_get_tracks',
      'fusion_get_stats',
      'fusion_set_config',
      'fusion_clear',
      'fusion_get_algorithms',
      'fusion_get_modalities',
    ])
    expect(invokeMock).toHaveBeenCalledWith('fusion_set_config', { config })
  })
})
