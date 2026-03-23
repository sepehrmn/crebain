/**
 * Tests for useSurveillanceCameras hook
 * Verifies module exports and type definitions
 */
import { describe, it, expect } from 'vitest'
import { useSurveillanceCameras } from '../useSurveillanceCameras'

describe('useSurveillanceCameras', () => {
  it('should export useSurveillanceCameras function', () => {
    expect(typeof useSurveillanceCameras).toBe('function')
  })
})
