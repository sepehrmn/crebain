/**
 * Tests for useScheduling utilities
 * Verifies module exports and type definitions
 */
import { describe, it, expect } from 'vitest'
import { 
  useAnimationLoop, 
  usePolling, 
  useRateLimitedPolling, 
  useDebouncedCallback 
} from '../useScheduling'

describe('useScheduling', () => {
  it('should export useAnimationLoop function', () => {
    expect(typeof useAnimationLoop).toBe('function')
  })

  it('should export usePolling function', () => {
    expect(typeof usePolling).toBe('function')
  })

  it('should export useRateLimitedPolling function', () => {
    expect(typeof useRateLimitedPolling).toBe('function')
  })

  it('should export useDebouncedCallback function', () => {
    expect(typeof useDebouncedCallback).toBe('function')
  })
})
