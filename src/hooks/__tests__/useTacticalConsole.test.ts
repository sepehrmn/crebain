/**
 * Tests for useTacticalConsole hook
 * Verifies module exports and type definitions
 */
import { describe, it, expect } from 'vitest'
import { useTacticalConsole } from '../useTacticalConsole'
import type { MessageLevel, TacticalMessage, TacticalError } from '../useTacticalConsole'

describe('useTacticalConsole', () => {
  it('should export useTacticalConsole function', () => {
    expect(typeof useTacticalConsole).toBe('function')
  })

  it('should have correct type definitions', () => {
    const levels: MessageLevel[] = ['info', 'success', 'warning', 'error', 'tactical', 'system']
    expect(levels).toHaveLength(6)

    const message: TacticalMessage = {
      id: 'test-1',
      level: 'info',
      text: 'Test message',
      timestamp: Date.now(),
      code: 'TEST_001',
    }
    expect(message.id).toBe('test-1')
    expect(message.level).toBe('info')

    const error: TacticalError = {
      severity: 'error',
      code: 'ERR_001',
      message: 'Test error',
      context: { foo: 'bar' },
    }
    expect(error.severity).toBe('error')
    expect(error.code).toBe('ERR_001')
  })
})
