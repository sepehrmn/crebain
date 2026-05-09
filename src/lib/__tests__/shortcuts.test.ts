import { describe, it, expect } from 'vitest'
import { APP_SHORTCUTS, DRONE_CONTROL_SHORTCUTS, VIEWER_SHORTCUTS, isTextInputTarget, normalizeShortcutKey } from '../shortcuts'

describe('shortcuts', () => {
  it('keeps app panel shortcuts centralized', () => {
    expect(APP_SHORTCUTS.togglePerformancePanel).toBe('p')
    expect(APP_SHORTCUTS.toggleROSPanel).toBe('n')
    expect(APP_SHORTCUTS.toggleFusionPanel).toBe('u')
  })

  it('keeps viewer shortcuts aligned with documented behavior', () => {
    expect(VIEWER_SHORTCUTS.toggleCameraFeeds).toBe('v')
    expect(VIEWER_SHORTCUTS.toggleDetectionPanel).toBe('t')
    expect(VIEWER_SHORTCUTS.toggleDetectionEnabled).toBe('y')
    expect(VIEWER_SHORTCUTS.focusContent).toBe('f')
    expect(VIEWER_SHORTCUTS.toggleGrid).toBe('g')
  })

  it('normalizes keyboard keys', () => {
    expect(normalizeShortcutKey('P')).toBe('p')
    expect(normalizeShortcutKey('Tab')).toBe('tab')
    expect(normalizeShortcutKey('Escape')).toBe('escape')
  })

  it('keeps drone control shortcuts centralized', () => {
    expect(DRONE_CONTROL_SHORTCUTS.forward).toBe('w')
    expect(DRONE_CONTROL_SHORTCUTS.backward).toBe('s')
    expect(DRONE_CONTROL_SHORTCUTS.left).toBe('a')
    expect(DRONE_CONTROL_SHORTCUTS.right).toBe('d')
    expect(DRONE_CONTROL_SHORTCUTS.yawLeft).toBe('q')
    expect(DRONE_CONTROL_SHORTCUTS.yawRight).toBe('e')
    expect(DRONE_CONTROL_SHORTCUTS.up).toBe(' ')
    expect(DRONE_CONTROL_SHORTCUTS.down).toBe('shift')
    expect(DRONE_CONTROL_SHORTCUTS.armToggle).toBe('r')
    expect(DRONE_CONTROL_SHORTCUTS.emergency).toBe('escape')
  })

  it('detects text input targets', () => {
    expect(isTextInputTarget(document.createElement('input'))).toBe(true)
    expect(isTextInputTarget(document.createElement('textarea'))).toBe(true)
    expect(isTextInputTarget(document.createElement('div'))).toBe(false)
    expect(isTextInputTarget(null)).toBe(false)
  })
})
