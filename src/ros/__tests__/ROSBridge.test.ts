import { describe, expect, it, vi } from 'vitest'
import { ROSBridge, validateRosUrl } from '../ROSBridge'
import type { PoseStamped, TwistStamped } from '../types'

describe('ROSBridge URL validation', () => {
  it('accepts websocket URLs', () => {
    expect(validateRosUrl('ws://localhost:9090')).toEqual({ valid: true })
    expect(validateRosUrl('wss://ros.example.com/bridge')).toEqual({ valid: true })
  })

  it('rejects non-websocket schemes', () => {
    expect(validateRosUrl('http://localhost:9090')).toMatchObject({
      valid: false,
      error: 'Invalid scheme: http:. Only ws:// and wss:// are allowed.',
    })
  })

  it('rejects malformed URLs', () => {
    expect(validateRosUrl('not-a-url')).toMatchObject({
      valid: false,
      error: 'Invalid URL format',
    })
  })

  it('rejects invalid hostname formats', () => {
    expect(validateRosUrl('ws://-bad-host:9090')).toMatchObject({
      valid: false,
      error: 'Invalid hostname format',
    })
  })

  it('throws when constructed with an invalid URL', () => {
    expect(() => new ROSBridge({ url: 'file:///tmp/socket' })).toThrow('Invalid ROS bridge URL')
  })

  it('normalizes namespaces for setpoint publishers', () => {
    const bridge = new ROSBridge({ url: 'ws://localhost:9090' })
    const publish = vi.spyOn(bridge, 'publish').mockImplementation(() => undefined)
    const pose: PoseStamped = {
      header: { stamp: { secs: 0, nsecs: 0 }, frame_id: 'map' },
      pose: {
        position: { x: 1, y: 2, z: 3 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    }
    const twist: TwistStamped = {
      header: { stamp: { secs: 0, nsecs: 0 }, frame_id: 'map' },
      twist: {
        linear: { x: 1, y: 0, z: 0 },
        angular: { x: 0, y: 0, z: 0 },
      },
    }

    bridge.publishSetpointPosition('/drone1/', pose)
    bridge.publishSetpointVelocity('///drone2///', twist)

    expect(publish).toHaveBeenNthCalledWith(1, '/drone1/mavros/setpoint_position/local', pose)
    expect(publish).toHaveBeenNthCalledWith(2, '/drone2/mavros/setpoint_velocity/cmd_vel', twist)
  })

  it('normalizes namespaces for MAVROS service helpers', async () => {
    const bridge = new ROSBridge({ url: 'ws://localhost:9090' })
    const callService = vi.spyOn(bridge, 'callService')
    callService.mockResolvedValueOnce({ mode_sent: true })
    callService.mockResolvedValueOnce({ success: true })

    await expect(bridge.setMode('/drone1/', 'OFFBOARD')).resolves.toBe(true)
    await expect(bridge.arm('///drone1///')).resolves.toBe(true)

    expect(callService).toHaveBeenNthCalledWith(1, '/drone1/mavros/set_mode', { custom_mode: 'OFFBOARD' })
    expect(callService).toHaveBeenNthCalledWith(2, '/drone1/mavros/cmd/arming', { value: true })
  })
})
