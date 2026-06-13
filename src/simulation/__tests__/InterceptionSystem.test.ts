import { afterEach, describe, expect, it, vi } from 'vitest'
import { InterceptionSystem } from '../InterceptionSystem'

function createSystem() {
  const system = new InterceptionSystem()
  system.registerInterceptor(
    'interceptor-1',
    { x: 0, y: 0, z: 0 },
    { x: 0, y: 0, z: 0 },
    { maxSpeed: 20 }
  )
  system.registerInterceptor(
    'interceptor-2',
    { x: 100, y: 0, z: 0 },
    { x: 0, y: 0, z: 0 },
    { maxSpeed: 20 }
  )
  system.updateTarget('target-1', { x: 40, y: 0, z: 0 }, { x: 1, y: 0, z: 0 })
  return system
}

describe('InterceptionSystem', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('predicts target positions and trajectories deterministically', () => {
    const system = createSystem()

    expect(system.predictTargetPosition('target-1', 5)).toEqual({ x: 45, y: 0, z: 0 })
    expect(system.predictTargetTrajectory('target-1', 1, 0.5)).toEqual([
      { position: { x: 40, y: 0, z: 0 }, velocity: { x: 1, y: 0, z: 0 }, time: 0 },
      { position: { x: 40.5, y: 0, z: 0 }, velocity: { x: 1, y: 0, z: 0 }, time: 0.5 },
      { position: { x: 41, y: 0, z: 0 }, velocity: { x: 1, y: 0, z: 0 }, time: 1 },
    ])
    expect(system.predictTargetPosition('missing', 5)).toBeNull()
    expect(system.predictTargetTrajectory('missing', 1)).toEqual([])
  })

  it('creates, activates, updates, and completes missions', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    const system = createSystem()
    const mission = system.createMission('interceptor-1', 'target-1', 'LEAD')

    expect(mission).toEqual(
      expect.objectContaining({
        id: 'mission_1',
        interceptorId: 'interceptor-1',
        targetId: 'target-1',
        strategy: 'LEAD',
        status: 'PENDING',
        startTime: 1_000,
      })
    )
    expect(system.createMission('interceptor-1', 'target-1')).toBeNull()
    expect(system.activateMission(mission!.id)).toBe(true)
    expect(system.getActiveMissions().map((active) => active.id)).toEqual([mission!.id])

    system.updateInterceptor('interceptor-1', { x: 39, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })
    expect(system.updateMission(mission!.id)).toEqual(
      expect.objectContaining({ status: 'COMPLETED' })
    )
    expect(system.getAvailableInterceptors().map((interceptor) => interceptor.id)).toContain(
      'interceptor-1'
    )
  })

  it('assigns the closest viable interceptor and provides guidance commands', () => {
    const system = createSystem()

    const assignment = system.assignBestInterceptor('target-1')
    expect(assignment?.interceptorId).toBe('interceptor-1')
    expect(assignment?.result.isPossible).toBe(true)

    const mission = system.createMission(
      assignment!.interceptorId,
      'target-1',
      assignment!.result.strategy
    )
    system.activateMission(mission!.id)

    const guidance = system.getGuidanceCommand('interceptor-1')
    expect(guidance).toEqual(expect.objectContaining({ y: 0, z: 0 }))
    expect(Math.hypot(guidance!.x, guidance!.y, guidance!.z)).toBeLessThanOrEqual(20)
  })

  it('releases interceptors when missions are aborted or targets are removed', () => {
    const system = createSystem()
    const mission = system.createMission('interceptor-1', 'target-1', 'LEAD')
    system.activateMission(mission!.id)

    expect(system.getAvailableInterceptors().map((interceptor) => interceptor.id)).not.toContain(
      'interceptor-1'
    )
    system.removeTarget('target-1')

    expect(system.getMission(mission!.id)).toEqual(expect.objectContaining({ status: 'ABORTED' }))
    expect(system.getAvailableInterceptors().map((interceptor) => interceptor.id)).toContain(
      'interceptor-1'
    )
    expect(system.getGuidanceCommand('interceptor-1')).toBeNull()
  })

  it('reports impossible intercepts for missing participants and unavailable targets', () => {
    const system = createSystem()

    expect(system.calculateIntercept('missing', 'target-1')).toEqual(
      expect.objectContaining({
        isPossible: false,
        reason: 'Interceptor not found',
      })
    )
    expect(system.calculateIntercept('interceptor-1', 'missing')).toEqual(
      expect.objectContaining({
        isPossible: false,
        reason: 'Target not found',
      })
    )
    expect(system.assignBestInterceptor('missing')).toBeNull()
  })
})
