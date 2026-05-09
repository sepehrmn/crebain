import { beforeEach, describe, expect, it, vi } from 'vitest'

const invokeMock = vi.hoisted(() => vi.fn())

vi.mock('@tauri-apps/api/core', () => ({
  invoke: invokeMock,
}))

import { SceneStateManager } from '../SceneState'

describe('SceneStateManager filesystem IPC', () => {
  beforeEach(() => {
    invokeMock.mockReset()
    localStorage.clear()
  })

  it('saves current scene state through the Tauri filesystem command', async () => {
    invokeMock.mockResolvedValue(undefined)
    const manager = new SceneStateManager()
    manager.createNew('IPC Scene')

    await manager.saveToFileSystem('/tmp/ipc-scene.json')

    expect(invokeMock).toHaveBeenCalledWith('scene_save_file', {
      path: '/tmp/ipc-scene.json',
      json: expect.stringContaining('"name": "IPC Scene"'),
    })
  })

  it('does not call IPC when no scene state exists', async () => {
    const manager = new SceneStateManager()

    await manager.saveToFileSystem('/tmp/empty.json')

    expect(invokeMock).not.toHaveBeenCalled()
  })

  it('falls back to browser file save using the path basename when IPC save fails', async () => {
    invokeMock.mockRejectedValue(new Error('not in tauri'))
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const manager = new SceneStateManager()
    const saveToFile = vi.spyOn(manager, 'saveToFile').mockImplementation(() => undefined)
    manager.createNew('Fallback Scene')

    try {
      await manager.saveToFileSystem('/tmp/fallback-scene.json')
    } finally {
      consoleWarn.mockRestore()
    }

    expect(saveToFile).toHaveBeenCalledWith('fallback-scene.json')
  })

  it('loads scene state through the Tauri filesystem command', async () => {
    const json = JSON.stringify({
      version: '1.0.0',
      timestamp: 123,
      name: 'Loaded Scene',
      cameras: [],
      drones: [],
      recentDetections: [],
      settings: {
        detectionEnabled: true,
        showDetectionPanel: true,
        showPerformancePanel: true,
        renderQuality: 'high',
        physicsEnabled: true,
        sensorSimulationEnabled: true,
      },
      viewCamera: {
        position: { x: 0, y: 5, z: 10 },
        target: { x: 0, y: 0, z: 0 },
      },
    })
    invokeMock.mockResolvedValue(json)
    const manager = new SceneStateManager()

    const state = await manager.loadFromFileSystem('/tmp/loaded-scene.json')

    expect(invokeMock).toHaveBeenCalledWith('scene_load_file', { path: '/tmp/loaded-scene.json' })
    expect(state?.name).toBe('Loaded Scene')
    expect(manager.getState()?.name).toBe('Loaded Scene')
  })

  it('returns null when IPC load fails', async () => {
    invokeMock.mockRejectedValue(new Error('missing file'))
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const manager = new SceneStateManager()

    let state
    try {
      state = await manager.loadFromFileSystem('/tmp/missing.json')
    } finally {
      consoleWarn.mockRestore()
    }

    expect(state).toBeNull()
  })
})
