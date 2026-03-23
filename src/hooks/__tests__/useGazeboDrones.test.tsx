import { describe, it, expect, vi } from 'vitest'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { useGazeboDrones } from '../useGazeboDrones'

;(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true

function Harness({ bridge, tick }: { bridge: any; tick: number }) {
  // tick forces re-render when we change the bridge's internal connection state
  void tick
  useGazeboDrones({ bridge })
  return null
}

describe('useGazeboDrones', () => {
  it('subscribes when the bridge becomes connected', async () => {
    let connected = false
    const unsubscribe = vi.fn()

    const bridge = {
      isConnected: () => connected,
      subscribeToModelStates: vi.fn(() => unsubscribe),
    }

    const container = document.createElement('div')
    const root = createRoot(container)

    await act(async () => {
      root.render(<Harness bridge={bridge} tick={0} />)
    })

    expect(bridge.subscribeToModelStates).not.toHaveBeenCalled()

    connected = true
    await act(async () => {
      root.render(<Harness bridge={bridge} tick={1} />)
    })

    expect(bridge.subscribeToModelStates).toHaveBeenCalledTimes(1)

    connected = false
    await act(async () => {
      root.render(<Harness bridge={bridge} tick={2} />)
    })

    expect(unsubscribe).toHaveBeenCalledTimes(1)

    await act(async () => {
      root.unmount()
    })
  })
})
