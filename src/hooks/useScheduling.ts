/**
 * CREBAIN Scheduling Utilities
 * Reusable hooks for animation loops and polling
 */

import { useEffect, useRef, useCallback } from 'react'

/**
 * Hook for requestAnimationFrame-based loops with delta time
 * Automatically handles cleanup and pause/resume
 */
export function useAnimationLoop(
  callback: (deltaMs: number) => void,
  enabled: boolean = true
): void {
  const callbackRef = useRef(callback)
  const lastTimeRef = useRef<number | null>(null)
  
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled) {
      lastTimeRef.current = null
      return
    }

    let frameId: number

    const loop = (timestamp: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = timestamp
      }
      
      const deltaMs = timestamp - lastTimeRef.current
      lastTimeRef.current = timestamp
      
      callbackRef.current(deltaMs)
      frameId = requestAnimationFrame(loop)
    }

    frameId = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(frameId)
      lastTimeRef.current = null
    }
  }, [enabled])
}

/**
 * Hook for interval-based polling that waits for each cycle to complete
 * Prevents overlapping executions and adapts to actual callback duration
 */
export function usePolling(
  callback: () => void | Promise<void>,
  intervalMs: number,
  enabled: boolean = true
): void {
  const callbackRef = useRef(callback)
  
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled || intervalMs <= 0) return

    let cancelled = false

    const loop = async () => {
      while (!cancelled) {
        try {
          await callbackRef.current()
        } catch {
          // Errors should be handled in the callback
        }
        
        if (!cancelled) {
          await new Promise(resolve => setTimeout(resolve, intervalMs))
        }
      }
    }

    loop()

    return () => {
      cancelled = true
    }
  }, [enabled, intervalMs])
}

/**
 * Hook for rate-limited interval polling (fixed rate, skips if busy)
 * Good for actuator loops where timing matters more than every execution
 */
export function useRateLimitedPolling(
  callback: () => void | Promise<void>,
  rateHz: number,
  enabled: boolean = true
): void {
  const callbackRef = useRef(callback)
  const isProcessingRef = useRef(false)
  
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled || rateHz <= 0) return

    const intervalMs = 1000 / rateHz

    const tick = async () => {
      if (isProcessingRef.current) return
      
      isProcessingRef.current = true
      try {
        await callbackRef.current()
      } catch {
        // Errors should be handled in the callback
      } finally {
        isProcessingRef.current = false
      }
    }

    const intervalId = setInterval(tick, intervalMs)

    return () => {
      clearInterval(intervalId)
      isProcessingRef.current = false
    }
  }, [enabled, rateHz])
}

/**
 * Hook for debounced callbacks
 */
export function useDebouncedCallback<T extends (...args: unknown[]) => void>(
  callback: T,
  delayMs: number
): T {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const callbackRef = useRef(callback)
  
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  return useCallback((...args: Parameters<T>) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    timeoutRef.current = setTimeout(() => {
      callbackRef.current(...args)
    }, delayMs)
  }, [delayMs]) as T
}
