/**
 * CREBAIN Performance Tracker Hook
 * Adaptive Response & Awareness System (ARAS)
 *
 * Tracks performance history for real-time monitoring
 */

import { useState, useCallback, useRef } from 'react'
import type { PerformanceData } from '../components/PerformancePanel'

interface UsePerformanceTrackerOptions {
  maxHistory?: number
}

interface UsePerformanceTrackerReturn {
  /** Current performance data */
  currentData: PerformanceData | null
  /** History of performance data */
  history: PerformanceData[]
  /** Record a new performance sample */
  recordSample: (data: Omit<PerformanceData, 'timestamp'>) => void
  /** Clear all history */
  clearHistory: () => void
  /** Get average inference time */
  getAverageInferenceTime: () => number
  /** Get total samples recorded */
  totalSamples: number
}

const DEFAULT_MAX_HISTORY = 100

/**
 * Hook to track performance history for the PerformancePanel
 */
export function usePerformanceTracker(
  options: UsePerformanceTrackerOptions = {}
): UsePerformanceTrackerReturn {
  const { maxHistory = DEFAULT_MAX_HISTORY } = options
  
  const [currentData, setCurrentData] = useState<PerformanceData | null>(null)
  const [history, setHistory] = useState<PerformanceData[]>([])
  const totalSamplesRef = useRef(0)

  const recordSample = useCallback((data: Omit<PerformanceData, 'timestamp'>) => {
    const sample: PerformanceData = {
      ...data,
      timestamp: Date.now(),
    }

    setCurrentData(sample)
    totalSamplesRef.current += 1

    setHistory(prev => {
      const newHistory = [...prev, sample]
      // Keep only the last maxHistory samples
      if (newHistory.length > maxHistory) {
        return newHistory.slice(-maxHistory)
      }
      return newHistory
    })
  }, [maxHistory])

  const clearHistory = useCallback(() => {
    setHistory([])
    setCurrentData(null)
    totalSamplesRef.current = 0
  }, [])

  const getAverageInferenceTime = useCallback(() => {
    if (history.length === 0) return 0
    const sum = history.reduce((acc, h) => acc + h.inferenceTimeMs, 0)
    return sum / history.length
  }, [history])

  return {
    currentData,
    history,
    recordSample,
    clearHistory,
    getAverageInferenceTime,
    totalSamples: totalSamplesRef.current,
  }
}

export default usePerformanceTracker
