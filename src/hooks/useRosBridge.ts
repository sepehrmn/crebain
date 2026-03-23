/**
 * CREBAIN ROS Bridge React Hook
 * Adaptive Response & Awareness System (ARAS)
 *
 * React hook for managing ROS bridge connection state
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { ROSBridge, ConnectionState } from '../ros/ROSBridge'
import { ZenohBridge } from '../ros/ZenohBridge'
import { ROSPerformanceMonitor, type ConnectionQuality, type PerformanceAlert, type TopicStats } from '../ros/ROSPerformanceMonitor'
import type { ROSMessageCallback } from '../ros/types'

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface UseRosBridgeConfig {
  /** Transport layer to use */
  transport: 'websocket' | 'zenoh'
  url: string
  autoConnect: boolean
  autoReconnect: boolean
  reconnectIntervalMs: number
  maxReconnectAttempts: number
  /** Enable performance monitoring (default: true) */
  enablePerformanceMonitoring: boolean
  /** High latency threshold in ms for alerts (default: 100) */
  highLatencyThresholdMs: number
}

export interface UseRosBridgeReturn {
  state: ConnectionState
  isConnected: boolean
  error: string | null
  bridge: ROSBridge | ZenohBridge | null
  connect: () => Promise<void>
  disconnect: () => void
  subscribe: <T>(
    topic: string,
    type: string,
    callback: ROSMessageCallback<T>,
    throttleRate?: number
  ) => () => void
  publish: <T>(topic: string, msg: T) => void
  callService: <TReq, TRes>(service: string, request: TReq) => Promise<TRes>
  /** Performance monitoring data */
  performance: {
    quality: ConnectionQuality | null
    topicStats: TopicStats[]
    alerts: PerformanceAlert[]
  }
  /** Record a message receipt for performance tracking */
  recordMessage: (topic: string, sizeBytes: number, latencyMs?: number) => void
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG: UseRosBridgeConfig = {
  transport: 'websocket',
  url: 'ws://localhost:9090',
  autoConnect: false,
  autoReconnect: true,
  reconnectIntervalMs: 3000,
  maxReconnectAttempts: 10,
  enablePerformanceMonitoring: true,
  highLatencyThresholdMs: 100,
}

// ─────────────────────────────────────────────────────────────────────────────
// HOOK
// ─────────────────────────────────────────────────────────────────────────────

export function useRosBridge(
  config: Partial<UseRosBridgeConfig> = {}
): UseRosBridgeReturn {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config }
  
  const [state, setState] = useState<ConnectionState>('disconnected')
  const [error, setError] = useState<string | null>(null)
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([])
  const [quality, setQuality] = useState<ConnectionQuality | null>(null)
  const [topicStats, setTopicStats] = useState<TopicStats[]>([])
  
  const bridgeRef = useRef<ROSBridge | ZenohBridge | null>(null)
  const performanceMonitorRef = useRef<ROSPerformanceMonitor | null>(null)

  // Initialize bridge and performance monitor
  useEffect(() => {
    let bridge: ROSBridge | ZenohBridge

    if (mergedConfig.transport === 'zenoh') {
      bridge = new ZenohBridge()
      bridge.onStateChange = setState
      // Handle auto-connect for Zenoh
      if (mergedConfig.autoConnect) {
        bridge.connect().catch((err) => {
          setError(err instanceof Error ? err.message : String(err))
        })
      }
    } else {
      bridge = new ROSBridge({
        url: mergedConfig.url,
        autoReconnect: mergedConfig.autoReconnect,
        reconnectIntervalMs: mergedConfig.reconnectIntervalMs,
        maxReconnectAttempts: mergedConfig.maxReconnectAttempts,
        onStateChange: setState,
        onError: (err) => setError(err.message),
        onConnect: () => {
          setError(null)
          // Reset performance monitor on connect
          performanceMonitorRef.current?.reset()
        },
      })
      
      // Auto-connect if configured (ROSBridge handles this internally if passed in constructor? 
      // No, ROSBridge constructor doesn't take autoConnect, useRosBridge handles it)
      if (mergedConfig.autoConnect) {
        bridge.connect().catch((err) => {
          setError(err.message)
        })
      }
    }

    bridgeRef.current = bridge

    // Initialize performance monitor if enabled
    if (mergedConfig.enablePerformanceMonitoring) {
      const monitor = new ROSPerformanceMonitor({
        highLatencyThresholdMs: mergedConfig.highLatencyThresholdMs,
      })
      
      // Subscribe to alerts
      monitor.onAlert((alert) => {
        setAlerts(prev => [...prev.slice(-99), alert]) // Keep last 100 alerts
      })
      
      performanceMonitorRef.current = monitor

      // Update stats periodically
      const statsInterval = setInterval(() => {
        if (performanceMonitorRef.current) {
          setQuality(performanceMonitorRef.current.getConnectionQuality())
          setTopicStats(performanceMonitorRef.current.getAllTopicStats())
        }
      }, 1000)

      // Cleanup stats interval
      // Monkey-patch disconnect to clear interval
      const originalDisconnect = bridge.disconnect.bind(bridge)
      bridge.disconnect = async () => {
        clearInterval(statsInterval)
        await originalDisconnect()
      }
    }

    return () => {
      bridge.disconnect()
      bridgeRef.current = null
      performanceMonitorRef.current = null
    }
  }, [
    mergedConfig.transport,
    mergedConfig.url,
    mergedConfig.autoConnect,
    mergedConfig.autoReconnect,
    mergedConfig.reconnectIntervalMs,
    mergedConfig.maxReconnectAttempts,
    mergedConfig.enablePerformanceMonitoring,
    mergedConfig.highLatencyThresholdMs,
  ])

  // Connect function
  const connect = useCallback(async () => {
    if (bridgeRef.current) {
      setError(null)
      try {
        await bridgeRef.current.connect()
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    }
  }, [])

  // Disconnect function
  const disconnect = useCallback(() => {
    if (bridgeRef.current) {
      bridgeRef.current.disconnect()
    }
  }, [])

  // Subscribe function
  const subscribe = useCallback(<T>(
    topic: string,
    type: string,
    callback: ROSMessageCallback<T>,
    throttleRate?: number
  ): (() => void) => {
    if (bridgeRef.current) {
      return bridgeRef.current.subscribe(topic, type, callback, throttleRate)
    }
    return () => {}
  }, [])

  // Publish function
  const publish = useCallback(<T>(topic: string, msg: T) => {
    if (bridgeRef.current) {
      bridgeRef.current.publish(topic, msg)
    }
  }, [])

  // Call service function
  const callService = useCallback(<TReq, TRes>(
    service: string,
    request: TReq
  ): Promise<TRes> => {
    if (bridgeRef.current) {
      return bridgeRef.current.callService(service, request)
    }
    return Promise.reject(new Error('ROS bridge not connected'))
  }, [])

  // Record message for performance tracking
  const recordMessage = useCallback((topic: string, sizeBytes: number, latencyMs?: number) => {
    performanceMonitorRef.current?.recordMessage(topic, sizeBytes, latencyMs)
  }, [])

  return {
    state,
    isConnected: state === 'connected',
    error,
    bridge: bridgeRef.current,
    connect,
    disconnect,
    subscribe,
    publish,
    callService,
    performance: {
      quality,
      topicStats,
      alerts,
    },
    recordMessage,
  }
}

export default useRosBridge
