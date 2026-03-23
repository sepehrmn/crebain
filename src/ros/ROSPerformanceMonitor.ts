/**
 * CREBAIN ROS Performance Monitor
 * Adaptive Response & Awareness System (ARAS)
 *
 * Tracks message latency, throughput, and connection quality
 * Provides automatic degradation detection
 */

import { CircularBuffer } from '../lib/CircularBuffer'
import { rosLogger as log } from '../lib/logger'

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface LatencySample {
  topic: string
  latencyMs: number
  timestamp: number
}

export interface ThroughputSample {
  topic: string
  messagesPerSecond: number
  bytesPerSecond: number
  timestamp: number
}

export interface TopicStats {
  topic: string
  messageCount: number
  byteCount: number
  lastReceived: number
  avgLatencyMs: number
  minLatencyMs: number
  maxLatencyMs: number
  p95LatencyMs: number
  messagesPerSecond: number
  bytesPerSecond: number
}

export interface ConnectionQuality {
  /** Overall quality score 0-100 */
  score: number
  /** Quality level */
  level: 'excellent' | 'good' | 'fair' | 'poor' | 'critical'
  /** Average latency across all topics */
  avgLatencyMs: number
  /** Total message throughput */
  totalMessagesPerSecond: number
  /** Number of dropped messages (estimated) */
  droppedMessages: number
  /** Connection uptime in seconds */
  uptimeSeconds: number
}

export interface PerformanceAlert {
  type: 'high_latency' | 'low_throughput' | 'message_gap' | 'connection_degraded'
  topic?: string
  message: string
  severity: 'warning' | 'error'
  timestamp: number
}

export interface PerformanceConfig {
  /** Window size for rolling statistics in ms (default: 5000) */
  windowSizeMs: number
  /** High latency threshold in ms (default: 100) */
  highLatencyThresholdMs: number
  /** Message gap threshold in ms (default: 1000) */
  messageGapThresholdMs: number
  /** Minimum expected messages per second (default: 1) */
  minMessagesPerSecond: number
  /** Maximum samples to keep per topic (default: 1000) */
  maxSamplesPerTopic: number
}

export type AlertCallback = (alert: PerformanceAlert) => void

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG: PerformanceConfig = {
  windowSizeMs: 5000,
  highLatencyThresholdMs: 100,
  messageGapThresholdMs: 1000,
  minMessagesPerSecond: 1,
  maxSamplesPerTopic: 1000,
}

// ─────────────────────────────────────────────────────────────────────────────
// PERFORMANCE MONITOR
// ─────────────────────────────────────────────────────────────────────────────

export class ROSPerformanceMonitor {
  private config: PerformanceConfig
  private topicLatencies: Map<string, CircularBuffer<LatencySample>> = new Map()
  private topicMessageCounts: Map<string, number> = new Map()
  private topicByteCounts: Map<string, number> = new Map()
  private topicLastReceived: Map<string, number> = new Map()
  private alertCallbacks: Set<AlertCallback> = new Set()
  private startTime: number = Date.now()
  private droppedMessages: number = 0
  private updateIntervalId: ReturnType<typeof setInterval> | null = null

  constructor(config: Partial<PerformanceConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // LIFECYCLE
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Start the performance monitor
   */
  start(): void {
    this.startTime = Date.now()

    // Start periodic stats calculation and alert checking
    this.updateIntervalId = setInterval(() => {
      this.checkForAlerts()
    }, 1000)
  }

  /**
   * Stop the performance monitor
   */
  stop(): void {
    if (this.updateIntervalId) {
      clearInterval(this.updateIntervalId)
      this.updateIntervalId = null
    }
  }

  /**
   * Reset all statistics
   */
  reset(): void {
    this.topicLatencies.clear()
    this.topicMessageCounts.clear()
    this.topicByteCounts.clear()
    this.topicLastReceived.clear()
    this.droppedMessages = 0
    this.startTime = Date.now()
  }

  // ───────────────────────────────────────────────────────────────────────────
  // DATA RECORDING
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Record a received message
   */
  recordMessage(topic: string, messageSize: number, sentTimestamp?: number): void {
    const now = Date.now()

    // Update message count
    const count = this.topicMessageCounts.get(topic) || 0
    this.topicMessageCounts.set(topic, count + 1)

    // Update byte count
    const bytes = this.topicByteCounts.get(topic) || 0
    this.topicByteCounts.set(topic, bytes + messageSize)

    // Check for message gap
    const lastReceived = this.topicLastReceived.get(topic)
    if (lastReceived && (now - lastReceived) > this.config.messageGapThresholdMs) {
      this.droppedMessages++
      this.emitAlert({
        type: 'message_gap',
        topic,
        message: `Message gap of ${now - lastReceived}ms detected on ${topic}`,
        severity: 'warning',
        timestamp: now,
      })
    }
    this.topicLastReceived.set(topic, now)

    // Record latency if timestamp provided
    if (sentTimestamp) {
      const latencyMs = now - sentTimestamp

      let buffer = this.topicLatencies.get(topic)
      if (!buffer) {
        buffer = new CircularBuffer<LatencySample>(this.config.maxSamplesPerTopic)
        this.topicLatencies.set(topic, buffer)
      }

      buffer.push({
        topic,
        latencyMs,
        timestamp: now,
      })

      // Check for high latency
      if (latencyMs > this.config.highLatencyThresholdMs) {
        this.emitAlert({
          type: 'high_latency',
          topic,
          message: `High latency ${latencyMs.toFixed(1)}ms on ${topic}`,
          severity: latencyMs > this.config.highLatencyThresholdMs * 2 ? 'error' : 'warning',
          timestamp: now,
        })
      }
    }
  }

  /**
   * Record a latency sample directly
   */
  recordLatency(topic: string, latencyMs: number): void {
    const now = Date.now()

    let buffer = this.topicLatencies.get(topic)
    if (!buffer) {
      buffer = new CircularBuffer<LatencySample>(this.config.maxSamplesPerTopic)
      this.topicLatencies.set(topic, buffer)
    }

    buffer.push({
      topic,
      latencyMs,
      timestamp: now,
    })
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STATISTICS
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Get statistics for a specific topic
   */
  getTopicStats(topic: string): TopicStats | null {
    const messageCount = this.topicMessageCounts.get(topic) || 0
    const byteCount = this.topicByteCounts.get(topic) || 0
    const lastReceived = this.topicLastReceived.get(topic) || 0
    const latencyBuffer = this.topicLatencies.get(topic)

    if (messageCount === 0) return null

    const uptimeSeconds = (Date.now() - this.startTime) / 1000

    // Calculate latency stats
    let avgLatencyMs = 0
    let minLatencyMs = Infinity
    let maxLatencyMs = 0
    let p95LatencyMs = 0

    if (latencyBuffer && latencyBuffer.length > 0) {
      const latencies: number[] = []
      latencyBuffer.forEach(sample => {
        latencies.push(sample.latencyMs)
      })

      latencies.sort((a, b) => a - b)
      minLatencyMs = latencies[0]
      maxLatencyMs = latencies[latencies.length - 1]
      avgLatencyMs = latencies.reduce((a, b) => a + b, 0) / latencies.length
      p95LatencyMs = latencies[Math.floor(latencies.length * 0.95)] || maxLatencyMs
    }

    return {
      topic,
      messageCount,
      byteCount,
      lastReceived,
      avgLatencyMs,
      minLatencyMs: minLatencyMs === Infinity ? 0 : minLatencyMs,
      maxLatencyMs,
      p95LatencyMs,
      messagesPerSecond: messageCount / uptimeSeconds,
      bytesPerSecond: byteCount / uptimeSeconds,
    }
  }

  /**
   * Get statistics for all topics
   */
  getAllTopicStats(): TopicStats[] {
    const topics = new Set<string>([
      ...this.topicMessageCounts.keys(),
      ...this.topicLatencies.keys(),
    ])

    const stats: TopicStats[] = []
    for (const topic of topics) {
      const topicStats = this.getTopicStats(topic)
      if (topicStats) {
        stats.push(topicStats)
      }
    }

    return stats
  }

  /**
   * Get overall connection quality
   */
  getConnectionQuality(): ConnectionQuality {
    const stats = this.getAllTopicStats()
    const uptimeSeconds = (Date.now() - this.startTime) / 1000

    if (stats.length === 0) {
      return {
        score: 0,
        level: 'critical',
        avgLatencyMs: 0,
        totalMessagesPerSecond: 0,
        droppedMessages: this.droppedMessages,
        uptimeSeconds,
      }
    }

    // Calculate averages
    const avgLatencyMs = stats.reduce((sum, s) => sum + s.avgLatencyMs, 0) / stats.length
    const totalMessagesPerSecond = stats.reduce((sum, s) => sum + s.messagesPerSecond, 0)

    // Calculate quality score (0-100)
    let score = 100

    // Latency penalty (up to -40 points)
    if (avgLatencyMs > 10) {
      score -= Math.min(40, (avgLatencyMs - 10) / 2)
    }

    // Throughput penalty (up to -30 points)
    const expectedMps = this.config.minMessagesPerSecond * stats.length
    if (totalMessagesPerSecond < expectedMps) {
      score -= Math.min(30, (1 - totalMessagesPerSecond / expectedMps) * 30)
    }

    // Dropped message penalty (up to -30 points)
    if (this.droppedMessages > 0) {
      const totalMessages = stats.reduce((sum, s) => sum + s.messageCount, 0)
      const dropRate = this.droppedMessages / (totalMessages + this.droppedMessages)
      score -= Math.min(30, dropRate * 100)
    }

    score = Math.max(0, Math.round(score))

    // Determine level
    let level: ConnectionQuality['level']
    if (score >= 90) level = 'excellent'
    else if (score >= 70) level = 'good'
    else if (score >= 50) level = 'fair'
    else if (score >= 25) level = 'poor'
    else level = 'critical'

    return {
      score,
      level,
      avgLatencyMs,
      totalMessagesPerSecond,
      droppedMessages: this.droppedMessages,
      uptimeSeconds,
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // ALERTS
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Register an alert callback
   */
  onAlert(callback: AlertCallback): () => void {
    this.alertCallbacks.add(callback)
    return () => this.alertCallbacks.delete(callback)
  }

  private emitAlert(alert: PerformanceAlert): void {
    for (const callback of this.alertCallbacks) {
      try {
        callback(alert)
      } catch (error) {
        log.error('Alert callback error', { error })
      }
    }
  }

  private checkForAlerts(): void {
    const now = Date.now()
    const quality = this.getConnectionQuality()

    // Check for degraded connection
    if (quality.level === 'poor' || quality.level === 'critical') {
      this.emitAlert({
        type: 'connection_degraded',
        message: `Connection quality ${quality.level}: score ${quality.score}/100`,
        severity: quality.level === 'critical' ? 'error' : 'warning',
        timestamp: now,
      })
    }

    // Check for low throughput on individual topics
    for (const [topic, lastReceived] of this.topicLastReceived) {
      if ((now - lastReceived) > this.config.windowSizeMs) {
        this.emitAlert({
          type: 'low_throughput',
          topic,
          message: `No messages received on ${topic} for ${((now - lastReceived) / 1000).toFixed(1)}s`,
          severity: 'warning',
          timestamp: now,
        })
      }
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // ACCESSORS
  // ───────────────────────────────────────────────────────────────────────────

  getConfig(): Readonly<PerformanceConfig> {
    return this.config
  }

  setConfig(config: Partial<PerformanceConfig>): void {
    this.config = { ...this.config, ...config }
  }

  getUptimeSeconds(): number {
    return (Date.now() - this.startTime) / 1000
  }

  getDroppedMessageCount(): number {
    return this.droppedMessages
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FACTORY
// ─────────────────────────────────────────────────────────────────────────────

let instance: ROSPerformanceMonitor | null = null

export function getPerformanceMonitor(): ROSPerformanceMonitor {
  if (!instance) {
    instance = new ROSPerformanceMonitor()
  }
  return instance
}

export function createPerformanceMonitor(
  config?: Partial<PerformanceConfig>
): ROSPerformanceMonitor {
  return new ROSPerformanceMonitor(config)
}
