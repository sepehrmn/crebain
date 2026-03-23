/**
 * CREBAIN Logger
 * Centralized logging utility with level control and structured output
 * 
 * In production builds, debug/info logs are suppressed.
 * Errors are always logged but can be captured for telemetry.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export interface LogEntry {
  level: LogLevel
  module: string
  message: string
  timestamp: number
  context?: Record<string, unknown>
}

export type LogHandler = (entry: LogEntry) => void

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
}

class Logger {
  private minLevel: LogLevel = import.meta.env.DEV ? 'debug' : 'warn'
  private handlers: LogHandler[] = []
  private enabled = true

  setMinLevel(level: LogLevel): void {
    this.minLevel = level
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled
  }

  addHandler(handler: LogHandler): () => void {
    this.handlers.push(handler)
    return () => {
      const idx = this.handlers.indexOf(handler)
      if (idx >= 0) this.handlers.splice(idx, 1)
    }
  }

  private shouldLog(level: LogLevel): boolean {
    return this.enabled && LOG_LEVELS[level] >= LOG_LEVELS[this.minLevel]
  }

  private log(level: LogLevel, module: string, message: string, context?: Record<string, unknown>): void {
    if (!this.shouldLog(level)) return

    const entry: LogEntry = {
      level,
      module,
      message,
      timestamp: Date.now(),
      context,
    }

    // Dispatch to handlers
    for (const handler of this.handlers) {
      try {
        handler(entry)
      } catch {
        // Ignore handler errors
      }
    }

    // Console output with module prefix
    const prefix = `[${module}]`
    const args = context ? [prefix, message, context] : [prefix, message]

    switch (level) {
      case 'debug':
        if (import.meta.env.DEV) console.debug(...args)
        break
      case 'info':
        if (import.meta.env.DEV) console.info(...args)
        break
      case 'warn':
        console.warn(...args)
        break
      case 'error':
        console.error(...args)
        break
    }
  }

  debug(module: string, message: string, context?: Record<string, unknown>): void {
    this.log('debug', module, message, context)
  }

  info(module: string, message: string, context?: Record<string, unknown>): void {
    this.log('info', module, message, context)
  }

  warn(module: string, message: string, context?: Record<string, unknown>): void {
    this.log('warn', module, message, context)
  }

  error(module: string, message: string, context?: Record<string, unknown>): void {
    this.log('error', module, message, context)
  }

  /**
   * Create a scoped logger for a specific module
   */
  scope(module: string): ScopedLogger {
    return new ScopedLogger(this, module)
  }
}

class ScopedLogger {
  constructor(
    private logger: Logger,
    private module: string
  ) {}

  debug(message: string, context?: Record<string, unknown>): void {
    this.logger.debug(this.module, message, context)
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.logger.info(this.module, message, context)
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.logger.warn(this.module, message, context)
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.logger.error(this.module, message, context)
  }
}

export const logger = new Logger()

// Pre-scoped loggers for common modules
export const rosLogger = logger.scope('ROS')
export const gazeboLogger = logger.scope('Gazebo')
export const detectionLogger = logger.scope('Detection')
export const fusionLogger = logger.scope('Fusion')
export const sceneLogger = logger.scope('Scene')

export default logger
