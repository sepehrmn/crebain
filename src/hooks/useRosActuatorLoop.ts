/**
 * CREBAIN ROS Actuator Loop Hook
 * Rate-limited motor command publishing for drone control
 */

import { useEffect, useRef } from 'react'
import { getROSBridge } from '../ros/ROSBridge'
import type { DronePhysicsWorld } from '../physics/DronePhysics'

export interface ActuatorLoopConfig {
  rateHz?: number
  maxRPM?: number
  enabled?: boolean
}

export interface UseRosActuatorLoopReturn {
  isRunning: boolean
  commandsSent: number
  lastError: string | null
}

const DEFAULT_CONFIG: Required<ActuatorLoopConfig> = {
  rateHz: 50,
  maxRPM: 1100,
  enabled: true,
}

export function useRosActuatorLoop(
  physicsWorld: DronePhysicsWorld | null,
  config: ActuatorLoopConfig = {}
): UseRosActuatorLoopReturn {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config }
  
  const isRunningRef = useRef(false)
  const commandsSentRef = useRef(0)
  const lastErrorRef = useRef<string | null>(null)
  const configRef = useRef(mergedConfig)
  configRef.current = mergedConfig

  useEffect(() => {
    if (!physicsWorld || !mergedConfig.enabled) {
      isRunningRef.current = false
      return
    }

    const intervalMs = 1000 / mergedConfig.rateHz
    let isProcessing = false

    const publishActuatorCommands = () => {
      if (isProcessing) return
      
      const bridge = getROSBridge()
      if (!bridge || !bridge.isConnected()) {
        return
      }

      isProcessing = true
      isRunningRef.current = true

      try {
        const drones = physicsWorld.getAllDrones()
        const cfg = configRef.current

        for (const drone of drones) {
          const cmds = drone.targetCommands
          const maxRPM = cfg.maxRPM

          try {
            bridge.publish(`${drone.id}/cmd/motor_speed/0`, { data: cmds.front_right * maxRPM })
            bridge.publish(`${drone.id}/cmd/motor_speed/1`, { data: cmds.front_left * maxRPM })
            bridge.publish(`${drone.id}/cmd/motor_speed/2`, { data: cmds.rear_left * maxRPM })
            bridge.publish(`${drone.id}/cmd/motor_speed/3`, { data: cmds.rear_right * maxRPM })
            commandsSentRef.current += 4
          } catch (err) {
            lastErrorRef.current = err instanceof Error ? err.message : String(err)
          }
        }
      } catch (err) {
        lastErrorRef.current = err instanceof Error ? err.message : String(err)
      } finally {
        isProcessing = false
      }
    }

    const intervalId = setInterval(publishActuatorCommands, intervalMs)

    return () => {
      clearInterval(intervalId)
      isRunningRef.current = false
    }
  }, [physicsWorld, mergedConfig.enabled, mergedConfig.rateHz])

  return {
    isRunning: isRunningRef.current,
    commandsSent: commandsSentRef.current,
    lastError: lastErrorRef.current,
  }
}

export default useRosActuatorLoop
