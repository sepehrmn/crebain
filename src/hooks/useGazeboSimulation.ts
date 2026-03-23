/**
 * CREBAIN Gazebo Simulation Hook
 * Adaptive Response & Awareness System (ARAS)
 *
 * Combined hook for ROS-Gazebo drone simulation with interception
 * Features continuous 20Hz guidance control loop
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useRosBridge } from './useRosBridge'
import { useGazeboDrones, DroneState } from './useGazeboDrones'
import {
  InterceptionSystem,
  InterceptionMission,
  InterceptionStrategy,
  TrajectoryPoint,
  getInterceptionSystem,
} from '../simulation/InterceptionSystem'
import {
  GuidanceController,
  createGuidanceController,
  type GuidanceCommand,
} from '../ros/GuidanceController'
import { getGazeboController } from '../ros/GazeboController'
import type { ConnectionState } from '../ros/ROSBridge'
import { logger } from '../lib/logger'

const log = logger.scope('GazeboSim')

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface UseGazeboSimulationConfig {
  /** Transport layer to use (default: zenoh) */
  transport: 'websocket' | 'zenoh'
  rosUrl: string
  autoConnect: boolean
  updateIntervalMs: number
  trajectoryDurationSec: number
  /** Enable continuous guidance control (default: true) */
  enableContinuousGuidance: boolean
  /** Guidance control rate in Hz (default: 20) */
  guidanceRateHz: number
}

export interface UseGazeboSimulationReturn {
  // Connection
  connectionState: ConnectionState
  transport: 'websocket' | 'zenoh'
  setTransport: (transport: 'websocket' | 'zenoh') => void
  rosUrl: string
  setRosUrl: (url: string) => void
  connect: () => Promise<void>
  disconnect: () => void
  connectionError: string | null

  // Drones
  allDrones: DroneState[]
  friendlyDrones: DroneState[]
  hostileDrones: DroneState[]
  getDrone: (id: string) => DroneState | undefined

  // Interception
  activeMissions: InterceptionMission[]
  trajectoryPredictions: Map<string, TrajectoryPoint[]>
  initiateIntercept: (targetId: string, strategy?: InterceptionStrategy) => InterceptionMission | null
  abortMission: (missionId: string) => boolean

  // Guidance
  guidanceControllers: Map<string, GuidanceController>
  lastGuidanceCommands: Map<string, GuidanceCommand>

  // Simulation control
  isSimulationActive: boolean
  toggleSimulation: () => void
  emergencyStopAll: () => void
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG: UseGazeboSimulationConfig = {
  transport: 'zenoh',
  rosUrl: 'ws://localhost:9090',
  autoConnect: false,
  updateIntervalMs: 100,
  trajectoryDurationSec: 10,
  enableContinuousGuidance: true,
  guidanceRateHz: 20,
}

// ─────────────────────────────────────────────────────────────────────────────
// HOOK
// ─────────────────────────────────────────────────────────────────────────────

export function useGazeboSimulation(
  config: Partial<UseGazeboSimulationConfig> = {}
): UseGazeboSimulationReturn {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config }

  // State
  const [transport, setTransport] = useState<'websocket' | 'zenoh'>(mergedConfig.transport)
  const [rosUrl, setRosUrl] = useState(mergedConfig.rosUrl)
  const [isSimulationActive, setIsSimulationActive] = useState(true)
  const [activeMissions, setActiveMissions] = useState<InterceptionMission[]>([])
  const [trajectoryPredictions, setTrajectoryPredictions] = useState<Map<string, TrajectoryPoint[]>>(new Map())
  const [lastGuidanceCommands, setLastGuidanceCommands] = useState<Map<string, GuidanceCommand>>(new Map())

  // Refs
  const interceptionSystemRef = useRef<InterceptionSystem>(getInterceptionSystem())
  const updateIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const guidanceControllersRef = useRef<Map<string, GuidanceController>>(new Map())
  const configRef = useRef(mergedConfig)
  configRef.current = mergedConfig

  // ROS Bridge
  const rosBridge = useRosBridge({
    transport,
    url: rosUrl,
    autoConnect: mergedConfig.autoConnect,
  })

  // Gazebo Drones
  const gazeboDrones = useGazeboDrones({
    bridge: rosBridge.bridge,
  })

  // Connect GazeboController singleton
  useEffect(() => {
    const controller = getGazeboController()
    if (rosBridge.bridge && rosBridge.isConnected) {
      controller.connect(rosBridge.bridge)
    } else {
      controller.disconnect()
    }
  }, [rosBridge.bridge, rosBridge.isConnected])

  // Memoized guidance controllers map for external access
  const guidanceControllers = useMemo(() => {
    return new Map(guidanceControllersRef.current)
  }, [activeMissions]) // Update when missions change

  // Sync drones with interception system
  useEffect(() => {
    const system = interceptionSystemRef.current

    // Update targets (hostile drones)
    for (const drone of gazeboDrones.hostileDrones) {
      system.updateTarget(
        drone.id,
        drone.pose.position,
        drone.velocity.linear
      )
    }

    // Update interceptors (friendly drones)
    for (const drone of gazeboDrones.friendlyDrones) {
      const existingInterceptor = system.getInterceptor(drone.id)
      if (existingInterceptor) {
        system.updateInterceptor(
          drone.id,
          drone.pose.position,
          drone.velocity.linear
        )
      } else {
        system.registerInterceptor(
          drone.id,
          drone.pose.position,
          drone.velocity.linear
        )
      }
    }
  }, [gazeboDrones.hostileDrones, gazeboDrones.friendlyDrones])

  // Manage guidance controllers for active missions
  useEffect(() => {
    const cfg = configRef.current
    
    // Early return if not connected, but still set up cleanup
    if (!rosBridge.bridge || !rosBridge.isConnected || !cfg.enableContinuousGuidance) {
      return () => {
        for (const controller of guidanceControllersRef.current.values()) {
          controller.stop()
        }
        guidanceControllersRef.current.clear()
      }
    }

    // Get current active mission interceptor IDs
    const activeInterceptorIds = new Set(
      activeMissions
        .filter(m => m.status === 'ACTIVE')
        .map(m => m.interceptorId)
    )

    // Create controllers for new missions
    for (const mission of activeMissions) {
      if (mission.status !== 'ACTIVE') continue

      const interceptorId = mission.interceptorId
      if (!guidanceControllersRef.current.has(interceptorId)) {
        const controller = createGuidanceController({
          rateHz: cfg.guidanceRateHz,
        })

        // Set up command callback for state updates
        controller.onCommand((cmd) => {
          setLastGuidanceCommands(prev => {
            const updated = new Map(prev)
            updated.set(interceptorId, cmd)
            return updated
          })
        })

        // Start the controller with validated bridge
        const bridge = rosBridge.bridge
        if (bridge) {
          controller.start(bridge, interceptorId)
          guidanceControllersRef.current.set(interceptorId, controller)
        }
      }
    }

    // Stop controllers for completed/aborted missions
    for (const [interceptorId, controller] of guidanceControllersRef.current) {
      if (!activeInterceptorIds.has(interceptorId)) {
        controller.stop()
        guidanceControllersRef.current.delete(interceptorId)
      }
    }

    // Cleanup on unmount
    return () => {
      for (const controller of guidanceControllersRef.current.values()) {
        controller.stop()
      }
      guidanceControllersRef.current.clear()
    }
  }, [activeMissions, rosBridge.bridge, rosBridge.isConnected])

  // Update guidance controllers with latest interception data
  useEffect(() => {
    const system = interceptionSystemRef.current

    for (const mission of activeMissions) {
      if (mission.status !== 'ACTIVE') continue

      const controller = guidanceControllersRef.current.get(mission.interceptorId)
      if (!controller) continue

      // Get guidance command from interception system
      const guidance = system.getGuidanceCommand(mission.interceptorId)
      if (guidance) {
        // Use direct velocity command for now
        // Could be enhanced with target position + velocity for full PD control
        controller.setDirectVelocity(guidance)
      }

      // Update controller with current drone position for PD feedback
      const drone = gazeboDrones.getDrone(mission.interceptorId)
      if (drone) {
        controller.updateCurrentPosition(drone.pose.position, drone.velocity.linear)
      }
    }
  }, [activeMissions, gazeboDrones])

  // Update loop for missions and trajectories
  useEffect(() => {
    if (!isSimulationActive) {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current)
        updateIntervalRef.current = null
      }
      return
    }

    const updateSimulation = () => {
      const system = interceptionSystemRef.current
      const cfg = configRef.current

      // Update active missions
      const missions = system.getActiveMissions()
      for (const mission of missions) {
        system.updateMission(mission.id)
      }
      setActiveMissions([...system.getActiveMissions()])

      // Selective trajectory prediction - only for active mission targets
      // This reduces computation by ~80% compared to predicting all hostiles
      const activeTargetIds = new Set(
        missions.filter(m => m.status === 'ACTIVE').map(m => m.targetId)
      )

      const newPredictions = new Map<string, TrajectoryPoint[]>()
      for (const targetId of activeTargetIds) {
        const trajectory = system.predictTargetTrajectory(
          targetId,
          cfg.trajectoryDurationSec
        )
        if (trajectory.length > 0) {
          newPredictions.set(targetId, trajectory)
        }
      }
      setTrajectoryPredictions(newPredictions)
    }

    updateIntervalRef.current = setInterval(updateSimulation, configRef.current.updateIntervalMs)

    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current)
        updateIntervalRef.current = null
      }
    }
  }, [isSimulationActive])

  // Initiate intercept
  const initiateIntercept = useCallback((
    targetId: string,
    strategy: InterceptionStrategy = 'LEAD'
  ): InterceptionMission | null => {
    const system = interceptionSystemRef.current

    // Find best available interceptor
    const assignment = system.assignBestInterceptor(targetId)
    if (!assignment) {
      log.warn('No available interceptor for target', { targetId })
      return null
    }

    // Create and activate mission
    const mission = system.createMission(
      assignment.interceptorId,
      targetId,
      strategy
    )

    if (mission) {
      system.activateMission(mission.id)
      setActiveMissions([...system.getActiveMissions()])
      // Guidance controller will be created automatically by the effect
      // when activeMissions state updates
    }

    return mission
  }, [])

  // Abort mission
  const abortMission = useCallback((missionId: string): boolean => {
    const system = interceptionSystemRef.current
    const success = system.abortMission(missionId)
    if (success) {
      setActiveMissions([...system.getActiveMissions()])
    }
    return success
  }, [])

  // Toggle simulation
  const toggleSimulation = useCallback(() => {
    setIsSimulationActive(prev => !prev)
  }, [])

  // Emergency stop all guidance controllers
  const emergencyStopAll = useCallback(() => {
    for (const controller of guidanceControllersRef.current.values()) {
      controller.emergencyStop()
    }
  }, [])

  return {
    // Connection
    connectionState: rosBridge.state,
    transport,
    setTransport,
    rosUrl,
    setRosUrl,
    connect: rosBridge.connect,
    disconnect: rosBridge.disconnect,
    connectionError: rosBridge.error,

    // Drones
    allDrones: Array.from(gazeboDrones.drones.values()),
    friendlyDrones: gazeboDrones.friendlyDrones,
    hostileDrones: gazeboDrones.hostileDrones,
    getDrone: gazeboDrones.getDrone,

    // Interception
    activeMissions,
    trajectoryPredictions,
    initiateIntercept,
    abortMission,

    // Guidance
    guidanceControllers,
    lastGuidanceCommands,

    // Simulation control
    isSimulationActive,
    toggleSimulation,
    emergencyStopAll,
  }
}

export default useGazeboSimulation
