/**
 * CREBAIN Hooks Index
 * Adaptive Response & Awareness System (ARAS)
 *
 * Central export point for all custom React hooks
 */

// 2D UI Dragging
export { useDraggable } from './useDraggable'
export { useDraggablePanel, PANEL_FONT_SIZE } from './useDraggablePanel'

// 3D Object Manipulation
export { useDraggable3D } from './useDraggable3D'

// 3D Object Selection
export { useObjectSelection } from './useObjectSelection'

// Detection System
export { useDetection } from './useDetection'
export { useDetectionLoop } from './useDetectionLoop'
export { useCoreMLDetection } from './useCoreMLDetection'

// Drone System
export { useDroneController } from './useDroneController'
export { useDroneSystem } from './useDroneSystem'

// Scene Management
export { useSceneState } from './useSceneState'

// Keyboard Controls
export { useKeyboardControls } from './useKeyboardControls'

// Performance Tracking
export { usePerformanceTracker } from './usePerformanceTracker'

// ROS Integration
export { useRosBridge } from './useRosBridge'

// Gazebo Simulation
export { useGazeboSimulation } from './useGazeboSimulation'
export { useGazeboDrones } from './useGazeboDrones'

// Scheduling Utilities
export { useAnimationLoop, usePolling, useRateLimitedPolling, useDebouncedCallback } from './useScheduling'

// Tactical Console (centralized messaging)
export { useTacticalConsole } from './useTacticalConsole'
export type { MessageLevel, TacticalMessage, TacticalError } from './useTacticalConsole'

// CoreML Diagnostics (test & benchmark)
export { useCoreMLDiagnostics } from './useCoreMLDiagnostics'

// Scene Management (Three.js lifecycle)
export { useCrebainScene } from './useCrebainScene'
export type { SceneConfig, MovementConfig, MoveState } from './useCrebainScene'

// ROS Actuator Loop (rate-limited motor commands)
export { useRosActuatorLoop } from './useRosActuatorLoop'

// Surveillance Cameras (camera management, PTZ, feeds)
export { useSurveillanceCameras } from './useSurveillanceCameras'
