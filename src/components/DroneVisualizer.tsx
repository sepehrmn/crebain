/**
 * CREBAIN Drone Visualizer
 * Adaptive Response & Awareness System (ARAS)
 *
 * Three.js visualization of drones from ROS-Gazebo simulation
 */

import { useEffect, useRef, useCallback } from 'react'
import * as THREE from 'three'
import type { DroneState, DroneType } from '../hooks/useGazeboDrones'
import type { InterceptionMission, TrajectoryPoint } from '../simulation/InterceptionSystem'
import { THREAT_LEVEL_COLORS } from '../detection/types'

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface DroneVisualizerProps {
  scene: THREE.Scene
  drones: Map<string, DroneState>
  activeMissions: InterceptionMission[]
  trajectoryPredictions: Map<string, TrajectoryPoint[]>
  showTrajectories: boolean
  showLabels: boolean
  scale: number
}

interface DroneVisual {
  group: THREE.Group
  body: THREE.Mesh
  rotors: THREE.Mesh[]
  trail: THREE.Line
  label: THREE.Sprite
  trajectoryLine: THREE.Line | null
  interceptLine: THREE.Line | null
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const TRAIL_LENGTH = 50
const ROTOR_SPEED = 0.5 // rotations per frame

// Map drone types to threat levels for consistent coloring
const DRONE_TYPE_TO_THREAT: Record<DroneType, number> = {
  friendly: 1, // Green
  hostile: 4,  // Red
  unknown: 3,  // Amber
}

function hexStringToNumber(hex: string): number {
  return parseInt(hex.replace('#', ''), 16)
}

function getDroneColor(type: DroneType): number {
  const threatLevel = DRONE_TYPE_TO_THREAT[type]
  return hexStringToNumber(THREAT_LEVEL_COLORS[threatLevel as keyof typeof THREAT_LEVEL_COLORS])
}

// ─────────────────────────────────────────────────────────────────────────────
// GEOMETRY CREATION
// ─────────────────────────────────────────────────────────────────────────────

function createDroneGeometry(type: DroneType, scale: number): THREE.Group {
  const group = new THREE.Group()
  const color = getDroneColor(type)

  // Drone body (simplified quadcopter shape)
  const bodyGeometry = new THREE.BoxGeometry(0.3 * scale, 0.1 * scale, 0.3 * scale)
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.7,
    roughness: 0.3,
    emissive: color,
    emissiveIntensity: 0.3,
  })
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial)
  group.add(body)

  // Arms
  const armGeometry = new THREE.BoxGeometry(0.5 * scale, 0.02 * scale, 0.02 * scale)
  const armMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 })

  const arm1 = new THREE.Mesh(armGeometry, armMaterial)
  arm1.rotation.y = Math.PI / 4
  group.add(arm1)

  const arm2 = new THREE.Mesh(armGeometry, armMaterial)
  arm2.rotation.y = -Math.PI / 4
  group.add(arm2)

  // Rotors
  const rotorGeometry = new THREE.CylinderGeometry(0.08 * scale, 0.08 * scale, 0.01 * scale, 8)
  const rotorMaterial = new THREE.MeshStandardMaterial({
    color: 0x666666,
    transparent: true,
    opacity: 0.7,
  })

  const rotorPositions = [
    { x: 0.15 * scale, z: 0.15 * scale },
    { x: -0.15 * scale, z: 0.15 * scale },
    { x: 0.15 * scale, z: -0.15 * scale },
    { x: -0.15 * scale, z: -0.15 * scale },
  ]

  for (const pos of rotorPositions) {
    const rotor = new THREE.Mesh(rotorGeometry, rotorMaterial)
    rotor.position.set(pos.x, 0.05 * scale, pos.z)
    rotor.userData.isRotor = true
    group.add(rotor)
  }

  return group
}

function createTrailLine(color: number): THREE.Line {
  const geometry = new THREE.BufferGeometry()
  const positions = new Float32Array(TRAIL_LENGTH * 3)
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.5,
  })

  return new THREE.Line(geometry, material)
}

function createLabel(text: string, color: number): THREE.Sprite {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  canvas.width = 256
  canvas.height = 64

  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  ctx.strokeStyle = `#${color.toString(16).padStart(6, '0')}`
  ctx.lineWidth = 2
  ctx.strokeRect(0, 0, canvas.width, canvas.height)

  ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`
  ctx.font = 'bold 24px monospace'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, canvas.width / 2, canvas.height / 2)

  const texture = new THREE.CanvasTexture(canvas)
  const material = new THREE.SpriteMaterial({ map: texture })
  const sprite = new THREE.Sprite(material)
  sprite.scale.set(2, 0.5, 1)

  return sprite
}

function createTrajectoryLine(color: number): THREE.Line {
  const geometry = new THREE.BufferGeometry()
  const material = new THREE.LineDashedMaterial({
    color,
    dashSize: 0.5,
    gapSize: 0.2,
    transparent: true,
    opacity: 0.6,
  })

  const line = new THREE.Line(geometry, material)
  line.computeLineDistances()
  return line
}

function createInterceptLine(): THREE.Line {
  const geometry = new THREE.BufferGeometry()
  const material = new THREE.LineDashedMaterial({
    color: 0xff00ff, // Magenta for intercept
    dashSize: 0.3,
    gapSize: 0.15,
    transparent: true,
    opacity: 0.8,
  })

  const line = new THREE.Line(geometry, material)
  line.computeLineDistances()
  return line
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export function DroneVisualizer({
  scene,
  drones,
  activeMissions,
  trajectoryPredictions,
  showTrajectories,
  showLabels,
  scale,
}: DroneVisualizerProps): null {
  const visualsRef = useRef<Map<string, DroneVisual>>(new Map())
  const frameRef = useRef(0)

  // Create or update drone visuals
  const updateDroneVisual = useCallback((drone: DroneState) => {
    let visual = visualsRef.current.get(drone.id)

    if (!visual) {
      // Create new visual
      const group = createDroneGeometry(drone.type, scale)
      const trail = createTrailLine(getDroneColor(drone.type))
      const label = createLabel(drone.name, getDroneColor(drone.type))

      scene.add(group)
      scene.add(trail)
      scene.add(label)

      visual = {
        group,
        body: group.children[0] as THREE.Mesh,
        rotors: group.children.filter(c => c.userData.isRotor) as THREE.Mesh[],
        trail,
        label,
        trajectoryLine: null,
        interceptLine: null,
      }

      visualsRef.current.set(drone.id, visual)
    }

    // Update position
    visual.group.position.set(
      drone.pose.position.x,
      drone.pose.position.z, // Swap Y/Z for Three.js coordinate system
      -drone.pose.position.y
    )

    // Update rotation from quaternion
    visual.group.quaternion.set(
      drone.pose.orientation.x,
      drone.pose.orientation.z,
      -drone.pose.orientation.y,
      drone.pose.orientation.w
    )

    // Animate rotors
    for (const rotor of visual.rotors) {
      rotor.rotation.y += ROTOR_SPEED * (drone.status === 'airborne' ? 1 : 0.1)
    }

    // Update trail
    const positions = visual.trail.geometry.attributes.position as THREE.BufferAttribute
    const posArray = positions.array as Float32Array

    // Shift existing positions
    for (let i = (TRAIL_LENGTH - 1) * 3; i >= 3; i -= 3) {
      posArray[i] = posArray[i - 3]
      posArray[i + 1] = posArray[i - 2]
      posArray[i + 2] = posArray[i - 1]
    }

    // Add new position
    posArray[0] = visual.group.position.x
    posArray[1] = visual.group.position.y
    posArray[2] = visual.group.position.z

    positions.needsUpdate = true

    // Update label position
    if (showLabels) {
      visual.label.visible = true
      visual.label.position.copy(visual.group.position)
      visual.label.position.y += 0.5 * scale
    } else {
      visual.label.visible = false
    }

    // Update trajectory
    if (showTrajectories) {
      const trajectory = trajectoryPredictions.get(drone.id)
      if (trajectory && trajectory.length > 0) {
        if (!visual.trajectoryLine) {
          visual.trajectoryLine = createTrajectoryLine(getDroneColor(drone.type))
          scene.add(visual.trajectoryLine)
        }

        const points = trajectory.map(
          t => new THREE.Vector3(t.position.x, t.position.z, -t.position.y)
        )
        visual.trajectoryLine.geometry.setFromPoints(points)
        visual.trajectoryLine.computeLineDistances()
        visual.trajectoryLine.visible = true
      }
    } else if (visual.trajectoryLine) {
      visual.trajectoryLine.visible = false
    }
  }, [scene, scale, showLabels, showTrajectories, trajectoryPredictions])

  // Update intercept lines
  const updateInterceptLines = useCallback(() => {
    for (const mission of activeMissions) {
      const interceptorVisual = visualsRef.current.get(mission.interceptorId)
      const targetVisual = visualsRef.current.get(mission.targetId)

      if (interceptorVisual && targetVisual && mission.interceptPoint) {
        if (!interceptorVisual.interceptLine) {
          interceptorVisual.interceptLine = createInterceptLine()
          scene.add(interceptorVisual.interceptLine)
        }

        const points = [
          interceptorVisual.group.position.clone(),
          new THREE.Vector3(
            mission.interceptPoint.x,
            mission.interceptPoint.z,
            -mission.interceptPoint.y
          ),
        ]

        interceptorVisual.interceptLine.geometry.setFromPoints(points)
        interceptorVisual.interceptLine.computeLineDistances()
        interceptorVisual.interceptLine.visible = true
      }
    }
  }, [scene, activeMissions])

  // Remove stale visuals
  const cleanupStaleVisuals = useCallback(() => {
    const currentIds = new Set(drones.keys())

    for (const [id, visual] of visualsRef.current) {
      if (!currentIds.has(id)) {
        scene.remove(visual.group)
        scene.remove(visual.trail)
        scene.remove(visual.label)
        if (visual.trajectoryLine) scene.remove(visual.trajectoryLine)
        if (visual.interceptLine) scene.remove(visual.interceptLine)

        visual.group.traverse(obj => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry.dispose()
            if (Array.isArray(obj.material)) {
              obj.material.forEach(m => m.dispose())
            } else {
              obj.material.dispose()
            }
          }
        })

        // Dispose trail geometry and material
        visual.trail.geometry.dispose()
        if (visual.trail.material instanceof THREE.Material) {
          visual.trail.material.dispose()
        }

        // Dispose label texture and material
        if (visual.label.material instanceof THREE.SpriteMaterial) {
          visual.label.material.map?.dispose()
          visual.label.material.dispose()
        }

        // Dispose trajectory line resources
        if (visual.trajectoryLine) {
          visual.trajectoryLine.geometry.dispose()
          if (visual.trajectoryLine.material instanceof THREE.Material) {
            visual.trajectoryLine.material.dispose()
          }
        }

        // Dispose intercept line resources
        if (visual.interceptLine) {
          visual.interceptLine.geometry.dispose()
          if (visual.interceptLine.material instanceof THREE.Material) {
            visual.interceptLine.material.dispose()
          }
        }

        visualsRef.current.delete(id)
      }
    }
  }, [scene, drones])

  // Main update effect
  useEffect(() => {
    frameRef.current++

    // Update all drones
    for (const drone of drones.values()) {
      updateDroneVisual(drone)
    }

    // Update intercept lines
    updateInterceptLines()

    // Cleanup stale visuals every 60 frames
    if (frameRef.current % 60 === 0) {
      cleanupStaleVisuals()
    }
  }, [drones, updateDroneVisual, updateInterceptLines, cleanupStaleVisuals])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const visual of visualsRef.current.values()) {
        scene.remove(visual.group)
        scene.remove(visual.trail)
        scene.remove(visual.label)
        if (visual.trajectoryLine) scene.remove(visual.trajectoryLine)
        if (visual.interceptLine) scene.remove(visual.interceptLine)

        // Dispose all geometries and materials
        visual.group.traverse(obj => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry.dispose()
            if (Array.isArray(obj.material)) {
              obj.material.forEach(m => m.dispose())
            } else {
              obj.material.dispose()
            }
          }
        })

        visual.trail.geometry.dispose()
        if (visual.trail.material instanceof THREE.Material) {
          visual.trail.material.dispose()
        }

        if (visual.label.material instanceof THREE.SpriteMaterial) {
          visual.label.material.map?.dispose()
          visual.label.material.dispose()
        }

        if (visual.trajectoryLine) {
          visual.trajectoryLine.geometry.dispose()
          if (visual.trajectoryLine.material instanceof THREE.Material) {
            visual.trajectoryLine.material.dispose()
          }
        }

        if (visual.interceptLine) {
          visual.interceptLine.geometry.dispose()
          if (visual.interceptLine.material instanceof THREE.Material) {
            visual.interceptLine.material.dispose()
          }
        }
      }
      visualsRef.current.clear()
    }
  }, [scene])

  return null
}

export default DroneVisualizer