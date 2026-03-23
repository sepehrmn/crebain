/**
 * CREBAIN 3D Positioning Utilities
 * Adaptive Response & Awareness System (ARAS)
 *
 * Utility functions for 3D object positioning:
 * - Floor snapping with configurable threshold
 * - Bounding box calculation
 * - Grid snapping
 * - Rotation helpers
 *
 * Adapted from Dreamweave's positioning utilities
 */

import * as THREE from 'three'

// ─────────────────────────────────────────────────────────────────────────────
// FLOOR SNAPPING
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Snaps a position's Y coordinate to the floor if within threshold
 */
export function snapToFloor(
  position: THREE.Vector3,
  floorY = 0,
  threshold = 0.5
): THREE.Vector3 {
  const result = position.clone()
  const distanceToFloor = Math.abs(result.y - floorY)
  if (distanceToFloor < threshold) {
    result.y = floorY
  }
  return result
}

/**
 * Checks if a position is close enough to snap to floor
 */
export function shouldSnapToFloor(
  position: THREE.Vector3,
  floorY = 0,
  threshold = 0.5
): boolean {
  return Math.abs(position.y - floorY) < threshold
}

// ─────────────────────────────────────────────────────────────────────────────
// GRID SNAPPING
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Snaps a position to a grid
 */
export function snapToGrid(
  position: THREE.Vector3,
  gridSize = 1.0
): THREE.Vector3 {
  if (gridSize === 0) return position.clone()
  return new THREE.Vector3(
    Math.round(position.x / gridSize) * gridSize,
    Math.round(position.y / gridSize) * gridSize,
    Math.round(position.z / gridSize) * gridSize
  )
}

/**
 * Snaps only XZ to grid, keeping Y unchanged
 */
export function snapToGridXZ(
  position: THREE.Vector3,
  gridSize = 1.0
): THREE.Vector3 {
  if (gridSize === 0) return position.clone()
  return new THREE.Vector3(
    Math.round(position.x / gridSize) * gridSize,
    position.y,
    Math.round(position.z / gridSize) * gridSize
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// BOUNDING BOX UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Gets the bounding box of an object
 */
export function getObjectBounds(object: THREE.Object3D): THREE.Box3 {
  return new THREE.Box3().setFromObject(object)
}

/**
 * Gets the size of an object's bounding box
 */
export function getObjectSize(object: THREE.Object3D): THREE.Vector3 {
  const box = getObjectBounds(object)
  return box.getSize(new THREE.Vector3())
}

/**
 * Gets the center of an object's bounding box
 */
export function getObjectCenter(object: THREE.Object3D): THREE.Vector3 {
  const box = getObjectBounds(object)
  return box.getCenter(new THREE.Vector3())
}

/**
 * Gets the maximum horizontal dimension (max of x and z)
 */
export function getMaxHorizontalSize(object: THREE.Object3D): number {
  const size = getObjectSize(object)
  return Math.max(size.x, size.z)
}

// ─────────────────────────────────────────────────────────────────────────────
// ROTATION UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/** Standard rotation step (22.5 degrees) */
export const ROTATION_STEP = Math.PI / 8

/**
 * Rotates an object by a step amount around an axis
 */
export function rotateByStep(
  object: THREE.Object3D,
  axis: 'x' | 'y' | 'z',
  direction: 1 | -1,
  step = ROTATION_STEP
): void {
  object.rotation[axis] += step * direction
}

/**
 * Snaps rotation to nearest step increment
 */
export function snapRotation(
  rotation: THREE.Euler,
  step = ROTATION_STEP
): THREE.Euler {
  return new THREE.Euler(
    Math.round(rotation.x / step) * step,
    Math.round(rotation.y / step) * step,
    Math.round(rotation.z / step) * step,
    rotation.order
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PLANE INTERSECTION UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates a drag plane parallel to the camera at a given point
 */
export function createDragPlane(
  camera: THREE.Camera,
  point: THREE.Vector3
): THREE.Plane {
  const normal = new THREE.Vector3()
  camera.getWorldDirection(normal)
  return new THREE.Plane().setFromNormalAndCoplanarPoint(normal, point)
}

/**
 * Intersects a ray with a plane and returns the intersection point
 */
export function intersectRayWithPlane(
  raycaster: THREE.Raycaster,
  plane: THREE.Plane
): THREE.Vector3 | null {
  const intersection = new THREE.Vector3()
  const result = raycaster.ray.intersectPlane(plane, intersection)
  return result ? intersection : null
}

/**
 * Gets the intersection point of a ray with the ground plane (Y=0)
 */
export function intersectRayWithGround(
  raycaster: THREE.Raycaster,
  groundY = 0
): THREE.Vector3 | null {
  const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -groundY)
  return intersectRayWithPlane(raycaster, groundPlane)
}

// ─────────────────────────────────────────────────────────────────────────────
// TRANSFORM UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/** Standard scale step */
export const SCALE_STEP = 0.05

/**
 * Scales an object uniformly by a step amount
 */
export function scaleUniform(
  object: THREE.Object3D,
  direction: 1 | -1,
  step = SCALE_STEP
): void {
  const delta = step * direction
  const newScale = Math.max(0.01, object.scale.x + delta)
  object.scale.set(newScale, newScale, newScale)
}

/**
 * Gets position nudge amount based on object scale
 */
export function getNudgeAmount(object: THREE.Object3D, baseAmount = 0.05): number {
  return baseAmount * object.scale.x
}

/**
 * Nudges an object's position along an axis
 */
export function nudgePosition(
  object: THREE.Object3D,
  axis: 'x' | 'y' | 'z',
  direction: 1 | -1,
  baseAmount = 0.05
): void {
  const amount = getNudgeAmount(object, baseAmount) * direction
  object.position[axis] += amount
}

// ─────────────────────────────────────────────────────────────────────────────
// LOOK AT UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Makes an object look at a target while keeping it upright (horizontal only)
 */
export function lookAtUpright(
  object: THREE.Object3D,
  target: THREE.Vector3
): void {
  // Create a target at the same Y level as the object to prevent tilting
  const adjustedTarget = target.clone()
  adjustedTarget.y = object.position.y
  object.lookAt(adjustedTarget)
}

/**
 * Makes an object face the camera
 */
export function faceCamera(
  object: THREE.Object3D,
  camera: THREE.Camera
): void {
  const cameraPosition = camera.position.clone()
  cameraPosition.y = object.position.y
  object.lookAt(cameraPosition)
}

// ─────────────────────────────────────────────────────────────────────────────
// DISTANCE UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Gets the horizontal distance between two points (ignoring Y)
 */
export function horizontalDistance(a: THREE.Vector3, b: THREE.Vector3): number {
  const dx = a.x - b.x
  const dz = a.z - b.z
  return Math.sqrt(dx * dx + dz * dz)
}

/**
 * Gets the 3D distance between two points
 */
export function distance3D(a: THREE.Vector3, b: THREE.Vector3): number {
  return a.distanceTo(b)
}

// ─────────────────────────────────────────────────────────────────────────────
// COORDINATE CONVERSION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Converts screen coordinates to normalized device coordinates
 */
export function screenToNDC(
  screenX: number,
  screenY: number,
  container: HTMLElement
): THREE.Vector2 {
  const rect = container.getBoundingClientRect()
  return new THREE.Vector2(
    ((screenX - rect.left) / rect.width) * 2 - 1,
    -((screenY - rect.top) / rect.height) * 2 + 1
  )
}

/**
 * Creates a raycaster from screen coordinates
 */
export function createRaycasterFromScreen(
  screenX: number,
  screenY: number,
  container: HTMLElement,
  camera: THREE.Camera
): THREE.Raycaster {
  const ndc = screenToNDC(screenX, screenY, container)
  const raycaster = new THREE.Raycaster()
  raycaster.setFromCamera(ndc, camera)
  return raycaster
}
