/**
 * CREBAIN 3D Object Dragging Hook
 * Adaptive Response & Awareness System (ARAS)
 *
 * Provides 3D object dragging with plane-based movement, floor snapping,
 * and smooth interaction. Adapted from Dreamweave's positioning system.
 *
 * Features:
 * - Plane-based dragging parallel to camera view
 * - Floor plane snapping with configurable threshold
 * - Offset calculation to prevent "jumping" on grab
 * - Automatic OrbitControls disabling during drag
 * - Support for any THREE.Object3D
 */

import { useRef, useCallback, useEffect, useState } from 'react'
import * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface Draggable3DConfig {
  /** Reference to the container element for pointer events */
  containerRef: React.RefObject<HTMLElement | null>
  /** Reference to the camera */
  cameraRef: React.RefObject<THREE.PerspectiveCamera | null>
  /** Reference to the scene */
  sceneRef: React.RefObject<THREE.Scene | null>
  /** Reference to orbit controls (will be disabled during drag) */
  controlsRef?: React.RefObject<OrbitControls | null>
  /** Objects that can be dragged */
  draggableObjects: THREE.Object3D[]
  /** Floor Y level for snapping (default: 0) */
  floorY?: number
  /** Snap distance threshold (default: 0.5) */
  snapThreshold?: number
  /** Enable floor snapping (default: true) */
  enableFloorSnap?: boolean
  /** Callback when drag starts */
  onDragStart?: (object: THREE.Object3D) => void
  /** Callback during drag */
  onDrag?: (object: THREE.Object3D, position: THREE.Vector3) => void
  /** Callback when drag ends */
  onDragEnd?: (object: THREE.Object3D, position: THREE.Vector3) => void
  /** Whether dragging is enabled (default: true) */
  enabled?: boolean
}

export interface Draggable3DState {
  isDragging: boolean
  draggedObjectId: string | null
  plane: THREE.Plane
  offset: THREE.Vector3
  startPosition: THREE.Vector3
}

export interface Draggable3DReturn {
  /** Whether currently dragging */
  isDragging: boolean
  /** ID of the currently dragged object (from userData.id or uuid) */
  draggedObjectId: string | null
  /** Start drag programmatically */
  startDrag: (object: THREE.Object3D, event: PointerEvent | MouseEvent) => void
  /** Cancel current drag */
  cancelDrag: () => void
  /** Get the raycaster for external use */
  raycaster: THREE.Raycaster
}

// ─────────────────────────────────────────────────────────────────────────────
// HOOK
// ─────────────────────────────────────────────────────────────────────────────

export function useDraggable3D(config: Draggable3DConfig): Draggable3DReturn {
  const {
    containerRef,
    cameraRef,
    sceneRef: _sceneRef,
    controlsRef,
    draggableObjects,
    floorY = 0,
    snapThreshold = 0.5,
    enableFloorSnap = true,
    onDragStart,
    onDrag,
    onDragEnd,
    enabled = true,
  } = config

  const [isDragging, setIsDragging] = useState(false)
  const [draggedObjectId, setDraggedObjectId] = useState<string | null>(null)

  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster())
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2())

  const dragState = useRef<Draggable3DState>({
    isDragging: false,
    draggedObjectId: null,
    plane: new THREE.Plane(),
    offset: new THREE.Vector3(),
    startPosition: new THREE.Vector3(),
  })

  const draggedObjectRef = useRef<THREE.Object3D | null>(null)

  // Get object ID helper
  const getObjectId = useCallback((obj: THREE.Object3D): string => {
    return obj.userData.id || obj.userData.assetId || obj.uuid
  }, [])

  // Convert pointer event to normalized device coordinates
  const getNDC = useCallback((event: PointerEvent | MouseEvent): THREE.Vector2 => {
    const container = containerRef.current
    if (!container) return new THREE.Vector2()

    const rect = container.getBoundingClientRect()
    return new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    )
  }, [containerRef])

  // Start drag
  const startDrag = useCallback((object: THREE.Object3D, event: PointerEvent | MouseEvent) => {
    const camera = cameraRef.current
    if (!camera || !enabled) return

    const ndc = getNDC(event)
    mouseRef.current.copy(ndc)
    raycasterRef.current.setFromCamera(mouseRef.current, camera)

    // Create drag plane parallel to camera at object position
    const normal = new THREE.Vector3()
    camera.getWorldDirection(normal)
    dragState.current.plane.setFromNormalAndCoplanarPoint(normal, object.position)

    // Calculate offset from intersection point to object center
    const intersectPoint = new THREE.Vector3()
    const intersected = raycasterRef.current.ray.intersectPlane(dragState.current.plane, intersectPoint)
    if (!intersected) {
      // Ray is parallel to plane, can't start drag
      return
    }
    dragState.current.offset.subVectors(object.position, intersectPoint)
    dragState.current.startPosition.copy(object.position)

    // Store references
    dragState.current.isDragging = true
    dragState.current.draggedObjectId = getObjectId(object)
    draggedObjectRef.current = object

    setIsDragging(true)
    setDraggedObjectId(getObjectId(object))

    // Disable orbit controls during drag
    if (controlsRef?.current) {
      controlsRef.current.enabled = false
    }

    onDragStart?.(object)
  }, [cameraRef, controlsRef, enabled, getNDC, getObjectId, onDragStart])

  // Cancel drag
  const cancelDrag = useCallback(() => {
    if (!dragState.current.isDragging || !draggedObjectRef.current) return

    // Restore original position
    draggedObjectRef.current.position.copy(dragState.current.startPosition)

    // Reset state
    dragState.current.isDragging = false
    dragState.current.draggedObjectId = null
    draggedObjectRef.current = null

    setIsDragging(false)
    setDraggedObjectId(null)

    // Re-enable orbit controls
    if (controlsRef?.current) {
      controlsRef.current.enabled = true
    }
  }, [controlsRef])

  // Pointer event handlers
  useEffect(() => {
    const container = containerRef.current
    const camera = cameraRef.current
    if (!container || !camera || !enabled) return

    const handlePointerDown = (event: PointerEvent) => {
      // Only handle left click
      if (event.button !== 0) return

      // Ignore UI elements
      const target = event.target as HTMLElement
      if (target.closest('button, input, [data-no-drag]')) return

      const ndc = getNDC(event)
      mouseRef.current.copy(ndc)
      raycasterRef.current.setFromCamera(mouseRef.current, camera)

      // Check intersection with draggable objects
      const intersects = raycasterRef.current.intersectObjects(draggableObjects, true)

      if (intersects.length > 0) {
        // Find the root draggable object
        let targetObject = intersects[0].object
        while (targetObject.parent && !draggableObjects.includes(targetObject)) {
          if (targetObject.parent.type === 'Scene') break
          targetObject = targetObject.parent
        }

        // If we found a draggable object, start drag
        const draggable = draggableObjects.find(obj => 
          obj === targetObject || obj.children.some(child => child === targetObject)
        )

        if (draggable) {
          event.preventDefault()
          event.stopPropagation()
          container.setPointerCapture(event.pointerId)
          startDrag(draggable, event)
        }
      }
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (!dragState.current.isDragging || !draggedObjectRef.current) return

      const ndc = getNDC(event)
      mouseRef.current.copy(ndc)
      raycasterRef.current.setFromCamera(mouseRef.current, camera)

      // Intersect ray with drag plane
      const intersectPoint = new THREE.Vector3()
      if (raycasterRef.current.ray.intersectPlane(dragState.current.plane, intersectPoint)) {
        // Apply offset to get new position
        const newPosition = intersectPoint.add(dragState.current.offset)

        // Apply floor snapping if enabled
        if (enableFloorSnap) {
          const distanceToFloor = Math.abs(newPosition.y - floorY)
          if (distanceToFloor < snapThreshold) {
            newPosition.y = floorY
          }
        }

        // Update object position
        draggedObjectRef.current.position.copy(newPosition)

        onDrag?.(draggedObjectRef.current, newPosition)
      }
    }

    const handlePointerUp = (event: PointerEvent) => {
      if (!dragState.current.isDragging || !draggedObjectRef.current) return

      container.releasePointerCapture(event.pointerId)

      const finalPosition = draggedObjectRef.current.position.clone()

      // Apply final floor snap if close enough
      if (enableFloorSnap) {
        const distanceToFloor = Math.abs(finalPosition.y - floorY)
        if (distanceToFloor < snapThreshold) {
          finalPosition.y = floorY
          draggedObjectRef.current.position.y = floorY
        }
      }

      onDragEnd?.(draggedObjectRef.current, finalPosition)

      // Reset state
      dragState.current.isDragging = false
      dragState.current.draggedObjectId = null
      draggedObjectRef.current = null

      setIsDragging(false)
      setDraggedObjectId(null)

      // Re-enable orbit controls
      if (controlsRef?.current) {
        controlsRef.current.enabled = true
      }
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && dragState.current.isDragging) {
        cancelDrag()
      }
    }

    // Handle pointer cancel (e.g., touch interrupted, pointer leaves window)
    const handlePointerCancel = (event: PointerEvent) => {
      if (dragState.current.isDragging) {
        try {
          container.releasePointerCapture(event.pointerId)
        } catch {
          // Pointer capture may already be released
        }
        cancelDrag()
      }
    }

    // Touch support - save original value for cleanup
    const originalTouchAction = container.style.touchAction
    container.style.touchAction = 'none'

    container.addEventListener('pointerdown', handlePointerDown)
    container.addEventListener('pointermove', handlePointerMove)
    container.addEventListener('pointerup', handlePointerUp)
    container.addEventListener('pointercancel', handlePointerCancel)
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      container.style.touchAction = originalTouchAction
      container.removeEventListener('pointerdown', handlePointerDown)
      container.removeEventListener('pointermove', handlePointerMove)
      container.removeEventListener('pointerup', handlePointerUp)
      container.removeEventListener('pointercancel', handlePointerCancel)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [
    containerRef,
    cameraRef,
    controlsRef,
    draggableObjects,
    enabled,
    enableFloorSnap,
    floorY,
    snapThreshold,
    getNDC,
    startDrag,
    cancelDrag,
    onDrag,
    onDragEnd,
  ])

  return {
    isDragging,
    draggedObjectId,
    startDrag,
    cancelDrag,
    raycaster: raycasterRef.current,
  }
}

export default useDraggable3D
