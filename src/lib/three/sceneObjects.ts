import type * as THREE from 'three'

/**
 * Type guard for {@link THREE.Mesh}.
 *
 * `node instanceof THREE.Mesh` narrows to `Mesh<any, any, any>` in TypeScript,
 * which silently leaks `any` through `.geometry` and `.material`. Narrowing via
 * the runtime `isMesh` flag yields a concrete `THREE.Mesh` whose members stay
 * fully typed.
 */
export function isMesh(object: THREE.Object3D): object is THREE.Mesh {
  return (object as THREE.Mesh).isMesh
}

/**
 * Invoke `callback` for every {@link THREE.Mesh} in the subtree rooted at `root`
 * (inclusive), with the mesh narrowed to a concrete, type-safe `THREE.Mesh`.
 */
export function forEachMesh(root: THREE.Object3D, callback: (mesh: THREE.Mesh) => void): void {
  root.traverse((node) => {
    if (isMesh(node)) {
      callback(node)
    }
  })
}

/**
 * Resolve a human-readable label for a scene object: its `name`, falling back to
 * a string `userData.id`, then to `fallback`. Centralizes the safe read of the
 * untyped `userData` bag so callers stay type-safe.
 */
export function objectLabel(object: THREE.Object3D, fallback = 'OBJEKT'): string {
  if (object.name) {
    return object.name
  }
  const id: unknown = object.userData.id
  return typeof id === 'string' ? id : fallback
}

/**
 * Resolve a stable identifier for a scene object: a string `userData.id`, then a
 * string `userData.assetId`, then the object's intrinsic `uuid`. Centralizes the
 * safe read of the untyped `userData` bag so callers stay type-safe.
 */
export function objectId(object: THREE.Object3D): string {
  const id: unknown = object.userData.id
  if (typeof id === 'string') {
    return id
  }
  const assetId: unknown = object.userData.assetId
  if (typeof assetId === 'string') {
    return assetId
  }
  return object.uuid
}

/**
 * Recursively dispose the GPU resources (geometries and materials) of every mesh
 * in the subtree rooted at `root`.
 *
 * three.js does not release these resources automatically when an object is
 * removed from a scene, so callers must dispose meshes explicitly to avoid GPU
 * memory leaks.
 */
export function disposeObject3D(root: THREE.Object3D): void {
  forEachMesh(root, (mesh) => {
    mesh.geometry.dispose()
    const material = mesh.material
    if (Array.isArray(material)) {
      for (const entry of material) {
        entry.dispose()
      }
    } else {
      material.dispose()
    }
  })
}
