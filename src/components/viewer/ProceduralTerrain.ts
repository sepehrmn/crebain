/**
 * @fileoverview Procedural terrain and floor texture generation for 3D scenes.
 * Provides deterministic, seeded generation of ground surfaces including
 * terrain meshes with height displacement and flat floors with various textures.
 *
 * @license MIT
 */

import * as THREE from 'three'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Available floor/ground surface styles.
 * - `concrete`: Grey noisy surface simulating concrete
 * - `grass`: Green noisy surface simulating grass
 * - `asphalt`: Dark noisy surface simulating asphalt
 * - `checker`: Checkerboard pattern for debugging/visualization
 * - `terrain`: 3D terrain mesh with height displacement
 */
export type FloorStyle = 'concrete' | 'grass' | 'asphalt' | 'checker' | 'terrain'

// ─────────────────────────────────────────────────────────────────────────────
// Random Number Generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GPU-style hash function commonly used in shader noise implementations.
 * Based on the well-known sin-based hash from "The Book of Shaders".
 */
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453
  return x - Math.floor(x)
}

/** POSIX linear congruential generator multiplier */
const LCG_MULTIPLIER = 1103515245
/** POSIX linear congruential generator increment */
const LCG_INCREMENT = 12345
/** POSIX linear congruential generator modulus (2^31 - 1) */
const LCG_MODULUS = 0x7fffffff

/**
 * Creates a seeded pseudo-random number generator using the linear congruential method.
 * Uses POSIX standard constants for compatibility and well-understood distribution.
 *
 * @param initialSeed - Starting seed value for the generator
 * @returns A function that returns the next random number in [0, 1) on each call
 */
function createSeededGenerator(initialSeed: number): () => number {
  let state = initialSeed
  return () => {
    state = (state * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MODULUS
    return state / LCG_MODULUS
  }
}

/**
 * Generates a hash code from a string using the djb2 algorithm.
 * This algorithm provides good distribution for short strings like style names.
 *
 * @param str - Input string to hash
 * @returns 32-bit unsigned integer hash
 */
function hashString(str: string): number {
  let hash = 5381
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i)
  }
  return hash >>> 0
}

// ─────────────────────────────────────────────────────────────────────────────
// Texture Generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generates a procedural noise texture using HTML5 Canvas.
 * Creates a base color with scattered dots of a secondary color,
 * useful for simulating natural surfaces like concrete, grass, or asphalt.
 *
 * @param width - Texture width in pixels
 * @param height - Texture height in pixels
 * @param color1 - Base/background color (CSS color string)
 * @param color2 - Noise dot color (CSS color string)
 * @param dotCount - Number of noise dots to render
 * @param seed - Random seed for deterministic generation
 * @returns THREE.Texture configured for repeating/tiling
 *
 * @example
 * ```ts
 * const concreteTexture = createNoiseTexture(512, 512, '#808080', '#909090', 5000, 12345)
 * concreteTexture.repeat.set(10, 10)
 * ```
 */
export function createNoiseTexture(
  width: number,
  height: number,
  color1: string,
  color2: string,
  dotCount: number,
  seed: number
): THREE.Texture {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get 2d canvas context')

  const rng = createSeededGenerator(seed)

  ctx.fillStyle = color1
  ctx.fillRect(0, 0, width, height)

  for (let i = 0; i < dotCount; i++) {
    const x = rng() * width
    const y = rng() * height
    const radius = rng() * 2 + 1

    ctx.fillStyle = color2
    ctx.globalAlpha = rng() * 0.1 + 0.05
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.colorSpace = THREE.SRGBColorSpace

  return texture
}

/**
 * Generates a checkerboard texture for debugging or stylized surfaces.
 *
 * @param size - Texture dimensions in pixels (square)
 * @param color1 - First checker color (CSS color string)
 * @param color2 - Second checker color (CSS color string)
 * @param segments - Number of checker squares per row/column
 * @returns THREE.Texture configured for repeating/tiling with nearest-neighbor filtering
 *
 * @example
 * ```ts
 * const gridTexture = createCheckerTexture(512, '#404040', '#505050', 8)
 * gridTexture.repeat.set(20, 20)
 * ```
 */
export function createCheckerTexture(
  size: number = 512,
  color1: string = '#cccccc',
  color2: string = '#888888',
  segments: number = 8
): THREE.Texture {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get 2d canvas context')

  const step = size / segments

  for (let row = 0; row < segments; row++) {
    for (let col = 0; col < segments; col++) {
      ctx.fillStyle = (col + row) % 2 === 0 ? color1 : color2
      ctx.fillRect(col * step, row * step, step, step)
    }
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.LinearMipMapLinearFilter
  texture.colorSpace = THREE.SRGBColorSpace

  return texture
}

// ─────────────────────────────────────────────────────────────────────────────
// Terrain Noise Functions
// ─────────────────────────────────────────────────────────────────────────────

/** Terrain center flat zone radius in world units */
const TERRAIN_FLAT_RADIUS = 30

/**
 * Multi-octave deterministic noise for terrain base shape.
 * Combines three sine/cosine waves at different frequencies to create
 * natural-looking rolling hills.
 */
function terrainNoise(x: number, z: number): number {
  const lowFreq = Math.sin(x * 0.05) * 4
  const midFreq = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 2
  const highFreq = Math.sin(x * 0.3 + 100) * Math.cos(z * 0.3 + 100)
  return lowFreq + midFreq + highFreq
}

/**
 * High-frequency detail noise for terrain surface roughness.
 * Uses GPU-style hash for per-vertex variation.
 */
function roughnessNoise(x: number, z: number): number {
  const seed = Math.sin(x * 12.9898 + z * 78.233) * 43758.5453
  return (seededRandom(seed) - 0.5) * 0.2
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh Generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates an uneven terrain mesh with procedural height displacement.
 * The terrain features rolling hills with a flattened center area for usability.
 * Uses flat shading to emphasize the low-poly aesthetic.
 *
 * @param width - Terrain width in world units (default: 200)
 * @param depth - Terrain depth in world units (default: 200)
 * @param segments - Geometry subdivision level (default: 64)
 * @returns THREE.Mesh with displaced geometry and grid texture
 *
 * @remarks
 * The mesh is configured with:
 * - `receiveShadow: true` for shadow mapping
 * - `userData.isFloor: true` for raycasting identification
 * - Double-sided rendering for viewing from below
 *
 * @example
 * ```ts
 * const terrain = createTerrainMesh(200, 200, 64)
 * scene.add(terrain)
 * ```
 */
export function createTerrainMesh(
  width: number = 200,
  depth: number = 200,
  segments: number = 64
): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(width, depth, segments, segments)
  geometry.rotateX(-Math.PI / 2)

  const posAttribute = geometry.attributes.position

  for (let i = 0; i < posAttribute.count; i++) {
    const x = posAttribute.getX(i)
    const z = posAttribute.getZ(i)

    let y = terrainNoise(x, z) + roughnessNoise(x, z)

    // Flatten center area for better usability (landing zone)
    const distFromCenter = Math.sqrt(x * x + z * z)
    if (distFromCenter < TERRAIN_FLAT_RADIUS) {
      y *= distFromCenter / TERRAIN_FLAT_RADIUS
    }

    posAttribute.setY(i, y)
  }

  geometry.computeVertexNormals()

  const texture = createCheckerTexture(512, '#2a2a2a', '#333333', 16)
  texture.repeat.set(width / 5, depth / 5)

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.9,
    metalness: 0.1,
    flatShading: true,
    side: THREE.DoubleSide,
  })

  const mesh = new THREE.Mesh(geometry, material)
  mesh.receiveShadow = true
  mesh.userData.isFloor = true

  return mesh
}

/**
 * Creates a flat floor mesh with a procedural texture based on the specified style.
 *
 * @param style - Floor surface style (concrete, grass, asphalt, or checker)
 * @param size - Floor dimensions in world units (default: 200)
 * @returns THREE.Mesh positioned slightly below y=0 to prevent z-fighting with grid
 *
 * @remarks
 * The mesh is configured with:
 * - `receiveShadow: true` for shadow mapping
 * - `userData.isFloor: true` for raycasting identification
 * - Position offset of -0.05 on Y axis to prevent z-fighting
 *
 * @example
 * ```ts
 * const floor = createProceduralFloor('concrete', 200)
 * scene.add(floor)
 * ```
 */
export function createProceduralFloor(
  style: Exclude<FloorStyle, 'terrain'>,
  size: number = 200
): THREE.Mesh {
  const seed = hashString(style)
  let texture: THREE.Texture
  let roughness: number

  switch (style) {
    case 'concrete':
      texture = createNoiseTexture(512, 512, '#808080', '#909090', 5000, seed)
      texture.repeat.set(10, 10)
      roughness = 0.9
      break
    case 'asphalt':
      texture = createNoiseTexture(512, 512, '#202020', '#303030', 8000, seed)
      texture.repeat.set(10, 10)
      roughness = 0.95
      break
    case 'grass':
      texture = createNoiseTexture(512, 512, '#2d5a27', '#3a6b35', 6000, seed)
      texture.repeat.set(20, 20)
      roughness = 1.0
      break
    case 'checker':
    default:
      texture = createCheckerTexture(512, '#404040', '#505050', 8)
      texture.repeat.set(size / 10, size / 10)
      roughness = 0.8
      break
  }

  const geometry = new THREE.PlaneGeometry(size, size)
  geometry.rotateX(-Math.PI / 2)

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    roughness,
    metalness: 0.1,
  })

  const mesh = new THREE.Mesh(geometry, material)
  mesh.receiveShadow = true
  mesh.position.y = -0.05
  mesh.userData.isFloor = true

  return mesh
}
