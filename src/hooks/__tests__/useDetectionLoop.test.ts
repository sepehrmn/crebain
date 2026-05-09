import { describe, it, expect } from 'vitest'
import { convertDetection, imageDataToRGBA } from '../useDetectionLoop'
import type { CoreMLDetection } from '../../detection/types'

describe('useDetectionLoop helpers', () => {
  it('maps native detections to tactical detections', () => {
    const nativeDetection: CoreMLDetection = {
      id: 'det-1',
      classLabel: 'kite',
      classIndex: 33,
      confidence: 0.9,
      bbox: { x1: 10, y1: 20, x2: 30, y2: 40 },
      timestamp: 1234,
    }

    const detection = convertDetection(nativeDetection, 640, 480)

    expect(detection).toMatchObject({
      id: 'det-1',
      class: 'drone',
      confidence: 0.9,
      bbox: [10, 20, 30, 40],
      timestamp: 1234,
      threatLevel: 4,
      frameWidth: 640,
      frameHeight: 480,
    })
  })

  it('maps aerial and unknown detection labels consistently', () => {
    expect(convertDetection({
      id: 'aircraft-1',
      classLabel: 'airplane',
      classIndex: 4,
      confidence: 0.99,
      bbox: { x1: 0, y1: 0, x2: 1, y2: 1 },
      timestamp: 1,
    }, 10, 10)).toMatchObject({
      class: 'aircraft',
      threatLevel: 2,
    })
    expect(convertDetection({
      id: 'unknown-1',
      classLabel: 'balloon',
      classIndex: 0,
      confidence: 0.8,
      bbox: { x1: 1, y1: 2, x2: 3, y2: 4 },
      timestamp: 2,
    }, 20, 20)).toMatchObject({
      class: 'unknown',
      threatLevel: 3,
      frameWidth: 20,
      frameHeight: 20,
    })
  })

  it('creates a zero-copy Uint8Array view over ImageData', () => {
    const imageData = {
      data: new Uint8ClampedArray([1, 2, 3, 4, 5, 6, 7, 8]),
      width: 1,
      height: 2,
      colorSpace: 'srgb',
    } as ImageData
    const rgba = imageDataToRGBA(imageData)

    expect(rgba).toBeInstanceOf(Uint8Array)
    expect(Array.from(rgba)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])

    imageData.data[0] = 9
    expect(rgba[0]).toBe(9)
  })

  it('preserves byte offsets when creating RGBA views', () => {
    const source = new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7])
    const imageData = {
      data: new Uint8ClampedArray(source.buffer, 4, 4),
      width: 1,
      height: 1,
      colorSpace: 'srgb',
    } as ImageData

    const rgba = imageDataToRGBA(imageData)

    expect(Array.from(rgba)).toEqual([4, 5, 6, 7])
    imageData.data[1] = 9
    expect(rgba[1]).toBe(9)
  })
})
