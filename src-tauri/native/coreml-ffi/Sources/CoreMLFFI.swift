/**
 * CREBAIN CoreML FFI Library
 * Zero-latency native CoreML inference via C ABI
 *
 * Exports C-compatible functions for direct Rust FFI calls
 * Eliminates JSON serialization, process spawning, and base64 encoding
 */

import Foundation
import CoreML
import Vision
import AppKit
import Accelerate

// MARK: - C ABI Compatible Types

// MARK: - C ABI Compatible Detection Struct

/// Detection bounding box - plain C struct for FFI
/// Must match the Rust side exactly
@frozen
public struct CDetectionBox {
    public var x1: Float
    public var y1: Float  
    public var x2: Float
    public var y2: Float
    public var confidence: Float
    public var classIndex: Int32
    
    public init(x1: Float, y1: Float, x2: Float, y2: Float, confidence: Float, classIndex: Int32) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.classIndex = classIndex
    }
}

// MARK: - Global State (Thread-Safe)

private var globalModel: VNCoreMLModel?
private var globalRequest: VNCoreMLRequest?
private var isInitialized = false
private let initLock = NSLock()
private let detectLock = NSLock()  // Separate lock for detection to prevent blocking init
private var warmupComplete = false

// Pre-allocated detection buffer (avoid malloc per frame)
private var detectionBuffer: [CDetectionBox] = []
private let maxDetections = 256
private var detectionBufferPtr: UnsafeMutablePointer<CDetectionBox>?

// Cached timebase info (constant value - only needs to be computed once)
private var cachedTimebaseInfo: mach_timebase_info_data_t = {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    return info
}()

// MARK: - C ABI Exports

/// Initialize the CoreML model - call once at startup
/// Returns 0 on success, error code on failure
@_cdecl("coreml_init")
public func coreml_init(_ modelPath: UnsafePointer<CChar>) -> Int32 {
    initLock.lock()
    defer { initLock.unlock() }
    
    if isInitialized {
        return 0 // Already initialized
    }
    
    let path = String(cString: modelPath)
    
    do {
        let modelURL = URL(fileURLWithPath: path)
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine // Optimal for Apple Silicon
        config.allowLowPrecisionAccumulationOnGPU = true
        
        let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
        globalModel = try VNCoreMLModel(for: mlModel)
        
        // Pre-create request for reuse
        globalRequest = VNCoreMLRequest(model: globalModel!) { _, _ in }
        globalRequest?.imageCropAndScaleOption = .scaleFill
        globalRequest?.preferBackgroundProcessing = false
        
        // Pre-allocate detection buffer
        detectionBuffer = [CDetectionBox](repeating: CDetectionBox(x1: 0, y1: 0, x2: 0, y2: 0, confidence: 0, classIndex: 0), count: maxDetections)
        detectionBufferPtr = UnsafeMutablePointer<CDetectionBox>.allocate(capacity: maxDetections)
        
        // Warm-up passes for JIT compilation
        warmupModel()
        
        isInitialized = true
        return 0
    } catch {
        fputs("CoreML init error: \(error.localizedDescription)\n", stderr)
        return -1
    }
}

/// Warm up the model with dummy inference
private func warmupModel() {
    guard let model = globalModel else { return }
    
    // Create small dummy image for warmup
    let warmupSize = 64
    var pixelData = [UInt8](repeating: 128, count: warmupSize * warmupSize * 4)
    
    guard let context = CGContext(
        data: &pixelData,
        width: warmupSize,
        height: warmupSize,
        bitsPerComponent: 8,
        bytesPerRow: warmupSize * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let cgImage = context.makeImage() else { return }
    
    let request = VNCoreMLRequest(model: model) { _, _ in }
    request.imageCropAndScaleOption = .scaleFill
    
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    
    // 3 warmup passes
    for _ in 0..<3 {
        try? handler.perform([request])
    }
    
    warmupComplete = true
}

/// Run detection on raw RGBA pixel buffer
/// Returns: detection count, writes detections to pre-allocated buffer
/// Output pointers: detectionCount, inferenceTimeNs, preprocessTimeNs, postprocessTimeNs
@_cdecl("coreml_detect")
public func coreml_detect(
    _ pixels: UnsafePointer<UInt8>,
    _ width: Int32,
    _ height: Int32,
    _ bytesPerRow: Int32,
    _ confidenceThreshold: Float,
    _ outDetectionCount: UnsafeMutablePointer<Int32>,
    _ outInferenceTimeNs: UnsafeMutablePointer<UInt64>,
    _ outPreprocessTimeNs: UnsafeMutablePointer<UInt64>,
    _ outPostprocessTimeNs: UnsafeMutablePointer<UInt64>
) -> Int32 {
    // Acquire lock to safely read shared state
    detectLock.lock()
    defer { detectLock.unlock() }
    
    guard isInitialized, let model = globalModel else {
        outDetectionCount.pointee = 0
        return -1
    }
    
    let preprocessStart = mach_absolute_time()
    
    // Create CGImage directly from pixel buffer (zero-copy when possible)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: UnsafeMutableRawPointer(mutating: pixels),
        width: Int(width),
        height: Int(height),
        bitsPerComponent: 8,
        bytesPerRow: Int(bytesPerRow),
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ), let cgImage = context.makeImage() else {
        outDetectionCount.pointee = 0
        return -2
    }
    
    let preprocessEnd = mach_absolute_time()
    
    // Run inference
    let inferenceStart = mach_absolute_time()
    
    var detections: [VNRecognizedObjectObservation] = []
    
    let request = VNCoreMLRequest(model: model) { req, _ in
        if let results = req.results as? [VNRecognizedObjectObservation] {
            detections = results
        }
    }
    request.imageCropAndScaleOption = .scaleFill
    request.preferBackgroundProcessing = false
    
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    
    do {
        try handler.perform([request])
    } catch {
        outDetectionCount.pointee = 0
        return -3
    }
    
    let inferenceEnd = mach_absolute_time()
    
    // Post-process
    let postprocessStart = mach_absolute_time()
    
    let imageWidth = Float(width)
    let imageHeight = Float(height)
    var detectionCount: Int32 = 0
    
    for observation in detections {
        guard observation.confidence >= confidenceThreshold else { continue }
        guard detectionCount < Int32(maxDetections) else { break }
        
        // Convert Vision coordinates to pixel coordinates
        let bbox = observation.boundingBox
        let x1 = Float(bbox.origin.x) * imageWidth
        let y1 = (1.0 - Float(bbox.origin.y + bbox.height)) * imageHeight
        let x2 = Float(bbox.origin.x + bbox.width) * imageWidth
        let y2 = (1.0 - Float(bbox.origin.y)) * imageHeight
        
        // Get class index
        let classIndex: Int32 = observation.labels.first.map { Int32(cocoClassIndex($0.identifier)) } ?? -1
        
        // Write to pre-allocated buffer
        detectionBufferPtr?[Int(detectionCount)] = CDetectionBox(
            x1: x1,
            y1: y1,
            x2: x2,
            y2: y2,
            confidence: observation.confidence,
            classIndex: classIndex
        )
        
        detectionCount += 1
    }
    
    let postprocessEnd = mach_absolute_time()
    
    // Convert mach time to nanoseconds using cached timebase info
    let numer = UInt64(cachedTimebaseInfo.numer)
    let denom = UInt64(cachedTimebaseInfo.denom)
    
    outDetectionCount.pointee = detectionCount
    outPreprocessTimeNs.pointee = (preprocessEnd - preprocessStart) * numer / denom
    outInferenceTimeNs.pointee = (inferenceEnd - inferenceStart) * numer / denom
    outPostprocessTimeNs.pointee = (postprocessEnd - postprocessStart) * numer / denom
    
    return 0
}

/// Get pointer to the detection buffer (for reading detections from Rust)
/// Returns null pointer if not initialized - caller MUST check for null
@_cdecl("coreml_get_detections")
public func coreml_get_detections() -> UnsafeRawPointer? {
    guard let ptr = detectionBufferPtr else {
        return nil
    }
    return UnsafeRawPointer(ptr)
}

/// Cleanup and release resources
@_cdecl("coreml_cleanup")
public func coreml_cleanup() {
    initLock.lock()
    defer { initLock.unlock() }
    
    globalModel = nil
    globalRequest = nil
    detectionBufferPtr?.deallocate()
    detectionBufferPtr = nil
    detectionBuffer = []
    isInitialized = false
    warmupComplete = false
}

/// Check if CoreML is initialized
@_cdecl("coreml_is_ready")
public func coreml_is_ready() -> Bool {
    return isInitialized && warmupComplete
}

/// Get the model input size (for preprocessing on Rust side if needed)
@_cdecl("coreml_get_input_size")
public func coreml_get_input_size(_ width: UnsafeMutablePointer<Int32>, _ height: UnsafeMutablePointer<Int32>) {
    // YOLOv8 default input size
    width.pointee = 640
    height.pointee = 640
}

// MARK: - COCO Class Mapping

private let cocoClasses = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

private func cocoClassIndex(_ identifier: String) -> Int {
    return cocoClasses.firstIndex(of: identifier.lowercased()) ?? -1
}
