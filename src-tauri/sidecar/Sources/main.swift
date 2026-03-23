/**
 * CREBAIN CoreML YOLOv8 Detector Sidecar
 * Adaptive Response & Awareness System (ARAS)
 *
 * OPTIMIZED VERSION - Nanosecond-level performance
 * 
 * Optimizations applied:
 * - Pre-allocated reusable VNCoreMLRequest
 * - Model warm-up pass for JIT compilation
 * - cpuAndNeuralEngine compute units for optimal ANE utilization
 * - allowLowPrecisionAccumulationOnGPU enabled
 * - Synchronous prediction for lowest latency
 * - mach_absolute_time for nanosecond precision timing
 * - Pre-allocated detection buffers
 */

import Foundation
import CoreML
import Vision
import AppKit

// MARK: - High-Precision Timing

/// Get nanosecond-precision timestamp
@inline(__always)
func getNanoseconds() -> UInt64 {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let time = mach_absolute_time()
    return time * UInt64(info.numer) / UInt64(info.denom)
}

/// Convert nanoseconds to milliseconds
@inline(__always)
func nanosToMs(_ nanos: UInt64) -> Double {
    return Double(nanos) / 1_000_000.0
}

// MARK: - Detection Types

struct BoundingBox: Codable {
    let x1: Double
    let y1: Double
    let x2: Double
    let y2: Double
}

struct Detection: Codable {
    let id: String
    let classLabel: String
    let classIndex: Int
    let confidence: Double
    let bbox: BoundingBox
    let timestamp: Int64
}

struct DetectionResult: Codable {
    let success: Bool
    let detections: [Detection]
    let inferenceTimeMs: Double
    let preprocessTimeMs: Double?
    let postprocessTimeMs: Double?
    let error: String?
}

struct InputPayload: Codable {
    let imageBase64: String
    let confidenceThreshold: Double?
    let iouThreshold: Double?
    let maxDetections: Int?
}

// MARK: - YOLO Class Labels (COCO 80 classes) - Static allocation

let cocoClasses: ContiguousArray<String> = [
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

// Pre-computed class index lookup for O(1) access
let cocoClassIndexMap: [String: Int] = {
    var map = [String: Int](minimumCapacity: cocoClasses.count)
    for (index, className) in cocoClasses.enumerated() {
        map[className] = index
    }
    return map
}()

// Map COCO classes to tactical classes - inlined for performance
@inline(__always)
func mapToTacticalClass(_ cocoClass: String) -> String {
    switch cocoClass {
    case "airplane": return "aircraft"
    case "bird": return "bird"
    case "kite", "frisbee": return "drone"
    default: return cocoClass
    }
}

// MARK: - Optimized CoreML Detector

final class OptimizedCoreMLDetector {
    private let model: VNCoreMLModel
    private let request: VNCoreMLRequest
    private let confidenceThreshold: Float
    private let maxDetections: Int
    
    // Pre-allocated detection buffer
    private var detectionBuffer: ContiguousArray<Detection>
    
    // Cached timestamp for batch operations
    private var cachedTimestamp: Int64 = 0
    
    // Detection counter for ID generation (faster than UUID)
    private var detectionCounter: UInt64 = 0
    
    init?(modelPath: String, confidenceThreshold: Double = 0.25, maxDetections: Int = 100) {
        self.confidenceThreshold = Float(confidenceThreshold)
        self.maxDetections = maxDetections
        self.detectionBuffer = ContiguousArray<Detection>()
        self.detectionBuffer.reserveCapacity(maxDetections)
        
        do {
            let modelURL = URL(fileURLWithPath: modelPath)
            let compiledURL: URL
            
            if modelPath.hasSuffix(".mlmodelc") {
                compiledURL = modelURL
            } else {
                compiledURL = try MLModel.compileModel(at: modelURL)
            }
            
            // OPTIMIZATION: Configure for maximum performance
            let config = MLModelConfiguration()
            
            // Use CPU + Neural Engine (skip GPU for lower latency on most models)
            // ANE is fastest for neural network ops on Apple Silicon
            config.computeUnits = .cpuAndNeuralEngine
            
            // Enable low precision accumulation for GPU fallback operations
            // This uses FP16 instead of FP32 for ~20-50% speedup
            if #available(macOS 12.0, *) {
                config.allowLowPrecisionAccumulationOnGPU = true
            }
            
            let mlModel = try MLModel(contentsOf: compiledURL, configuration: config)
            self.model = try VNCoreMLModel(for: mlModel)
            
            // OPTIMIZATION: Pre-allocate reusable VNCoreMLRequest
            self.request = VNCoreMLRequest(model: self.model)
            
            // Configure request for optimal performance
            self.request.imageCropAndScaleOption = .scaleFill
            
            // Prefer background processing disabled for lowest latency
            if #available(macOS 12.0, *) {
                self.request.preferBackgroundProcessing = false
            }
            
            // OPTIMIZATION: Warm-up inference pass
            // This triggers JIT compilation and caches kernel executions
            performWarmup()
            
        } catch {
            fputs("Error loading CoreML model: \(error.localizedDescription)\n", stderr)
            return nil
        }
    }
    
    /// Perform warm-up inference to trigger JIT compilation
    private func performWarmup() {
        fputs("Performing warm-up inference...\n", stderr)
        
        // Create a small dummy image (64x64 is enough to warm up the pipeline)
        let warmupSize = 64
        let bitsPerComponent = 8
        let bytesPerRow = warmupSize * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        var pixelData = [UInt8](repeating: 128, count: warmupSize * warmupSize * 4)
        
        guard let context = CGContext(
            data: &pixelData,
            width: warmupSize,
            height: warmupSize,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ), let warmupImage = context.makeImage() else {
            fputs("Warning: Could not create warm-up image\n", stderr)
            return
        }
        
        // Run 3 warm-up passes to ensure full JIT compilation
        for i in 1...3 {
            let handler = VNImageRequestHandler(cgImage: warmupImage, options: [:])
            do {
                try handler.perform([self.request])
            } catch {
                fputs("Warm-up pass \(i) error (non-fatal): \(error.localizedDescription)\n", stderr)
            }
        }
        
        fputs("Warm-up complete\n", stderr)
    }
    
    /// Generate fast detection ID (faster than UUID)
    @inline(__always)
    private func generateDetectionId() -> String {
        detectionCounter &+= 1
        return "D\(detectionCounter)"
    }
    
    /// Optimized detection with nanosecond timing
    func detect(image: CGImage) -> (detections: [Detection], inferenceTimeNs: UInt64, preprocessTimeNs: UInt64, postprocessTimeNs: UInt64) {
        // Update cached timestamp once per detection batch
        cachedTimestamp = Int64(Date().timeIntervalSince1970 * 1000)
        
        // Clear and reuse detection buffer
        detectionBuffer.removeAll(keepingCapacity: true)
        
        let imageWidth = Double(image.width)
        let imageHeight = Double(image.height)
        
        // PREPROCESS: Create image handler
        let preprocessStart = getNanoseconds()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        let preprocessEnd = getNanoseconds()
        
        // INFERENCE: Synchronous execution for lowest latency
        let inferenceStart = getNanoseconds()
        do {
            try handler.perform([self.request])
        } catch {
            fputs("Vision request error: \(error.localizedDescription)\n", stderr)
            return ([], 0, preprocessEnd - preprocessStart, 0)
        }
        let inferenceEnd = getNanoseconds()
        
        // POSTPROCESS: Extract results
        let postprocessStart = getNanoseconds()
        
        guard let results = self.request.results as? [VNRecognizedObjectObservation] else {
            return ([], inferenceEnd - inferenceStart, preprocessEnd - preprocessStart, 0)
        }
        
        // Process observations with pre-filtering
        let threshold = self.confidenceThreshold
        var count = 0
        let maxCount = self.maxDetections
        
        for observation in results {
            guard count < maxCount else { break }
            guard observation.confidence >= threshold else { continue }
            guard let topLabel = observation.labels.first else { continue }
            
            let bbox = observation.boundingBox
            let x1 = bbox.origin.x * imageWidth
            let y1 = (1.0 - bbox.origin.y - bbox.height) * imageHeight
            let x2 = (bbox.origin.x + bbox.width) * imageWidth
            let y2 = (1.0 - bbox.origin.y) * imageHeight
            
            let identifier = topLabel.identifier
            let detection = Detection(
                id: generateDetectionId(),
                classLabel: mapToTacticalClass(identifier),
                classIndex: cocoClassIndexMap[identifier] ?? -1,
                confidence: Double(observation.confidence),
                bbox: BoundingBox(x1: x1, y1: y1, x2: x2, y2: y2),
                timestamp: cachedTimestamp
            )
            
            detectionBuffer.append(detection)
            count += 1
        }
        
        let postprocessEnd = getNanoseconds()
        
        return (
            Array(detectionBuffer),
            inferenceEnd - inferenceStart,
            preprocessEnd - preprocessStart,
            postprocessEnd - postprocessStart
        )
    }
}

// MARK: - Optimized Image Decoding

/// Fast base64 image decoding with minimal allocations
func decodeBase64ImageFast(_ base64String: String) -> CGImage? {
    var base64 = base64String
    
    // Fast prefix check and removal
    if base64.hasPrefix("data:") {
        if let commaIndex = base64.firstIndex(of: ",") {
            base64 = String(base64[base64.index(after: commaIndex)...])
        }
    }
    
    guard let imageData = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else {
        return nil
    }
    
    // Use ImageIO for fast decoding
    let options: [CFString: Any] = [
        kCGImageSourceShouldCache: true,
        kCGImageSourceShouldAllowFloat: false
    ]
    
    guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, options as CFDictionary),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, options as CFDictionary) else {
        return nil
    }
    
    return cgImage
}

// MARK: - Main

func main() {
    let totalStart = getNanoseconds()
    
    // Find model path - check bundle, relative paths, and environment variable
    let envModelPath = ProcessInfo.processInfo.environment["CREBAIN_MODEL_PATH"]
    let executableDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let possiblePaths: [String?] = [
        envModelPath,
        Bundle.main.resourcePath.map { "\($0)/yolov8s.mlmodelc" },
        executableDir.appendingPathComponent("../resources/yolov8s.mlmodelc").path,
        executableDir.appendingPathComponent("resources/yolov8s.mlmodelc").path
    ]
    let validPaths = possiblePaths.compactMap { $0 }
    
    var modelPath: String?
    for path in validPaths {
        if FileManager.default.fileExists(atPath: path) {
            modelPath = path
            break
        }
    }
    
    guard let finalModelPath = modelPath else {
        let errorResult = DetectionResult(
            success: false,
            detections: [],
            inferenceTimeMs: 0,
            preprocessTimeMs: nil,
            postprocessTimeMs: nil,
            error: "CoreML model not found"
        )
        let jsonData = try! JSONEncoder().encode(errorResult)
        print(String(data: jsonData, encoding: .utf8)!)
        exit(1)
    }
    
    fputs("Loading optimized CoreML model from: \(finalModelPath)\n", stderr)
    
    // Initialize optimized detector (includes warm-up)
    guard let detector = OptimizedCoreMLDetector(modelPath: finalModelPath) else {
        let errorResult = DetectionResult(
            success: false,
            detections: [],
            inferenceTimeMs: 0,
            preprocessTimeMs: nil,
            postprocessTimeMs: nil,
            error: "Failed to initialize detector"
        )
        let jsonData = try! JSONEncoder().encode(errorResult)
        print(String(data: jsonData, encoding: .utf8)!)
        exit(1)
    }
    
    let initTime = getNanoseconds() - totalStart
    fputs("Model initialized in \(nanosToMs(initTime))ms\n", stderr)
    
    // Read input from stdin
    guard let inputData = FileHandle.standardInput.availableData.isEmpty ? nil : FileHandle.standardInput.readDataToEndOfFile(),
          !inputData.isEmpty else {
        let errorResult = DetectionResult(
            success: false,
            detections: [],
            inferenceTimeMs: 0,
            preprocessTimeMs: nil,
            postprocessTimeMs: nil,
            error: "No input data received"
        )
        let jsonData = try! JSONEncoder().encode(errorResult)
        print(String(data: jsonData, encoding: .utf8)!)
        exit(1)
    }
    
    // Parse input JSON
    let payload: InputPayload
    do {
        payload = try JSONDecoder().decode(InputPayload.self, from: inputData)
    } catch {
        let errorResult = DetectionResult(
            success: false,
            detections: [],
            inferenceTimeMs: 0,
            preprocessTimeMs: nil,
            postprocessTimeMs: nil,
            error: "Failed to parse input JSON: \(error.localizedDescription)"
        )
        let jsonData = try! JSONEncoder().encode(errorResult)
        print(String(data: jsonData, encoding: .utf8)!)
        exit(1)
    }
    
    // Decode image with optimized decoder
    let decodeStart = getNanoseconds()
    guard let cgImage = decodeBase64ImageFast(payload.imageBase64) else {
        let errorResult = DetectionResult(
            success: false,
            detections: [],
            inferenceTimeMs: 0,
            preprocessTimeMs: nil,
            postprocessTimeMs: nil,
            error: "Failed to decode base64 image"
        )
        let jsonData = try! JSONEncoder().encode(errorResult)
        print(String(data: jsonData, encoding: .utf8)!)
        exit(1)
    }
    let decodeTime = getNanoseconds() - decodeStart
    fputs("Image decoded in \(nanosToMs(decodeTime))ms (\(cgImage.width)x\(cgImage.height))\n", stderr)
    
    // Run optimized detection
    let (detections, inferenceNs, preprocessNs, postprocessNs) = detector.detect(image: cgImage)
    
    fputs("Inference: \(nanosToMs(inferenceNs))ms, Preprocess: \(nanosToMs(preprocessNs))ms, Postprocess: \(nanosToMs(postprocessNs))ms\n", stderr)
    fputs("Detected \(detections.count) objects\n", stderr)
    
    // Output result
    let result = DetectionResult(
        success: true,
        detections: detections,
        inferenceTimeMs: nanosToMs(inferenceNs),
        preprocessTimeMs: nanosToMs(preprocessNs),
        postprocessTimeMs: nanosToMs(postprocessNs),
        error: nil
    )
    
    let jsonEncoder = JSONEncoder()
    let jsonData = try! jsonEncoder.encode(result)
    print(String(data: jsonData, encoding: .utf8)!)
}

main()
