///! CREBAIN Cross-Platform ML Detector
///! Zero-latency object detection with platform-specific backends
///!
///! Backends:
///! - CoreML (macOS): Native Apple ML framework via Neural Engine
///! - MPS (macOS): Metal Performance Shaders for GPU compute
///! - CUDA (Linux/Windows): NVIDIA GPU via cuDNN/ONNX Runtime
///! - CPU (All): Fallback SIMD-optimized inference
const std = @import("std");
const builtin = @import("builtin");

// ─────────────────────────────────────────────────────────────────────────────
// DETECTION TYPES (C ABI Compatible)
// ─────────────────────────────────────────────────────────────────────────────

/// Bounding box detection result
pub const Detection = extern struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_index: i32,
};

/// Detection result with timing information
pub const DetectionResult = extern struct {
    detections: [*]Detection,
    count: i32,
    inference_time_ns: u64,
    preprocess_time_ns: u64,
    postprocess_time_ns: u64,
    success: bool,
    error_code: i32,
};

/// Backend type enum
pub const BackendType = enum(i32) {
    CoreML = 0,
    MPS = 1,
    MLX = 2,
    CPU = 3,
    CUDA = 4,
    TensorRT = 5,
    ONNX = 6,
    Unknown = -1,
};

/// Detector configuration
pub const DetectorConfig = extern struct {
    model_path: [*:0]const u8,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: i32,
    preferred_backend: BackendType,
};

// ─────────────────────────────────────────────────────────────────────────────
// BACKEND INTERFACE (Extern C functions from platform-specific sources)
// ─────────────────────────────────────────────────────────────────────────────

// CoreML backend (macOS only)
extern fn coreml_backend_init(model_path: [*:0]const u8) i32;
extern fn coreml_backend_detect(
    pixels: [*]const u8,
    width: i32,
    height: i32,
    bytes_per_row: i32,
    confidence_threshold: f32,
    out_count: *i32,
    out_inference_ns: *u64,
    out_preprocess_ns: *u64,
    out_postprocess_ns: *u64,
) i32;
extern fn coreml_backend_get_detections() [*]Detection;
extern fn coreml_backend_cleanup() void;
extern fn coreml_backend_is_ready() bool;

// MPS backend (macOS GPU fallback)
extern fn mps_backend_init(model_path: [*:0]const u8) i32;
extern fn mps_backend_detect(
    pixels: [*]const u8,
    width: i32,
    height: i32,
    bytes_per_row: i32,
    confidence_threshold: f32,
    out_count: *i32,
    out_inference_ns: *u64,
    out_preprocess_ns: *u64,
    out_postprocess_ns: *u64,
) i32;
extern fn mps_backend_get_detections() [*]Detection;
extern fn mps_backend_cleanup() void;
extern fn mps_backend_is_ready() bool;

// CUDA backend (Linux/Windows NVIDIA GPU)
extern fn cuda_backend_init(model_path: [*:0]const u8) i32;
extern fn cuda_backend_detect(
    pixels: [*]const u8,
    width: i32,
    height: i32,
    bytes_per_row: i32,
    confidence_threshold: f32,
    out_count: *i32,
    out_inference_ns: *u64,
    out_preprocess_ns: *u64,
    out_postprocess_ns: *u64,
) i32;
extern fn cuda_backend_get_detections() [*]Detection;
extern fn cuda_backend_cleanup() void;
extern fn cuda_backend_is_ready() bool;

// Common utilities
extern fn common_nms(
    detections: [*]Detection,
    count: i32,
    iou_threshold: f32,
    out_count: *i32,
) void;

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL STATE
// ─────────────────────────────────────────────────────────────────────────────

var current_backend: BackendType = .Unknown;
var is_initialized: bool = false;
var config: DetectorConfig = undefined;

// Pre-allocated detection buffer
const MAX_DETECTIONS = 256;
var detection_buffer: [MAX_DETECTIONS]Detection = undefined;

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC API (C ABI Exports)
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize the detector with the given configuration
/// Returns 0 on success, negative error code on failure
export fn crebain_init(cfg: *const DetectorConfig) i32 {
    if (is_initialized) {
        return 0; // Already initialized
    }

    config = cfg.*;

    // Try backends in order of preference
    const backends_to_try = switch (cfg.preferred_backend) {
        .CoreML => &[_]BackendType{ .CoreML, .MPS, .CPU },
        .MPS => &[_]BackendType{ .MPS, .CoreML, .CPU },
        .MLX => &[_]BackendType{ .MLX, .CoreML, .MPS, .CPU },
        .CUDA => &[_]BackendType{ .CUDA, .CPU },
        // TensorRT/ONNX are accepted for ABI compatibility, but currently
        // map to the CUDA backend (TensorRT EP not implemented here).
        .TensorRT => &[_]BackendType{ .CUDA, .CPU },
        .ONNX => &[_]BackendType{ .CUDA, .CPU },
        .CPU => &[_]BackendType{.CPU},
        .Unknown => if (comptime builtin.os.tag == .linux)
            &[_]BackendType{ .CUDA, .CPU }
        else
            &[_]BackendType{ .CoreML, .MPS, .CPU },
    };

    for (backends_to_try) |backend| {
        const result = initBackend(backend, cfg.model_path);
        if (result == 0) {
            current_backend = backend;
            is_initialized = true;
            return 0;
        }
    }

    return -1; // All backends failed
}

/// Run detection on raw RGBA pixel data
export fn crebain_detect(
    pixels: [*]const u8,
    width: i32,
    height: i32,
    bytes_per_row: i32,
    result: *DetectionResult,
) i32 {
    if (!is_initialized) {
        result.success = false;
        result.error_code = -1;
        result.count = 0;
        return -1;
    }

    var count: i32 = 0;
    var inference_ns: u64 = 0;
    var preprocess_ns: u64 = 0;
    var postprocess_ns: u64 = 0;

    const ret = switch (current_backend) {
        .CoreML => coreml_backend_detect(
            pixels,
            width,
            height,
            bytes_per_row,
            config.confidence_threshold,
            &count,
            &inference_ns,
            &preprocess_ns,
            &postprocess_ns,
        ),
        .MPS => mps_backend_detect(
            pixels,
            width,
            height,
            bytes_per_row,
            config.confidence_threshold,
            &count,
            &inference_ns,
            &preprocess_ns,
            &postprocess_ns,
        ),
        .CUDA, .TensorRT, .ONNX => cuda_backend_detect(
            pixels,
            width,
            height,
            bytes_per_row,
            config.confidence_threshold,
            &count,
            &inference_ns,
            &preprocess_ns,
            &postprocess_ns,
        ),
        else => -2, // Unsupported backend
    };

    if (ret != 0) {
        result.success = false;
        result.error_code = ret;
        result.count = 0;
        return ret;
    }

    // Get detections from backend
    const backend_detections = switch (current_backend) {
        .CoreML => coreml_backend_get_detections(),
        .MPS => mps_backend_get_detections(),
        .CUDA, .TensorRT, .ONNX => cuda_backend_get_detections(),
        else => &detection_buffer,
    };

    // Copy to our buffer and apply NMS if needed
    const copy_count = @min(count, MAX_DETECTIONS);
    @memcpy(detection_buffer[0..@intCast(copy_count)], backend_detections[0..@intCast(copy_count)]);

    // Apply NMS
    var nms_count: i32 = 0;
    common_nms(&detection_buffer, copy_count, config.iou_threshold, &nms_count);

    // Limit to max detections
    const final_count = @min(nms_count, config.max_detections);

    result.detections = &detection_buffer;
    result.count = final_count;
    result.inference_time_ns = inference_ns;
    result.preprocess_time_ns = preprocess_ns;
    result.postprocess_time_ns = postprocess_ns;
    result.success = true;
    result.error_code = 0;

    return 0;
}

/// Get the current backend type
export fn crebain_get_backend() BackendType {
    return current_backend;
}

/// Check if detector is ready
export fn crebain_is_ready() bool {
    return is_initialized and switch (current_backend) {
        .CoreML => coreml_backend_is_ready(),
        .MPS => mps_backend_is_ready(),
        .CUDA, .TensorRT, .ONNX => cuda_backend_is_ready(),
        else => false,
    };
}

/// Get backend name as string
export fn crebain_get_backend_name() [*:0]const u8 {
    return switch (current_backend) {
        .CoreML => "CoreML (Neural Engine)",
        .MPS => "MPS (Metal GPU)",
        .MLX => "MLX (Apple Silicon)",
        .CPU => "CPU (SIMD)",
        .CUDA => "CUDA (NVIDIA GPU)",
        .TensorRT => "TensorRT (NVIDIA)",
        .ONNX => "ONNX Runtime",
        .Unknown => "Not Initialized",
    };
}

/// Cleanup and release resources
export fn crebain_cleanup() void {
    if (!is_initialized) return;

    switch (current_backend) {
        .CoreML => coreml_backend_cleanup(),
        .MPS => mps_backend_cleanup(),
        .CUDA, .TensorRT, .ONNX => cuda_backend_cleanup(),
        else => {},
    }

    is_initialized = false;
    current_backend = .Unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// INTERNAL HELPERS
// ─────────────────────────────────────────────────────────────────────────────

fn initBackend(backend: BackendType, model_path: [*:0]const u8) i32 {
    return switch (backend) {
        .CoreML => if (comptime builtin.os.tag == .macos) coreml_backend_init(model_path) else -1,
        .MPS => if (comptime builtin.os.tag == .macos) mps_backend_init(model_path) else -1,
        .CUDA, .TensorRT, .ONNX => if (comptime builtin.os.tag == .linux or builtin.os.tag == .windows) cuda_backend_init(model_path) else -1,
        .MLX => -1, // Not yet implemented
        .CPU => -1, // Not yet implemented
        .Unknown => -1,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

test "detection struct size" {
    try std.testing.expectEqual(@sizeOf(Detection), 24);
}

test "backend enum values" {
    try std.testing.expectEqual(@intFromEnum(BackendType.CoreML), 0);
    try std.testing.expectEqual(@intFromEnum(BackendType.MPS), 1);
}
