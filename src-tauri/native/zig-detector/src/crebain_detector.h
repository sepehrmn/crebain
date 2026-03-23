/**
 * CREBAIN Cross-Platform ML Detector
 * C Header for Rust/FFI Integration
 */

#ifndef CREBAIN_DETECTOR_H
#define CREBAIN_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// Detection bounding box
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t class_index;
} CrebainDetection;

/// Detection result with timing
typedef struct {
    CrebainDetection* detections;
    int32_t count;
    uint64_t inference_time_ns;
    uint64_t preprocess_time_ns;
    uint64_t postprocess_time_ns;
    bool success;
    int32_t error_code;
} CrebainDetectionResult;

/// Backend type
typedef enum {
    CREBAIN_BACKEND_COREML = 0,
    CREBAIN_BACKEND_MPS = 1,
    CREBAIN_BACKEND_MLX = 2,
    CREBAIN_BACKEND_CPU = 3,
    CREBAIN_BACKEND_CUDA = 4,
    CREBAIN_BACKEND_TENSORRT = 5,
    CREBAIN_BACKEND_ONNX = 6,
    CREBAIN_BACKEND_UNKNOWN = -1,
} CrebainBackendType;

/// Detector configuration
typedef struct {
    const char* model_path;
    float confidence_threshold;
    float iou_threshold;
    int32_t max_detections;
    CrebainBackendType preferred_backend;
} CrebainDetectorConfig;

// ─────────────────────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize detector with configuration
/// @return 0 on success, negative error code on failure
int32_t crebain_init(const CrebainDetectorConfig* config);

/// Run detection on RGBA pixel buffer
/// @return 0 on success, negative error code on failure
int32_t crebain_detect(
    const uint8_t* pixels,
    int32_t width,
    int32_t height,
    int32_t bytes_per_row,
    CrebainDetectionResult* result
);

/// Get current backend type
CrebainBackendType crebain_get_backend(void);

/// Check if detector is ready
bool crebain_is_ready(void);

/// Get backend name string
const char* crebain_get_backend_name(void);

/// Cleanup resources
void crebain_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* CREBAIN_DETECTOR_H */
