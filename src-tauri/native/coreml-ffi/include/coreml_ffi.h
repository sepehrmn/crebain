/**
 * CREBAIN CoreML FFI Header
 * C ABI interface for Rust FFI calls
 */

#ifndef COREML_FFI_H
#define COREML_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Detection bounding box and confidence
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t class_index;
} CDetection;

/// Result structure from detection
typedef struct {
    CDetection* detections;
    int32_t count;
    uint64_t inference_time_ns;
    uint64_t preprocess_time_ns;
    uint64_t postprocess_time_ns;
    bool success;
    int32_t error_code;
} CDetectionResult;

/// Initialize CoreML model - call once at startup
/// @param model_path Path to .mlmodelc directory
/// @return 0 on success, negative error code on failure
int32_t coreml_init(const char* model_path);

/// Run detection on RGBA pixel buffer
/// @param pixels Raw RGBA pixel data
/// @param width Image width
/// @param height Image height
/// @param bytes_per_row Bytes per row (usually width * 4)
/// @param confidence_threshold Minimum confidence (0.0-1.0)
/// @param result Output result structure
/// @return 0 on success, negative error code on failure
int32_t coreml_detect(
    const uint8_t* pixels,
    int32_t width,
    int32_t height,
    int32_t bytes_per_row,
    float confidence_threshold,
    CDetectionResult* result
);

/// Check if CoreML is initialized and ready
/// @return true if ready for detection
bool coreml_is_ready(void);

/// Get expected model input size
/// @param width Output width
/// @param height Output height
void coreml_get_input_size(int32_t* width, int32_t* height);

/// Cleanup and release resources
void coreml_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* COREML_FFI_H */
