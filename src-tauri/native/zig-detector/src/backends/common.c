/**
 * CREBAIN Common Backend Utilities
 * Cross-platform helper functions for ML detection
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Detection structure (must match Zig definition)
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t class_index;
} Detection;

// ─────────────────────────────────────────────────────────────────────────────
// INTERSECTION OVER UNION (IoU)
// ─────────────────────────────────────────────────────────────────────────────

static inline float max_f(float a, float b) { return a > b ? a : b; }
static inline float min_f(float a, float b) { return a < b ? a : b; }

static float compute_iou(const Detection* a, const Detection* b) {
    float inter_x1 = max_f(a->x1, b->x1);
    float inter_y1 = max_f(a->y1, b->y1);
    float inter_x2 = min_f(a->x2, b->x2);
    float inter_y2 = min_f(a->y2, b->y2);
    
    float inter_width = max_f(0.0f, inter_x2 - inter_x1);
    float inter_height = max_f(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;
    
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    float union_area = area_a + area_b - inter_area;
    
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// ─────────────────────────────────────────────────────────────────────────────
// NON-MAXIMUM SUPPRESSION (NMS)
// ─────────────────────────────────────────────────────────────────────────────

// Comparison function for qsort (descending confidence)
static int compare_detections(const void* a, const void* b) {
    const Detection* det_a = (const Detection*)a;
    const Detection* det_b = (const Detection*)b;
    if (det_b->confidence > det_a->confidence) return 1;
    if (det_b->confidence < det_a->confidence) return -1;
    return 0;
}

/**
 * Apply Non-Maximum Suppression to filter overlapping detections
 * @param detections Array of detections (modified in place)
 * @param count Number of detections
 * @param iou_threshold IoU threshold for suppression
 * @param out_count Output: number of remaining detections
 */
void common_nms(Detection* detections, int32_t count, float iou_threshold, int32_t* out_count) {
    if (count <= 0) {
        *out_count = 0;
        return;
    }
    
    if (count == 1) {
        *out_count = 1;
        return;
    }
    
    // Sort by confidence (descending)
    qsort(detections, count, sizeof(Detection), compare_detections);
    
    // Track which detections to keep
    bool* keep = (bool*)calloc(count, sizeof(bool));
    if (!keep) {
        *out_count = count; // Fallback: keep all
        return;
    }
    
    for (int32_t i = 0; i < count; i++) {
        keep[i] = true;
    }
    
    // NMS algorithm
    for (int32_t i = 0; i < count; i++) {
        if (!keep[i]) continue;
        
        for (int32_t j = i + 1; j < count; j++) {
            if (!keep[j]) continue;
            
            // Only suppress same-class detections
            if (detections[i].class_index != detections[j].class_index) continue;
            
            float iou = compute_iou(&detections[i], &detections[j]);
            if (iou > iou_threshold) {
                keep[j] = false; // Suppress lower-confidence detection
            }
        }
    }
    
    // Compact the array
    int32_t write_idx = 0;
    for (int32_t i = 0; i < count; i++) {
        if (keep[i]) {
            if (write_idx != i) {
                detections[write_idx] = detections[i];
            }
            write_idx++;
        }
    }
    
    free(keep);
    *out_count = write_idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD UTILITIES (for future CPU backend)
// ─────────────────────────────────────────────────────────────────────────────

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

// NEON-optimized softmax (for post-processing)
void neon_softmax(float* data, int32_t size) {
    if (size <= 0) return;
    
    // Find max for numerical stability
    float32x4_t max_vec = vdupq_n_f32(-1e9f);
    int32_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        max_vec = vmaxq_f32(max_vec, v);
    }
    float max_val = vmaxvq_f32(max_vec);
    for (; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
        for (i = 0; i + 4 <= size; i += 4) {
            float32x4_t v = vld1q_f32(&data[i]);
            vst1q_f32(&data[i], vmulq_f32(v, inv_sum_vec));
        }
        for (; i < size; i++) {
            data[i] *= inv_sum;
        }
    }
}

#else

// Scalar fallback
void neon_softmax(float* data, int32_t size) {
    if (size <= 0) return;
    
    float max_val = data[0];
    for (int32_t i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    float sum = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    
    if (sum > 0.0f) {
        for (int32_t i = 0; i < size; i++) {
            data[i] /= sum;
        }
    }
}

#endif
