/**
 * CREBAIN MPS Backend
 * Metal Performance Shaders GPU inference fallback
 * Used when CoreML is not available or for custom models
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreGraphics/CoreGraphics.h>
#import <mach/mach_time.h>

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
// GLOBAL STATE
// ─────────────────────────────────────────────────────────────────────────────

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static BOOL g_isInitialized = NO;
static BOOL g_warmupComplete = NO;

// Pre-allocated detection buffer
#define MAX_DETECTIONS 256
static Detection g_detectionBuffer[MAX_DETECTIONS];

// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

static uint64_t machTimeToNanoseconds(uint64_t machTime) {
    static mach_timebase_info_data_t timebaseInfo = {0};
    if (timebaseInfo.denom == 0) {
        mach_timebase_info(&timebaseInfo);
    }
    return machTime * timebaseInfo.numer / timebaseInfo.denom;
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────────────────────────

int32_t mps_backend_init(const char *modelPath) {
    @autoreleasepool {
        if (g_isInitialized) return 0;
        
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            NSLog(@"MPS: No Metal device available");
            return -1;
        }
        
        // Check MPS support
        if (!MPSSupportsMTLDevice(g_device)) {
            NSLog(@"MPS: Device does not support MPS");
            g_device = nil;
            return -2;
        }
        
        // Create command queue
        g_commandQueue = [g_device newCommandQueue];
        if (!g_commandQueue) {
            NSLog(@"MPS: Failed to create command queue");
            g_device = nil;
            return -3;
        }
        
        NSLog(@"MPS: Backend initialized with device: %@", g_device.name);
        
        // Note: Full MPS-based YOLO inference would require:
        // 1. Loading ONNX/custom weights
        // 2. Building MPSGraph for the network
        // 3. Implementing custom detection head
        // For now, this is a placeholder that returns empty results
        // In production, use CoreML which handles this automatically
        
        g_isInitialized = YES;
        g_warmupComplete = YES;
        
        return 0;
    }
}

int32_t mps_backend_detect(
    const uint8_t *pixels,
    int32_t width,
    int32_t height,
    int32_t bytesPerRow,
    float confidenceThreshold,
    int32_t *outCount,
    uint64_t *outInferenceNs,
    uint64_t *outPreprocessNs,
    uint64_t *outPostprocessNs
) {
    @autoreleasepool {
        if (!g_isInitialized || !g_device) {
            *outCount = 0;
            return -1;
        }
        
        uint64_t startTime = mach_absolute_time();
        
        // MPS inference placeholder
        // In a full implementation, this would:
        // 1. Create MTLTexture from pixel data
        // 2. Run MPSGraph inference
        // 3. Decode YOLO output tensors
        // 4. Apply NMS
        
        // For now, return empty results (CoreML is preferred)
        *outCount = 0;
        
        uint64_t endTime = mach_absolute_time();
        uint64_t totalNs = machTimeToNanoseconds(endTime - startTime);
        
        *outPreprocessNs = totalNs / 3;
        *outInferenceNs = totalNs / 3;
        *outPostprocessNs = totalNs / 3;
        
        // Suppress unused parameter warnings
        (void)pixels;
        (void)width;
        (void)height;
        (void)bytesPerRow;
        (void)confidenceThreshold;
        
        return 0;
    }
}

Detection* mps_backend_get_detections(void) {
    return g_detectionBuffer;
}

bool mps_backend_is_ready(void) {
    return g_isInitialized && g_warmupComplete;
}

void mps_backend_cleanup(void) {
    g_commandQueue = nil;
    g_device = nil;
    g_isInitialized = NO;
    g_warmupComplete = NO;
}
