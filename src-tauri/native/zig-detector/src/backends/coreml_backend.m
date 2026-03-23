/**
 * CREBAIN CoreML Backend
 * Native CoreML inference via Vision framework
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
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

static VNCoreMLModel *g_vnModel = nil;
static BOOL g_isInitialized = NO;
static BOOL g_warmupComplete = NO;

// Pre-allocated detection buffer
#define MAX_DETECTIONS 256
static Detection g_detectionBuffer[MAX_DETECTIONS];

// COCO class names for mapping
static NSArray<NSString*> *g_cocoClasses = nil;

// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

static void initCocoClasses(void) {
    if (g_cocoClasses) return;
    g_cocoClasses = @[
        @"person", @"bicycle", @"car", @"motorcycle", @"airplane", @"bus", @"train", @"truck", @"boat",
        @"traffic light", @"fire hydrant", @"stop sign", @"parking meter", @"bench", @"bird", @"cat",
        @"dog", @"horse", @"sheep", @"cow", @"elephant", @"bear", @"zebra", @"giraffe", @"backpack",
        @"umbrella", @"handbag", @"tie", @"suitcase", @"frisbee", @"skis", @"snowboard", @"sports ball",
        @"kite", @"baseball bat", @"baseball glove", @"skateboard", @"surfboard", @"tennis racket",
        @"bottle", @"wine glass", @"cup", @"fork", @"knife", @"spoon", @"bowl", @"banana", @"apple",
        @"sandwich", @"orange", @"broccoli", @"carrot", @"hot dog", @"pizza", @"donut", @"cake", @"chair",
        @"couch", @"potted plant", @"bed", @"dining table", @"toilet", @"tv", @"laptop", @"mouse",
        @"remote", @"keyboard", @"cell phone", @"microwave", @"oven", @"toaster", @"sink", @"refrigerator",
        @"book", @"clock", @"vase", @"scissors", @"teddy bear", @"hair drier", @"toothbrush"
    ];
}

static int32_t getClassIndex(NSString *identifier) {
    if (!g_cocoClasses) initCocoClasses();
    NSUInteger idx = [g_cocoClasses indexOfObject:[identifier lowercaseString]];
    return (idx == NSNotFound) ? -1 : (int32_t)idx;
}

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

int32_t coreml_backend_init(const char *modelPath) {
    @autoreleasepool {
        if (g_isInitialized) return 0;
        
        initCocoClasses();
        
        NSString *path = [NSString stringWithUTF8String:modelPath];
        NSURL *modelURL = [NSURL fileURLWithPath:path];
        
        // Configure for maximum performance
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        config.allowLowPrecisionAccumulationOnGPU = YES;
        
        NSError *error = nil;
        MLModel *mlModel = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        
        if (!mlModel || error) {
            NSLog(@"CoreML: Failed to load model: %@", error.localizedDescription);
            return -1;
        }
        
        g_vnModel = [VNCoreMLModel modelForMLModel:mlModel error:&error];
        if (!g_vnModel || error) {
            NSLog(@"CoreML: Failed to create VNCoreMLModel: %@", error.localizedDescription);
            return -2;
        }
        
        // Warmup passes
        NSLog(@"CoreML: Warming up model...");
        uint8_t warmupData[64 * 64 * 4];
        memset(warmupData, 128, sizeof(warmupData));
        
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(warmupData, 64, 64, 8, 64 * 4, colorSpace,
                                                  kCGImageAlphaPremultipliedLast);
        CGImageRef warmupImage = CGBitmapContextCreateImage(ctx);
        
        for (int i = 0; i < 3; i++) {
            VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:g_vnModel];
            request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
            
            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:warmupImage options:@{}];
            [handler performRequests:@[request] error:nil];
        }
        
        CGImageRelease(warmupImage);
        CGContextRelease(ctx);
        CGColorSpaceRelease(colorSpace);
        
        g_isInitialized = YES;
        g_warmupComplete = YES;
        NSLog(@"CoreML: Backend initialized successfully");
        
        return 0;
    }
}

int32_t coreml_backend_detect(
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
        if (!g_isInitialized || !g_vnModel) {
            *outCount = 0;
            return -1;
        }
        
        uint64_t preprocessStart = mach_absolute_time();
        
        // Create CGImage from pixel buffer
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate((void *)pixels, width, height, 8, bytesPerRow, colorSpace,
                                                  kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        
        if (!ctx) {
            CGColorSpaceRelease(colorSpace);
            *outCount = 0;
            return -2;
        }
        
        CGImageRef cgImage = CGBitmapContextCreateImage(ctx);
        CGContextRelease(ctx);
        CGColorSpaceRelease(colorSpace);
        
        if (!cgImage) {
            *outCount = 0;
            return -3;
        }
        
        uint64_t preprocessEnd = mach_absolute_time();
        
        // Run inference
        uint64_t inferenceStart = mach_absolute_time();
        
        __block NSArray<VNRecognizedObjectObservation *> *results = nil;
        
        VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:g_vnModel completionHandler:^(VNRequest *req, NSError *err) {
            if (!err) {
                results = req.results;
            }
        }];
        request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
        request.preferBackgroundProcessing = NO;
        
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
        NSError *error = nil;
        [handler performRequests:@[request] error:&error];
        
        CGImageRelease(cgImage);
        
        uint64_t inferenceEnd = mach_absolute_time();
        
        if (error) {
            *outCount = 0;
            return -4;
        }
        
        // Post-process results
        uint64_t postprocessStart = mach_absolute_time();
        
        int32_t count = 0;
        float imgWidth = (float)width;
        float imgHeight = (float)height;
        
        for (VNRecognizedObjectObservation *obs in results) {
            if (count >= MAX_DETECTIONS) break;
            if (obs.confidence < confidenceThreshold) continue;
            
            CGRect bbox = obs.boundingBox;
            
            // Convert Vision coordinates (origin bottom-left) to pixel coords (origin top-left)
            g_detectionBuffer[count].x1 = bbox.origin.x * imgWidth;
            g_detectionBuffer[count].y1 = (1.0f - bbox.origin.y - bbox.size.height) * imgHeight;
            g_detectionBuffer[count].x2 = (bbox.origin.x + bbox.size.width) * imgWidth;
            g_detectionBuffer[count].y2 = (1.0f - bbox.origin.y) * imgHeight;
            g_detectionBuffer[count].confidence = obs.confidence;
            
            // Get class index
            if (obs.labels.count > 0) {
                g_detectionBuffer[count].class_index = getClassIndex(obs.labels[0].identifier);
            } else {
                g_detectionBuffer[count].class_index = -1;
            }
            
            count++;
        }
        
        uint64_t postprocessEnd = mach_absolute_time();
        
        *outCount = count;
        *outPreprocessNs = machTimeToNanoseconds(preprocessEnd - preprocessStart);
        *outInferenceNs = machTimeToNanoseconds(inferenceEnd - inferenceStart);
        *outPostprocessNs = machTimeToNanoseconds(postprocessEnd - postprocessStart);
        
        return 0;
    }
}

Detection* coreml_backend_get_detections(void) {
    return g_detectionBuffer;
}

bool coreml_backend_is_ready(void) {
    return g_isInitialized && g_warmupComplete;
}

void coreml_backend_cleanup(void) {
    g_vnModel = nil;
    g_isInitialized = NO;
    g_warmupComplete = NO;
}
