/**
 * CREBAIN CUDA Backend
 * GPU-accelerated inference for Linux/NVIDIA systems
 * 
 * Uses ONNX Runtime with CUDA Execution Provider for model inference
 * Preprocessing and postprocessing done with custom CUDA kernels
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef CREBAIN_CUDA

#include <cuda_runtime.h>

// Check if ONNX Runtime is available
#ifdef CREBAIN_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

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

static bool g_isInitialized = false;
static bool g_warmupComplete = false;

// Pre-allocated detection buffer
#define MAX_DETECTIONS 256
static Detection g_detectionBuffer[MAX_DETECTIONS];

// Input/output buffers on GPU
static float *g_deviceInput = NULL;
static size_t g_inputSize = 0;
static size_t g_outputSize = 0;

// Model parameters
static int32_t g_inputWidth = 640;
static int32_t g_inputHeight = 640;
static int32_t g_numClasses = 80;
static int32_t g_numAnchors = 8400;

// Timing
static cudaEvent_t g_startEvent = NULL;
static cudaEvent_t g_stopEvent = NULL;

// Forward declaration (used for early-init cleanup paths).
void cuda_backend_cleanup(void);

#ifdef CREBAIN_ONNXRUNTIME
// ONNX Runtime state
static const OrtApi* g_ort = NULL;
static OrtEnv* g_ortEnv = NULL;
static OrtSession* g_ortSession = NULL;
static OrtSessionOptions* g_ortSessionOptions = NULL;
static OrtMemoryInfo* g_ortMemoryInfo = NULL;
static char* g_inputName = NULL;
static char* g_outputName = NULL;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// CUDA ERROR HANDLING
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK_GOTO(call, label) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            goto label; \
        } \
    } while(0)

#ifdef CREBAIN_ONNXRUNTIME
#define ORT_CHECK_GOTO(call, label) \
    do { \
        OrtStatus* status = call; \
        if (status != NULL) { \
            const char* msg = (g_ort != NULL) ? g_ort->GetErrorMessage(status) : "Unknown"; \
            fprintf(stderr, "ONNX Runtime error at %s:%d: %s\n", __FILE__, __LINE__, msg); \
            if (g_ort != NULL) { \
                g_ort->ReleaseStatus(status); \
            } \
            goto label; \
        } \
    } while(0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSING (CPU fallback - CUDA kernel version available when nvcc used)
// ─────────────────────────────────────────────────────────────────────────────

// CPU preprocessing: RGBA to normalized RGB with resize
static void preprocess_cpu(
    const uint8_t* input,
    float* output,
    int srcWidth,
    int srcHeight,
    int srcBytesPerRow,
    int dstWidth,
    int dstHeight
) {
    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x++) {
            // Calculate source coordinates (nearest neighbor for simplicity)
            int srcX = (int)((float)x * srcWidth / dstWidth);
            int srcY = (int)((float)y * srcHeight / dstHeight);
            
            srcX = srcX < srcWidth ? srcX : srcWidth - 1;
            srcY = srcY < srcHeight ? srcY : srcHeight - 1;
            
            int srcIdx = srcY * srcBytesPerRow + srcX * 4;
            
            // NCHW format: output[c * H * W + y * W + x]
            for (int c = 0; c < 3; c++) {
                output[c * dstHeight * dstWidth + y * dstWidth + x] = 
                    (float)input[srcIdx + c] / 255.0f;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────────────────────────

int32_t cuda_backend_init(const char *modelPath) {
    if (g_isInitialized) return 0;
    int32_t rc = -1;
    
    // Check CUDA device availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA: No CUDA-capable devices found\n");
        return -1;
    }
    
    // Select best device (most SMs)
    int bestDevice = 0;
    cudaDeviceProp bestProps;
    CUDA_CHECK_GOTO(cudaGetDeviceProperties(&bestProps, 0), fail);
    
    for (int i = 1; i < deviceCount; i++) {
        cudaDeviceProp props;
        CUDA_CHECK_GOTO(cudaGetDeviceProperties(&props, i), fail);
        if (props.multiProcessorCount > bestProps.multiProcessorCount) {
            bestDevice = i;
            bestProps = props;
        }
    }
    
    CUDA_CHECK_GOTO(cudaSetDevice(bestDevice), fail);
    printf("CUDA: Using device %d: %s (SM %d.%d, %d MPs)\n", 
           bestDevice, bestProps.name, bestProps.major, bestProps.minor,
           bestProps.multiProcessorCount);
    
    // Create timing events
    CUDA_CHECK_GOTO(cudaEventCreate(&g_startEvent), fail);
    CUDA_CHECK_GOTO(cudaEventCreate(&g_stopEvent), fail);
    
    // Allocate input buffer (NCHW format: 1 x 3 x 640 x 640)
    g_inputSize = 1 * 3 * g_inputHeight * g_inputWidth * sizeof(float);
    CUDA_CHECK_GOTO(cudaMalloc(&g_deviceInput, g_inputSize), fail);
    
    // Allocate output buffer (YOLOv8: 1 x 84 x 8400 for 640x640 input)
    g_outputSize = 1 * (4 + g_numClasses) * g_numAnchors * sizeof(float);
    
#ifdef CREBAIN_ONNXRUNTIME
    // Initialize ONNX Runtime
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "CUDA: Failed to get ONNX Runtime API\n");
        goto fail;
    }
    
    // Create environment
    ORT_CHECK_GOTO(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "crebain", &g_ortEnv), fail);
    
    // Create session options
    ORT_CHECK_GOTO(g_ort->CreateSessionOptions(&g_ortSessionOptions), fail);
    ORT_CHECK_GOTO(g_ort->SetIntraOpNumThreads(g_ortSessionOptions, 1), fail);
    ORT_CHECK_GOTO(g_ort->SetSessionGraphOptimizationLevel(g_ortSessionOptions, ORT_ENABLE_ALL), fail);
    
    // Add CUDA execution provider
    OrtCUDAProviderOptions cudaOptions;
    memset(&cudaOptions, 0, sizeof(cudaOptions));
    cudaOptions.device_id = bestDevice;
    cudaOptions.arena_extend_strategy = 0;
    cudaOptions.gpu_mem_limit = 0;  // No limit
    cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cudaOptions.do_copy_in_default_stream = 1;
    
    ORT_CHECK_GOTO(g_ort->SessionOptionsAppendExecutionProvider_CUDA(g_ortSessionOptions, &cudaOptions), fail);
    
    // Load the model
    printf("CUDA: Loading model from %s\n", modelPath);
    ORT_CHECK_GOTO(g_ort->CreateSession(g_ortEnv, modelPath, g_ortSessionOptions, &g_ortSession), fail);
    
    // Create memory info for CUDA
    ORT_CHECK_GOTO(g_ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, bestDevice, OrtMemTypeDefault, &g_ortMemoryInfo), fail);
    
    // Get input/output names
    OrtAllocator* allocator;
    ORT_CHECK_GOTO(g_ort->GetAllocatorWithDefaultOptions(&allocator), fail);
    
    ORT_CHECK_GOTO(g_ort->SessionGetInputName(g_ortSession, 0, allocator, &g_inputName), fail);
    ORT_CHECK_GOTO(g_ort->SessionGetOutputName(g_ortSession, 0, allocator, &g_outputName), fail);
    
    printf("CUDA: Model loaded - Input: %s, Output: %s\n", g_inputName, g_outputName);
#else
    fprintf(stderr, "CUDA: ONNX Runtime not enabled in this build - CUDA backend unavailable\n");
    fprintf(stderr, "CUDA: Rebuild with `zig build -Dcuda=true -Donnx=true -Donnx-path=...`\n");
    fprintf(stderr, "CUDA: Model path: %s (not loaded)\n", modelPath);
    cuda_backend_cleanup();
    return -1;
#endif
    
    // Warmup
    printf("CUDA: Warming up...\n");
    CUDA_CHECK_GOTO(cudaMemset(g_deviceInput, 0, g_inputSize), fail);
    CUDA_CHECK_GOTO(cudaDeviceSynchronize(), fail);
    
    g_isInitialized = true;
    g_warmupComplete = true;
    printf("CUDA: Backend initialized successfully\n");
    
    return 0;

fail:
    cuda_backend_cleanup();
    return rc;
}

int32_t cuda_backend_detect(
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
    if (!g_isInitialized) {
        *outCount = 0;
        *outPreprocessNs = 0;
        *outInferenceNs = 0;
        *outPostprocessNs = 0;
        return -1;
    }
    int32_t rc = -1;
    
    float preprocessMs = 0, inferenceMs = 0, postprocessMs = 0;
    
    // Allocate host input buffer
    float *hostInput = NULL;
    float *hostOutput = NULL;
    hostInput = (float*)malloc(g_inputSize);
    if (!hostInput) {
        *outCount = 0;
        *outPreprocessNs = 0;
        *outInferenceNs = 0;
        *outPostprocessNs = 0;
        return -2;
    }
    
    // ─── PREPROCESS ───
    CUDA_CHECK_GOTO(cudaEventRecord(g_startEvent), cleanup);
    
    // CPU preprocessing (resize + normalize)
    preprocess_cpu(pixels, hostInput, width, height, bytesPerRow, g_inputWidth, g_inputHeight);
    
    // Copy to GPU
    CUDA_CHECK_GOTO(cudaMemcpy(g_deviceInput, hostInput, g_inputSize, cudaMemcpyHostToDevice), cleanup);
    
    CUDA_CHECK_GOTO(cudaEventRecord(g_stopEvent), cleanup);
    CUDA_CHECK_GOTO(cudaEventSynchronize(g_stopEvent), cleanup);
    CUDA_CHECK_GOTO(cudaEventElapsedTime(&preprocessMs, g_startEvent, g_stopEvent), cleanup);
    
    // ─── INFERENCE ───
    CUDA_CHECK_GOTO(cudaEventRecord(g_startEvent), cleanup);
    
#ifdef CREBAIN_ONNXRUNTIME
    // Create input tensor on GPU
    int64_t inputShape[] = {1, 3, g_inputHeight, g_inputWidth};
    OrtValue* inputTensor = NULL;
    OrtValue* outputTensor = NULL;
    OrtMemoryInfo* outputMemInfo = NULL;
    const char* outputAllocatorName = NULL;

    ORT_CHECK_GOTO(g_ort->CreateTensorWithDataAsOrtValue(
        g_ortMemoryInfo,
        g_deviceInput,
        g_inputSize,
        inputShape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputTensor
    ), cleanup_ort);
    
    // Run inference
    const char* inputNames[] = {g_inputName};
    const char* outputNames[] = {g_outputName};

    OrtStatus* runStatus = g_ort->Run(
        g_ortSession,
        NULL,  // Run options
        inputNames,
        (const OrtValue* const*)&inputTensor,
        1,
        outputNames,
        1,
        &outputTensor
    );
    
    if (inputTensor) {
        g_ort->ReleaseValue(inputTensor);
        inputTensor = NULL;
    }
    
    if (runStatus != NULL) {
        const char* msg = g_ort->GetErrorMessage(runStatus);
        fprintf(stderr, "CUDA: Inference failed: %s\n", msg);
        g_ort->ReleaseStatus(runStatus);
        *outCount = 0;
        rc = -3;
        goto cleanup_ort;
    }
    
    // Get output data
    float* outputData = NULL;
    ORT_CHECK_GOTO(g_ort->GetTensorMutableData(outputTensor, (void**)&outputData), cleanup_ort);
    
    // Copy output to our buffer if on GPU
    ORT_CHECK_GOTO(g_ort->GetTensorMemoryInfo(outputTensor, &outputMemInfo), cleanup_ort);
    ORT_CHECK_GOTO(g_ort->MemoryInfoGetName(outputMemInfo, &outputAllocatorName), cleanup_ort);

    hostOutput = (float*)malloc(g_outputSize);
    if (!hostOutput) {
        *outCount = 0;
        rc = -2;
        goto cleanup_ort;
    }
    if (outputAllocatorName && strcmp(outputAllocatorName, "Cuda") == 0) {
        CUDA_CHECK_GOTO(cudaMemcpy(hostOutput, outputData, g_outputSize, cudaMemcpyDeviceToHost), cleanup_ort);
    } else {
        memcpy(hostOutput, outputData, g_outputSize);
    }

    if (outputMemInfo) {
        g_ort->ReleaseMemoryInfo(outputMemInfo);
        outputMemInfo = NULL;
    }
    if (outputTensor) {
        g_ort->ReleaseValue(outputTensor);
        outputTensor = NULL;
    }
#else
    // No ONNX Runtime - produce empty output
    hostOutput = (float*)calloc(1, g_outputSize);
#endif
    
    CUDA_CHECK_GOTO(cudaEventRecord(g_stopEvent), cleanup);
    CUDA_CHECK_GOTO(cudaEventSynchronize(g_stopEvent), cleanup);
    CUDA_CHECK_GOTO(cudaEventElapsedTime(&inferenceMs, g_startEvent, g_stopEvent), cleanup);
    
    // ─── POSTPROCESS ───
    CUDA_CHECK_GOTO(cudaEventRecord(g_startEvent), cleanup);
    
    // Decode YOLOv8 output: [1, 84, 8400]
    // Format: [4 box coords + 80 class scores] x 8400 anchors
    //
    // Ultralytics-style YOLO exports emit bbox coords in model-input pixels
    // (0..g_inputWidth/Height). Scale back to original image dimensions.
    int32_t count = 0;
    const float scaleX = (float)width / (float)g_inputWidth;
    const float scaleY = (float)height / (float)g_inputHeight;
    
    for (int i = 0; i < g_numAnchors && count < MAX_DETECTIONS; i++) {
        // YOLOv8 output is transposed: [84, 8400]
        float cx = hostOutput[0 * g_numAnchors + i];
        float cy = hostOutput[1 * g_numAnchors + i];
        float w = hostOutput[2 * g_numAnchors + i];
        float h = hostOutput[3 * g_numAnchors + i];
        
        // Find best class
        float maxScore = 0.0f;
        int maxClass = 0;
        for (int c = 0; c < g_numClasses; c++) {
            float score = hostOutput[(4 + c) * g_numAnchors + i];
            if (score > maxScore) {
                maxScore = score;
                maxClass = c;
            }
        }
        
        if (maxScore >= confidenceThreshold) {
            // Convert from center format to corner format
            // Scale from model-input pixels to original pixel coordinates
            float x1 = (cx - w / 2.0f) * scaleX;
            float y1 = (cy - h / 2.0f) * scaleY;
            float x2 = (cx + w / 2.0f) * scaleX;
            float y2 = (cy + h / 2.0f) * scaleY;
            
            // Clamp to image bounds
            g_detectionBuffer[count].x1 = x1 < 0 ? 0 : (x1 > width ? width : x1);
            g_detectionBuffer[count].y1 = y1 < 0 ? 0 : (y1 > height ? height : y1);
            g_detectionBuffer[count].x2 = x2 < 0 ? 0 : (x2 > width ? width : x2);
            g_detectionBuffer[count].y2 = y2 < 0 ? 0 : (y2 > height ? height : y2);
            g_detectionBuffer[count].confidence = maxScore;
            g_detectionBuffer[count].class_index = maxClass;
            count++;
        }
    }
    
    free(hostOutput);
    free(hostInput);
    
    hostOutput = NULL;
    hostInput = NULL;

    CUDA_CHECK_GOTO(cudaEventRecord(g_stopEvent), cleanup);
    CUDA_CHECK_GOTO(cudaEventSynchronize(g_stopEvent), cleanup);
    CUDA_CHECK_GOTO(cudaEventElapsedTime(&postprocessMs, g_startEvent, g_stopEvent), cleanup);
    
    *outCount = count;
    *outPreprocessNs = (uint64_t)(preprocessMs * 1000000.0f);
    *outInferenceNs = (uint64_t)(inferenceMs * 1000000.0f);
    *outPostprocessNs = (uint64_t)(postprocessMs * 1000000.0f);
    
    return 0;

#ifdef CREBAIN_ONNXRUNTIME
cleanup_ort:
    if (outputMemInfo) {
        g_ort->ReleaseMemoryInfo(outputMemInfo);
    }
    if (outputTensor) {
        g_ort->ReleaseValue(outputTensor);
    }
#endif
cleanup:
    if (hostOutput) {
        free(hostOutput);
    }
    if (hostInput) {
        free(hostInput);
    }
    *outCount = 0;
    *outPreprocessNs = 0;
    *outInferenceNs = 0;
    *outPostprocessNs = 0;
    return rc;
}

Detection* cuda_backend_get_detections(void) {
    return g_detectionBuffer;
}

bool cuda_backend_is_ready(void) {
    return g_isInitialized && g_warmupComplete;
}

void cuda_backend_cleanup(void) {
#ifdef CREBAIN_ONNXRUNTIME
    if (g_ort != NULL) {
        if (g_inputName || g_outputName) {
            OrtAllocator* allocator = NULL;
            OrtStatus* status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
            if (status == NULL && allocator != NULL) {
                if (g_inputName) allocator->Free(allocator, g_inputName);
                if (g_outputName) allocator->Free(allocator, g_outputName);
            } else if (status != NULL) {
                g_ort->ReleaseStatus(status);
            }
        }
        if (g_ortSession) g_ort->ReleaseSession(g_ortSession);
        if (g_ortSessionOptions) g_ort->ReleaseSessionOptions(g_ortSessionOptions);
        if (g_ortMemoryInfo) g_ort->ReleaseMemoryInfo(g_ortMemoryInfo);
        if (g_ortEnv) g_ort->ReleaseEnv(g_ortEnv);
    }
    g_ortSession = NULL;
    g_ortSessionOptions = NULL;
    g_ortMemoryInfo = NULL;
    g_ortEnv = NULL;
    g_inputName = NULL;
    g_outputName = NULL;
#endif
    
    if (g_deviceInput) cudaFree(g_deviceInput);
    
    if (g_startEvent) cudaEventDestroy(g_startEvent);
    if (g_stopEvent) cudaEventDestroy(g_stopEvent);
    
    g_deviceInput = NULL;
    g_startEvent = NULL;
    g_stopEvent = NULL;
    g_isInitialized = false;
    g_warmupComplete = false;
}

const char* cuda_backend_get_device_name(void) {
    static char deviceName[256] = "Unknown CUDA Device";
    
    int device;
    if (cudaGetDevice(&device) == cudaSuccess) {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, device) == cudaSuccess) {
            snprintf(deviceName, sizeof(deviceName), "%s (CUDA)", props.name);
        }
    }
    return deviceName;
}

#else // !CREBAIN_CUDA

// Stub implementations for non-CUDA builds

typedef struct {
    float x1, y1, x2, y2;
    float confidence;
    int32_t class_index;
} Detection;

int32_t cuda_backend_init(const char *modelPath) {
    (void)modelPath;
    fprintf(stderr, "CUDA: Backend not compiled - CREBAIN_CUDA not defined\n");
    return -1;
}

int32_t cuda_backend_detect(
    const uint8_t *pixels, int32_t width, int32_t height, int32_t bytesPerRow,
    float confidenceThreshold, int32_t *outCount, uint64_t *outInferenceNs,
    uint64_t *outPreprocessNs, uint64_t *outPostprocessNs
) {
    (void)pixels; (void)width; (void)height; (void)bytesPerRow;
    (void)confidenceThreshold; (void)outInferenceNs;
    (void)outPreprocessNs; (void)outPostprocessNs;
    *outCount = 0;
    return -1;
}

Detection* cuda_backend_get_detections(void) { return NULL; }
bool cuda_backend_is_ready(void) { return false; }
void cuda_backend_cleanup(void) {}
const char* cuda_backend_get_device_name(void) { return "CUDA Not Available"; }

#endif // CREBAIN_CUDA
