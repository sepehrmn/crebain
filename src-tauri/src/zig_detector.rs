//! CREBAIN Zig Detector Integration
//!
//! Dynamic loading of the Zig-compiled cross-platform ML detector library.
//!
//! # Safety
//!
//! This module uses FFI to call into a dynamically loaded Zig library.
//! The following invariants must be maintained:
//!
//! 1. The `Library` must outlive all function pointer calls
//! 2. Function pointers are only valid while the Library is loaded
//! 3. FFI calls must validate all pointer arguments before dereferencing
//!
//! # Thread Safety
//!
//! The Zig detector is NOT thread-safe. All calls must be serialized.
//! The global detector is protected by `OnceLock` for initialization only.

use libloading::Library;
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FFI TYPES (must match crebain_detector.h exactly)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Maximum number of detections we'll accept from FFI to prevent buffer overread
const MAX_FFI_DETECTIONS: usize = 1000;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CrebainDetection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_index: i32,
}

#[repr(C)]
#[derive(Debug)]
pub struct CrebainDetectionResult {
    pub detections: *mut CrebainDetection,
    pub count: i32,
    pub inference_time_ns: u64,
    pub preprocess_time_ns: u64,
    pub postprocess_time_ns: u64,
    pub success: bool,
    pub error_code: i32,
}

#[repr(C)]
#[allow(dead_code)] // FFI enum - variants used by Zig detector library
#[allow(clippy::upper_case_acronyms)] // Industry-standard ML backend names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrebainBackendType {
    CoreML = 0,
    MPS = 1,
    MLX = 2,
    CPU = 3,
    CUDA = 4,
    TensorRT = 5,
    ONNX = 6,
    Unknown = -1,
}

#[repr(C)]
pub struct CrebainDetectorConfig {
    pub model_path: *const i8,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_detections: i32,
    pub preferred_backend: CrebainBackendType,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RUST-FRIENDLY TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_index: i32,
    pub class_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZigDetectionResult {
    pub success: bool,
    pub detections: Vec<Detection>,
    pub inference_time_ms: f64,
    pub preprocess_time_ms: f64,
    pub postprocess_time_ms: f64,
    pub backend: String,
    pub error: Option<String>,
}

// Use the shared COCO classes from the common module
use crate::common::coco::get_class_name;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FFI FUNCTION POINTER TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

type InitFn = unsafe extern "C" fn(*const CrebainDetectorConfig) -> i32;
type DetectFn = unsafe extern "C" fn(*const u8, i32, i32, i32, *mut CrebainDetectionResult) -> i32;
type GetBackendFn = unsafe extern "C" fn() -> CrebainBackendType;
type IsReadyFn = unsafe extern "C" fn() -> bool;
type GetBackendNameFn = unsafe extern "C" fn() -> *const i8;
type CleanupFn = unsafe extern "C" fn();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DYNAMIC LIBRARY WRAPPER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Raw function pointers extracted from the dynamic library.
///
/// # Safety
///
/// These pointers are only valid while the parent `ZigDetector`'s
/// `_library` field is alive. The struct is designed to ensure this
/// by storing both together and never exposing the library.
struct FunctionPointers {
    init: InitFn,
    detect: DetectFn,
    get_backend: GetBackendFn,
    is_ready: IsReadyFn,
    get_backend_name: GetBackendNameFn,
    cleanup: CleanupFn,
}

/// Zig detector wrapper with proper lifetime management.
///
/// The library is stored alongside its function pointers to ensure
/// the pointers remain valid for the lifetime of the detector.
pub struct ZigDetector {
    /// The loaded dynamic library. MUST be kept alive for function pointers to be valid.
    /// Stored as Box to ensure stable address (not strictly necessary but defensive).
    _library: Box<Library>,
    /// Function pointers extracted from the library
    fns: FunctionPointers,
    /// Whether init() has been called successfully
    initialized: bool,
    /// Cached confidence threshold for reference
    confidence_threshold: f32,
}

// ZigDetector is NOT Send or Sync because:
// 1. The underlying Zig library may not be thread-safe
// 2. Function pointers don't carry thread-safety guarantees
// We explicitly do NOT implement Send/Sync.

/// Global singleton for the Zig detector.
/// Only one detector instance is supported.
static ZIG_DETECTOR: OnceLock<Option<ZigDetector>> = OnceLock::new();

impl ZigDetector {
    /// Load the Zig detector library from the given path.
    ///
    /// # Safety
    ///
    /// The library at `library_path` must:
    /// - Be a valid dynamic library for the current platform
    /// - Export all required symbols with correct signatures
    /// - Remain on disk for the lifetime of this detector
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded or required symbols are missing.
    pub fn load(library_path: &str) -> Result<Self, String> {
        // Validate path doesn't contain suspicious patterns
        if library_path.contains("..") || library_path.contains('\0') {
            return Err("Invalid library path: contains forbidden characters".to_string());
        }

        // Load the library
        let library = unsafe {
            Library::new(library_path)
                .map_err(|e| format!("Failed to load Zig detector library '{}': {}", library_path, e))?
        };

        // Extract function pointers while library reference is valid
        // SAFETY: We store the library in the struct, ensuring pointers remain valid
        let fns = unsafe {
            let init: InitFn = *library
                .get(b"crebain_init\0")
                .map_err(|e| format!("Symbol 'crebain_init' not found: {}", e))?;

            let detect: DetectFn = *library
                .get(b"crebain_detect\0")
                .map_err(|e| format!("Symbol 'crebain_detect' not found: {}", e))?;

            let get_backend: GetBackendFn = *library
                .get(b"crebain_get_backend\0")
                .map_err(|e| format!("Symbol 'crebain_get_backend' not found: {}", e))?;

            let is_ready: IsReadyFn = *library
                .get(b"crebain_is_ready\0")
                .map_err(|e| format!("Symbol 'crebain_is_ready' not found: {}", e))?;

            let get_backend_name: GetBackendNameFn = *library
                .get(b"crebain_get_backend_name\0")
                .map_err(|e| format!("Symbol 'crebain_get_backend_name' not found: {}", e))?;

            let cleanup: CleanupFn = *library
                .get(b"crebain_cleanup\0")
                .map_err(|e| format!("Symbol 'crebain_cleanup' not found: {}", e))?;

            FunctionPointers {
                init,
                detect,
                get_backend,
                is_ready,
                get_backend_name,
                cleanup,
            }
        };

        Ok(ZigDetector {
            _library: Box::new(library),
            fns,
            initialized: false,
            confidence_threshold: 0.25,
        })
    }

    /// Initialize the detector with a model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ML model file
    /// * `confidence_threshold` - Minimum confidence for detections (0.0-1.0)
    /// * `iou_threshold` - IoU threshold for NMS (0.0-1.0)
    /// * `max_detections` - Maximum number of detections to return
    /// * `preferred_backend` - Preferred compute backend
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn init(
        &mut self,
        model_path: &str,
        confidence_threshold: f32,
        iou_threshold: f32,
        max_detections: i32,
        preferred_backend: CrebainBackendType,
    ) -> Result<(), String> {
        // Validate thresholds
        if !(0.0..=1.0).contains(&confidence_threshold) {
            return Err("confidence_threshold must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&iou_threshold) {
            return Err("iou_threshold must be between 0.0 and 1.0".to_string());
        }
        if max_detections <= 0 || max_detections > MAX_FFI_DETECTIONS as i32 {
            return Err(format!("max_detections must be between 1 and {}", MAX_FFI_DETECTIONS));
        }

        let model_path_c = CString::new(model_path)
            .map_err(|_| "Invalid model path: contains null byte".to_string())?;

        let config = CrebainDetectorConfig {
            model_path: model_path_c.as_ptr(),
            confidence_threshold,
            iou_threshold,
            max_detections,
            preferred_backend,
        };

        // SAFETY: config is valid for the duration of this call,
        // and the library function is expected to not store the pointer
        let result = unsafe { (self.fns.init)(&config) };

        if result == 0 {
            self.initialized = true;
            self.confidence_threshold = confidence_threshold;
            log::info!(
                "[ZigDetector] Initialized with backend: {}, confidence: {:.2}",
                self.get_backend_name(),
                confidence_threshold
            );
            Ok(())
        } else {
            Err(format!(
                "Zig detector initialization failed with error code: {}",
                result
            ))
        }
    }

    /// Run detection on raw RGBA pixel data.
    ///
    /// # Arguments
    ///
    /// * `pixels` - RGBA pixel data (4 bytes per pixel)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails or detector is not initialized.
    pub fn detect(&self, pixels: &[u8], width: u32, height: u32) -> Result<ZigDetectionResult, String> {
        if !self.initialized {
            return Err("Zig detector not initialized".to_string());
        }

        // Validate input dimensions
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|v| v.checked_mul(4))
            .ok_or("Image dimensions overflow")?;

        if pixels.len() != expected_len {
            return Err(format!(
                "Invalid pixel buffer: expected {} bytes for {}x{} RGBA, got {}",
                expected_len, width, height, pixels.len()
            ));
        }

        let bytes_per_row = (width as i32).checked_mul(4)
            .ok_or("bytes_per_row overflow")?;

        let mut result = CrebainDetectionResult {
            detections: std::ptr::null_mut(),
            count: 0,
            inference_time_ns: 0,
            preprocess_time_ns: 0,
            postprocess_time_ns: 0,
            success: false,
            error_code: 0,
        };

        // SAFETY: pixels is valid for the duration of the call,
        // result is a valid mutable pointer
        let ret = unsafe {
            (self.fns.detect)(
                pixels.as_ptr(),
                width as i32,
                height as i32,
                bytes_per_row,
                &mut result,
            )
        };

        if ret != 0 || !result.success {
            return Err(format!(
                "Detection failed: ret={}, error_code={}",
                ret, result.error_code
            ));
        }

        // Convert C detections to Rust with bounds checking
        let detection_count = result.count as usize;

        // Validate count is reasonable to prevent buffer overread
        if detection_count > MAX_FFI_DETECTIONS {
            log::error!(
                "[ZigDetector] FFI returned suspicious detection count: {}",
                detection_count
            );
            return Err(format!(
                "Detection count {} exceeds maximum {}",
                detection_count, MAX_FFI_DETECTIONS
            ));
        }

        let mut detections = Vec::with_capacity(detection_count);

        if !result.detections.is_null() && detection_count > 0 {
            // SAFETY: We've validated count is within bounds,
            // and the FFI contract guarantees detections array is valid for count elements
            unsafe {
                for i in 0..detection_count {
                    let det = *result.detections.add(i);

                    // Validate detection values are reasonable
                    if det.confidence < 0.0 || det.confidence > 1.0 {
                        log::warn!("[ZigDetector] Invalid confidence value: {}", det.confidence);
                        continue;
                    }

                    detections.push(Detection {
                        x1: det.x1,
                        y1: det.y1,
                        x2: det.x2,
                        y2: det.y2,
                        confidence: det.confidence,
                        class_index: det.class_index,
                        class_name: get_class_name(det.class_index as usize),
                    });
                }
            }
        }

        Ok(ZigDetectionResult {
            success: true,
            detections,
            inference_time_ms: result.inference_time_ns as f64 / 1_000_000.0,
            preprocess_time_ms: result.preprocess_time_ns as f64 / 1_000_000.0,
            postprocess_time_ms: result.postprocess_time_ns as f64 / 1_000_000.0,
            backend: self.get_backend_name(),
            error: None,
        })
    }

    /// Check if the detector is ready for inference.
    #[inline]
    pub fn is_ready(&self) -> bool {
        // SAFETY: is_ready is a simple query with no side effects
        self.initialized && unsafe { (self.fns.is_ready)() }
    }

    /// Get the current backend type.
    #[allow(dead_code)]
    pub fn get_backend(&self) -> CrebainBackendType {
        // SAFETY: get_backend is a simple query with no side effects
        unsafe { (self.fns.get_backend)() }
    }

    /// Get the backend name as a string.
    pub fn get_backend_name(&self) -> String {
        // SAFETY: get_backend_name returns a static string pointer
        unsafe {
            let name_ptr = (self.fns.get_backend_name)();
            if name_ptr.is_null() {
                return "Unknown".to_string();
            }

            // Use from_ptr but limit string length to prevent unbounded read
            let cstr = CStr::from_ptr(name_ptr);
            let bytes = cstr.to_bytes();

            // Sanity check: backend names should be short
            if bytes.len() > 64 {
                log::warn!("[ZigDetector] Suspiciously long backend name, truncating");
                return String::from_utf8_lossy(&bytes[..64]).into_owned();
            }

            cstr.to_string_lossy().into_owned()
        }
    }

    /// Cleanup detector resources.
    fn cleanup(&mut self) {
        if self.initialized {
            // SAFETY: cleanup is idempotent and safe to call
            unsafe { (self.fns.cleanup)() };
            self.initialized = false;
            log::debug!("[ZigDetector] Cleanup complete");
        }
    }
}

impl Drop for ZigDetector {
    fn drop(&mut self) {
        self.cleanup();
        // Library will be dropped automatically, unloading the dynamic library
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GLOBAL API
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Find the Zig detector library on the filesystem.
fn find_library_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let lib_name = "libcrebain_detector.dylib";

    #[cfg(target_os = "linux")]
    let lib_name = "libcrebain_detector.so";

    #[cfg(target_os = "windows")]
    let lib_name = "crebain_detector.dll";

    let search_paths: Vec<Option<PathBuf>> = vec![
        // Bundled in Tauri app (macOS)
        #[cfg(target_os = "macos")]
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join(format!("../Resources/{}", lib_name)))),
        // Bundled in Tauri app (Linux)
        #[cfg(target_os = "linux")]
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join(lib_name))),
        // Nix-style install layout: $out/bin/<exe> and $out/lib/<libcrebain_detector.so>
        #[cfg(target_os = "linux")]
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("..").join("lib").join(lib_name))),
        #[cfg(target_os = "linux")]
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("..").join("lib64").join(lib_name))),
        // Development path
        std::env::current_dir()
            .ok()
            .map(|p| p.join(format!("src-tauri/binaries/{}", lib_name))),
        // Build output
        std::env::current_dir()
            .ok()
            .map(|p| p.join(format!("src-tauri/native/zig-detector/zig-out/lib/{}", lib_name))),
        // System-wide locations (Linux)
        #[cfg(target_os = "linux")]
        Some(PathBuf::from(format!("/usr/lib/{}", lib_name))),
        #[cfg(target_os = "linux")]
        Some(PathBuf::from(format!("/usr/local/lib/{}", lib_name))),
        #[cfg(target_os = "linux")]
        Some(PathBuf::from(format!("/opt/crebain/lib/{}", lib_name))),
    ];

    for path in search_paths.into_iter().flatten() {
        if path.exists() {
            return Some(path);
        }
    }

    // Check environment variable (lower priority than bundled paths for security)
    if let Ok(custom_path) = std::env::var("CREBAIN_DETECTOR_LIB") {
        let path = PathBuf::from(&custom_path);

        // Security: validate the path
        if custom_path.contains("..") {
            log::warn!("[ZigDetector] Rejecting CREBAIN_DETECTOR_LIB with path traversal");
            return None;
        }

        if path.exists() {
            log::info!("[ZigDetector] Using library from CREBAIN_DETECTOR_LIB: {:?}", path);
            return Some(path);
        }
    }

    None
}

/// Find the model path on the filesystem.
fn find_model_path() -> Option<PathBuf> {
    // Environment variables (highest priority but validated)
    if let Ok(custom_path) = std::env::var("CREBAIN_ONNX_MODEL") {
        // Security: reject path traversal
        if custom_path.contains("..") {
            log::warn!("[ZigDetector] Rejecting CREBAIN_ONNX_MODEL with path traversal");
        } else {
            let path = PathBuf::from(&custom_path);
            if path.exists() {
                return Some(path);
            }
        }
    }

    if let Ok(custom_path) = std::env::var("CREBAIN_MODEL_PATH") {
        // Security: reject path traversal
        if custom_path.contains("..") {
            log::warn!("[ZigDetector] Rejecting CREBAIN_MODEL_PATH with path traversal");
        } else {
            let path = PathBuf::from(&custom_path);
            if path.exists() {
                return Some(path);
            }
        }
    }

    let search_paths: Vec<Option<PathBuf>> = vec![
        // Bundled in Tauri app
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("../Resources/yolov8s.mlmodelc"))),
        // Nix-style install layout: $out/bin/<exe> and $out/share/crebain/models/...
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("../share/crebain/models/yolov8s.onnx"))),
        // Development path
        std::env::current_dir()
            .ok()
            .map(|p| p.join("src-tauri/resources/yolov8s.mlmodelc")),
        // Linux: Use ONNX model
        #[cfg(target_os = "linux")]
        std::env::current_dir()
            .ok()
            .map(|p| p.join("src-tauri/resources/yolov8s.onnx")),
        #[cfg(target_os = "linux")]
        Some(PathBuf::from("/usr/share/crebain/models/yolov8s.onnx")),
        #[cfg(target_os = "linux")]
        Some(PathBuf::from("/opt/crebain/models/yolov8s.onnx")),
    ];

    search_paths.into_iter().flatten().find(|p| p.exists())
}

/// Initialize the global Zig detector singleton.
///
/// This function is idempotent - calling it multiple times is safe
/// and will return the same result after the first initialization.
pub fn init_global_detector() -> Result<(), String> {
    ZIG_DETECTOR.get_or_init(|| {
        let library_path = match find_library_path() {
            Some(p) => p,
            None => {
                log::warn!("[ZigDetector] Library not found in any search path");
                return None;
            }
        };

        let model_path = match find_model_path() {
            Some(p) => p,
            None => {
                log::warn!("[ZigDetector] Model not found in any search path");
                return None;
            }
        };

        log::info!("[ZigDetector] Loading library: {:?}", library_path);
        log::info!("[ZigDetector] Using model: {:?}", model_path);

        // Select preferred backend based on platform
        #[cfg(target_os = "macos")]
        let preferred_backend = CrebainBackendType::CoreML;

        #[cfg(target_os = "linux")]
        let preferred_backend = CrebainBackendType::CUDA;

        #[cfg(target_os = "windows")]
        let preferred_backend = CrebainBackendType::CUDA;

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        let preferred_backend = CrebainBackendType::CPU;

        match ZigDetector::load(&library_path.to_string_lossy()) {
            Ok(mut detector) => {
                match detector.init(
                    &model_path.to_string_lossy(),
                    0.25, // confidence threshold
                    0.45, // IoU threshold
                    100,  // max detections
                    preferred_backend,
                ) {
                    Ok(()) => Some(detector),
                    Err(e) => {
                        log::error!("[ZigDetector] Init failed: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                log::error!("[ZigDetector] Load failed: {}", e);
                None
            }
        }
    });

    if ZIG_DETECTOR.get().and_then(|d| d.as_ref()).is_some() {
        Ok(())
    } else {
        Err("Failed to initialize Zig detector".to_string())
    }
}

/// Get a reference to the global Zig detector.
///
/// Returns `None` if the detector was not successfully initialized.
pub fn get_global_detector() -> Option<&'static ZigDetector> {
    ZIG_DETECTOR.get().and_then(|opt| opt.as_ref())
}

/// Run detection using the global Zig detector.
///
/// # Errors
///
/// Returns an error if the detector is not initialized or detection fails.
pub fn detect_with_zig(pixels: &[u8], width: u32, height: u32) -> Result<ZigDetectionResult, String> {
    get_global_detector()
        .ok_or_else(|| "Zig detector not initialized".to_string())?
        .detect(pixels, width, height)
}

/// Check if the global Zig detector is available and ready.
#[allow(dead_code)]
pub fn is_zig_detector_ready() -> bool {
    get_global_detector().map(|d| d.is_ready()).unwrap_or(false)
}

/// Get diagnostic information about the Zig detector.
pub fn get_zig_detector_info() -> serde_json::Value {
    match get_global_detector() {
        Some(detector) => serde_json::json!({
            "available": true,
            "ready": detector.is_ready(),
            "backend": detector.get_backend_name(),
        }),
        None => serde_json::json!({
            "available": false,
            "ready": false,
            "backend": "Not Loaded",
        }),
    }
}
