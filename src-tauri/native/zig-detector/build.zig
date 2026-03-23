const std = @import("std");

/// CREBAIN Cross-Platform ML Detector Build System
/// Supports: CoreML (macOS), MPS (macOS GPU), MLX (Apple Silicon), CUDA (Linux/Windows)
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const enable_cuda = b.option(bool, "cuda", "Enable CUDA backend (requires CUDA toolkit)") orelse false;
    const enable_onnx = b.option(bool, "onnx", "Enable ONNX Runtime for CUDA inference") orelse false;
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA toolkit") orelse "/usr/local/cuda";
    const onnx_path = b.option([]const u8, "onnx-path", "Path to ONNX Runtime") orelse "/usr/local";

    // Detect platform for backend selection
    const is_macos = target.result.os.tag == .macos;
    const is_linux = target.result.os.tag == .linux;
    const is_windows = target.result.os.tag == .windows;

    // Build the cross-platform detector library
    const lib = b.addLibrary(.{
        .name = "crebain_detector",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/detector.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Platform-specific configurations
    if (is_macos) {
        // Link macOS frameworks for CoreML/MPS/MLX
        lib.linkFramework("Foundation");
        lib.linkFramework("CoreML");
        lib.linkFramework("Vision");
        lib.linkFramework("Metal");
        lib.linkFramework("MetalPerformanceShaders");
        lib.linkFramework("Accelerate");
        lib.linkFramework("CoreGraphics");
        lib.linkFramework("AppKit");

        // Add Objective-C runtime for CoreML interop
        lib.linkSystemLibrary("objc");

        // Define macOS-specific build flags
        lib.root_module.addCMacro("CREBAIN_MACOS", "1");
        lib.root_module.addCMacro("CREBAIN_COREML", "1");
        lib.root_module.addCMacro("CREBAIN_MPS", "1");
    } else if (is_linux) {
        lib.root_module.addCMacro("CREBAIN_LINUX", "1");

	        if (enable_cuda) {
	            // CUDA backend for Linux
	            lib.root_module.addCMacro("CREBAIN_CUDA", "1");
	            lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_path}) });
	            // CUDA Toolkit library locations vary across distros/Nixpkgs. Add a
	            // few common candidates (including stubs for linking in sandboxed
	            // environments where the NVIDIA driver isn't present at build time).
	            lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_path}) });
	            lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{cuda_path}) });
	            lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64/stubs", .{cuda_path}) });
	            lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib/stubs", .{cuda_path}) });
	            lib.linkSystemLibrary("cuda");
	            lib.linkSystemLibrary("cudart");

	            // ONNX Runtime for actual inference
	            if (enable_onnx) {
	                lib.root_module.addCMacro("CREBAIN_ONNXRUNTIME", "1");
	                lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{onnx_path}) });
	                lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{onnx_path}) });
	                lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{onnx_path}) });
	                lib.linkSystemLibrary("onnxruntime");
	            }
	        } else {
	            lib.root_module.addCMacro("CREBAIN_CPU_ONLY", "1");
	        }
    } else if (is_windows) {
        lib.root_module.addCMacro("CREBAIN_WINDOWS", "1");

        if (enable_cuda) {
            lib.root_module.addCMacro("CREBAIN_CUDA", "1");
            lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_path}) });
            lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib/x64", .{cuda_path}) });
            lib.linkSystemLibrary("cuda");
            lib.linkSystemLibrary("cudart");

            // ONNX Runtime for actual inference
            if (enable_onnx) {
                lib.root_module.addCMacro("CREBAIN_ONNXRUNTIME", "1");
                lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{onnx_path}) });
                lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{onnx_path}) });
                lib.linkSystemLibrary("onnxruntime");
            }
        } else {
            lib.root_module.addCMacro("CREBAIN_CPU_ONLY", "1");
        }
    }

    // Add C source files for platform-specific backends
	    lib.addCSourceFiles(.{
	        .files = &.{
	            "src/backends/common.c",
	        },
	        .flags = &.{
	            "-std=c17",
	            "-O3",
	            "-ffast-math",
	            "-DNDEBUG",
	        },
	    });

	    // CUDA backend sources (always compile a stub so the library links on
	    // non-CUDA platforms; enable the real backend only when requested).
	    const cuda_c_flags: []const []const u8 = if ((is_linux or is_windows) and enable_cuda)
	        (if (enable_onnx)
	            &.{ "-std=c17", "-O3", "-DCREBAIN_CUDA=1", "-DCREBAIN_ONNXRUNTIME=1" }
	        else
	            &.{ "-std=c17", "-O3", "-DCREBAIN_CUDA=1" })
	    else
	        &.{ "-std=c17", "-O3" };

	    lib.addCSourceFiles(.{
	        .files = &.{
	            "src/backends/cuda_backend.c",
	        },
	        .flags = cuda_c_flags,
	    });

	    // macOS-specific Objective-C sources
	    if (is_macos) {
	        lib.addCSourceFiles(.{
	            .files = &.{
	                "src/backends/coreml_backend.m",
	                "src/backends/mps_backend.m",
	            },
	            .flags = &.{
	                "-fobjc-arc",
	                "-O3",
	                "-ffast-math",
	            },
	        });
	    }

    // Install the library
    b.installArtifact(lib);

    // Create a static library variant for Rust linking
    const static_lib = b.addLibrary(.{
        .name = "crebain_detector_static",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/detector.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    if (is_macos) {
        static_lib.linkFramework("Foundation");
        static_lib.linkFramework("CoreML");
        static_lib.linkFramework("Vision");
        static_lib.linkFramework("Metal");
        static_lib.linkFramework("MetalPerformanceShaders");
        static_lib.linkFramework("Accelerate");
        static_lib.linkFramework("CoreGraphics");
        static_lib.linkSystemLibrary("objc");
        static_lib.root_module.addCMacro("CREBAIN_MACOS", "1");
        static_lib.root_module.addCMacro("CREBAIN_COREML", "1");
        static_lib.root_module.addCMacro("CREBAIN_MPS", "1");
    }

	    if ((is_linux or is_windows) and enable_cuda) {
	        static_lib.root_module.addCMacro("CREBAIN_CUDA", "1");
	        static_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_path}) });
	        if (is_linux) {
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_path}) });
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{cuda_path}) });
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64/stubs", .{cuda_path}) });
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib/stubs", .{cuda_path}) });
	        } else {
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib/x64", .{cuda_path}) });
	        }
	        static_lib.linkSystemLibrary("cuda");
	        static_lib.linkSystemLibrary("cudart");

	        if (enable_onnx) {
	            static_lib.root_module.addCMacro("CREBAIN_ONNXRUNTIME", "1");
	            static_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{onnx_path}) });
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{onnx_path}) });
	            static_lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{onnx_path}) });
	            static_lib.linkSystemLibrary("onnxruntime");
	        }
	    }

    b.installArtifact(static_lib);

    // Generate C header for FFI
    const header_step = b.addInstallFileWithDir(
        b.path("src/crebain_detector.h"),
        .header,
        "crebain_detector.h",
    );
    b.getInstallStep().dependOn(&header_step.step);

    // Test step - placeholder for now
    const test_step = b.step("test", "Run unit tests");
    _ = test_step;

    // Benchmark step - placeholder for now
    const bench_step = b.step("bench", "Run benchmarks");
    _ = bench_step;
}
