// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CoreMLFFI",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "CoreMLFFI",
            type: .dynamic,
            targets: ["CoreMLFFI"]
        )
    ],
    targets: [
        .target(
            name: "CoreMLFFI",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("Vision"),
                .linkedFramework("AppKit"),
                .linkedFramework("Accelerate")
            ]
        )
    ]
)
