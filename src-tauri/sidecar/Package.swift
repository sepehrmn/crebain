// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "coreml-detector",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "coreml-detector",
            path: "Sources"
        )
    ]
)
