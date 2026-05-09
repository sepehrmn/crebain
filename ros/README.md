# CREBAIN ROS Integration

Reference files for ROS/Gazebo integration. These files document message formats and launch configurations intended for CREBAIN-compatible ROS packages.

## Structure

```
ros/
├── msg/                    # Message definitions
│   ├── ThermalDetection.msg
│   ├── ThermalDetectionArray.msg
│   ├── AcousticDetection.msg
│   ├── AcousticDetectionArray.msg
│   ├── RadarDetection.msg
│   ├── RadarDetectionArray.msg
│   ├── DroneTarget.msg
│   ├── InterceptionCommand.msg
│   └── InterceptionStatus.msg
├── srv/                    # Service definitions
│   ├── InitiateIntercept.srv
│   └── AbortMission.srv
└── launch/                 # Launch files
    ├── simulation.launch   # Full simulation
    ├── multi_drone.launch  # Multi-drone spawning
    └── rosbridge.launch    # WebSocket bridge config
```

## Usage

These are reference files, not a complete standalone ROS package. To use them with a full ROS package:

1. Create a catkin package:

   ```bash
   # Run from ~/catkin_ws/src
   catkin_create_pkg crebain_msgs std_msgs geometry_msgs
   ```

2. Copy msg/ and srv/ to the package
3. Update CMakeLists.txt and package.xml
4. Build:

   ```bash
   catkin_make
   ```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/crebain/thermal/detections` | ThermalDetectionArray | Thermal camera detections |
| `/crebain/acoustic/detections` | AcousticDetectionArray | Microphone array detections |
| `/crebain/radar/detections` | RadarDetectionArray | Radar returns |
| `/crebain/targets` | DroneTarget[] | Tracked targets |

CREBAIN also contains WebSocket-based rosbridge integration and Zenoh-oriented transport adapters. Treat latency and throughput as deployment-specific; measure them in the target ROS/Gazebo topology instead of relying on generic transport assumptions.

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/crebain/initiate_intercept` | InitiateIntercept | Start interception |
| `/crebain/abort_mission` | AbortMission | Abort mission |

## Quick Start

```bash
# Terminal 1: Launch simulation
roslaunch crebain_gazebo simulation.launch

# Terminal 2: Start CREBAIN
bun run dev
```

Connect to `ws://localhost:9090` in CREBAIN.
