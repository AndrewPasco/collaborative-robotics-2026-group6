# TidyBot2 Repository Layout & Execution Guide

This document provides a detailed overview of the system architecture, file layout, and instructions for running the various ROS2 nodes and scripts within the TidyBot2 project.

## 1. ROS2 Workspace Structure (`ros2_ws/src`)

The ROS2 workspace is organized into several packages, each responsible for a specific aspect of the robot's functionality.

### 1.1 Core TidyBot2 Packages

- **`tidybot_bringup`**: The entry point for launching the robot. Contains launch files, configurations (YAML), RViz profiles, and test scripts.
- **`tidybot_control`**: Contains hardware interface nodes and wrappers.
  - `arm_wrapper_node.py`: Translates simulation-compatible joint commands to real hardware commands.
  - `gripper_wrapper_node.py`: Translates simulation-compatible gripper commands to real hardware commands.
  - `phoenix6_base_node.py`: Driver for the Phoenix 6 mobile base.
  - `microphone_node.py`: Node for recording audio from the robot's microphone.
  - `pan_tilt_node.py`: Controls the camera's pan-tilt mechanism.
- **`tidybot_ik`**: Motion planning and Inverse Kinematics.
  - `motion_planner_node.py`: IK planner for simulation.
  - `motion_planner_real_node.py`: IK planner optimized for real hardware.
- **`tidybot_mujoco_bridge`**: The bridge between ROS2 and the MuJoCo simulation.
  - `mujoco_bridge_node.py`: Synchronizes ROS2 topics with the MuJoCo simulation environment.
- **`tidybot_description`**: URDF/Xacro models and mesh assets for the robot.
- **`tidybot_msgs`**: Custom ROS2 message and service definitions.
- **`tidybot_network_bridge`**: Utilities for remote operation.
  - `image_compression_node.py`: Compresses camera streams for low-bandwidth remote viewing.

### 1.2 Third-Party Integrations

- **`interbotix`**: Submodule containing drivers and modules for the WidowX WX250s arms.

---

## 2. Launching the System

### 2.2 Full System (Brain + Simulation)
To start the entire system including speech, navigation, manipulation, and simulation:
```bash
ros2 launch tidybot_bringup brain.launch.py
```
This is the primary entry point for task execution. It launches:
- **Speech System**: Microphone capture + STT-to-Gemini pipeline.
- **Orchestration**: The Brain Node master state machine.
- **Support Nodes**: Navigation and Manipulation controllers.
- **Simulation**: MuJoCo bridge and visualization.

### 2.3 Real Hardware
To launch the drivers for the physical robot:
```bash
ros2 launch tidybot_bringup real.launch.py
```
**Key Arguments:**
- `use_base:='true'/'false'` (Default: true)
- `use_arms:='true'/'false'` (Default: true)
- `use_camera:='true'/'false'` (Default: true)
- `use_microphone:='true'/'false'` (Default: true)

---

## 3. Running Specific Nodes & Scripts

All core nodes can be run individually using `ros2 run`. Note that most hardware nodes require specific environment variables or configurations usually handled by launch files.

### 3.1 Control & Hardware Nodes (`tidybot_control`)

| Node | Description | Run Command |
|---|---|---|
| **Base Driver** | Controls the holonomic base | `ros2 run tidybot_control phoenix6_base_node` |
| **Arm Wrapper** | Sim-to-Real command bridge | `ros2 run tidybot_control arm_wrapper_node` |
| **Microphone** | Audio capture | `ros2 run tidybot_control microphone_node` |
| **Dynamixel Bus** | Multi-arm bus manager (alternative to xs_sdk) | `ros2 run tidybot_control dynamixel_bus_node` |

### 3.2 Motion Planning (`tidybot_ik`)

| Node | Description | Run Command |
|---|---|---|
| **Planner (Real)** | IK for physical robot | `ros2 run tidybot_ik motion_planner_real_node --ros-args -p urdf_path:=<path_to_xacro>` |

---

## 4. Test & Utility Scripts (`tidybot_bringup/scripts`)

These scripts are useful for verifying individual components of the system.

### 4.1 Base & Arm Testing
- **Test Base (Sim):** `ros2 run tidybot_bringup test_base_sim.py`
- **Test Base (Real):** `ros2 run tidybot_bringup test_base_real.py`
- **Test Arms (Sim):** `ros2 run tidybot_bringup test_arms_sim.py`
- **Test Arms (Real):** `ros2 run tidybot_bringup test_arms_real.py`
- **Test Grippers:** `ros2 run tidybot_bringup test_grippers_real.py`
- **Test Torque Hold:** `ros2 run tidybot_bringup test_torque_hold.py`

### 4.2 Sensors & Perception
- **Test Camera (Sim):** `ros2 run tidybot_bringup test_camera_sim.py`
- **Test Microphone:** `ros2 run tidybot_bringup test_microphone.py`
- **Test TF Tree:** `ros2 run tidybot_bringup test_tf.py`

### 4.3 Audio & Speech
- **Test STT:** `ros2 run tidybot_bringup test_stt.py`
- **Audio Item Detector:** `ros2 run tidybot_bringup audio_item_detector.py`

### 4.4 High-Level logic & Perception
- **Brain Node:** `ros2 run tidybot_bringup brain_node.py`
  - The central master orchestrator.
  - Manages the state machine: Listening -> Navigating -> Grabbing -> Returning -> Releasing.
- **Speech Node:** `ros2 run tidybot_bringup speech_node.py`
  - Manages audio recording and triggers the extraction pipeline.
- **Audio Item Detector:** `ros2 run tidybot_bringup audio_item_detector.py`
  - Implements the two-stage pipeline: **Google Cloud STT** (transcription) followed by **Gemini-2.0-Flash** (text-based item extraction).
- **Example State Machine:** `ros2 run tidybot_bringup example_state_machine.py`
- **Trajectory Tracking:** `ros2 run tidybot_bringup trajectory_tracking.py`

---

## 5. Standalone MuJoCo Simulation (`simulation/scripts`)

These scripts do **not** use ROS2 and interact directly with the MuJoCo engine. They are useful for rapid prototyping of motion or control logic.

- **Arm Movement Demo:** `uv run python simulation/scripts/test_move_sim.py`
- **Pick and Place Demo:** `uv run python simulation/scripts/pick_up_block_sim.py`

---

## 6. ROS2 API Reference (Topics & Services)

Below is an exhaustive list of the primary ROS2 topics and services available in TidyBot2, grouped by functionality.

### 6.1 Base Control & Odometry
| Topic | Type | Description |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Continuous velocity control (x, y, theta). |
| `/base/target_pose` | `geometry_msgs/Pose2D` | Target position (go-to-goal) in robot-relative or world frame. |
| `/odom` | `nav_msgs/Odometry` | Odometry feedback from the mobile base. |
| `/base/goal_reached` | `std_msgs/Bool` | Published when the robot reaches a `/base/target_pose`. |

### 6.2 Arm & Gripper Control
| Topic | Type | Description |
|---|---|---|
| `/right_arm/joint_cmd` | `std_msgs/Float64MultiArray` | Command 6 joint positions for the right arm. |
| `/left_arm/joint_cmd` | `std_msgs/Float64MultiArray` | Command 6 joint positions for the left arm. |
| `/right_gripper/cmd` | `std_msgs/Float64MultiArray` | Normalized [0-1] (0=open, 1=closed) for right gripper. |
| `/left_gripper/cmd` | `std_msgs/Float64MultiArray` | Normalized [0-1] (0=open, 1=closed) for left gripper. |
| `/joint_states` | `sensor_msgs/JointState` | Aggregated joint state (positions, velocities) of the entire robot. |

### 6.3 Camera & Perception
| Topic | Type | Description |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Raw RGB image stream (640x480). |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Raw 16-bit depth image stream (mm). |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Calibration metadata for RGB camera. |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Calibration metadata for depth camera. |
| `/camera/pan_tilt_cmd` | `std_msgs/Float64MultiArray` | Command [pan, tilt] radians. |
| `/camera/pan_tilt_state` | `std_msgs/Float64MultiArray` | Current [pan, tilt] positions. |

### 6.4 Network Bridge (Remote Viewing)
| Topic | Type | Description |
|---|---|---|
| `/camera/color/image_compressed` | `sensor_msgs/CompressedImage` | JPEG-compressed RGB stream. |
| `/camera/depth/image_compressed` | `sensor_msgs/CompressedImage` | PNG-compressed depth stream. |

### 6.5 Audio & Services
| Service | Type | Description |
|---|---|---|
| `/microphone/record` | `tidybot_msgs/srv/AudioRecord` | Service to start/stop audio recording and retrieve data. |

---

## 7. TF Frame Hierarchy

The robot publishes a full transform tree. Key frames include:
- `odom`: The global origin for odometry.
- `base_link`: The center of the mobile base.
- `camera_link`: Base of the pan-tilt camera.
- `camera_color_optical_frame`: Optical frame for RGB image processing.
- `right_arm/base_link` & `left_arm/base_link`: Bases of the respective arms.
- `right_gripper_link` & `left_gripper_link`: End-effector frames.
