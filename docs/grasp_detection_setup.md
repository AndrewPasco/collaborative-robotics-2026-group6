# Grasp Detection Setup (PointNetGPD)

This project uses [PointNetGPD](https://github.com/lianghongzhuo/PointNetGPD) for grasp pose detection. This is a lightweight, Python-based approach that does not require a GPU, making it suitable for the robot's onboard Intel NUC.

## Architecture

The grasping pipeline consists of:
1.  **Input**: RGB-D from RealSense, converted to `PointCloud2` by `depth_image_proc`.
2.  **Node**: `pointnet_gpd_node.py` (ROS2 wrapper).
3.  **Sampling**: `GpgGraspSamplerPcl` (from `dex-net` library) uses geometric heuristics (surface normals, curvature) to generate candidate grasps.
4.  **Evaluation**: `PointNet` (PyTorch) evaluates each candidate and assigns a score.
5.  **Motion Planning**: The node automatically transforms the best grasp to the `base_link` frame and calls the `/plan_to_target` service to execute the pick.

## Mechanism of Operation

### 1. Candidate Generation (Sampling)
Grasp candidates are generated geometrically using the **GPG (Grasp Pose Generator)** algorithm via `dex-net`.
*   **Surface Analysis**: The system samples points on the object surface and computes local surface normals and principal curvature axes using Open3D.
*   **Collision Checking**: A geometric box model of the **WidowX 250s** gripper is checked against the point cloud to filter out grasps that would result in collisions with the object or the table.

### 2. Scoring (PointNetGPD)
Once geometrically valid candidates are found, they are scored for robustness:
*   **Cropping**: For each candidate, the global point cloud is transformed into the gripper's local frame. Points falling within the gripper's closing region are cropped.
*   **Inference**: This local point cloud (resampled to 500 points) is fed into the **PointNet** neural network.
*   **Ranking**: The network outputs a score. The system ranks all candidates and selects the best position.

### 3. Integrated Execution
*   **Coordinate Transformation**: The detected pose is transformed from the camera optical frame to the robot's `base_link` using `tf2_ros`.
*   **Top-Down Override**: For maximum reliability in picking from flat surfaces, the system currently overrides the GPD orientation with a fixed **top-down (fingers down)** approach.
*   **Planner Call**: The node sends a synchronous planning and execution request to the `/plan_to_target` service.

## Installation Instructions

We have vendored the patched PointNetGPD code into `ros2_ws/src/third_party/pointnet_gpd` to ensure stability and portability.

### 1. Run Setup Script

We provide a dedicated script to install the necessary system and python dependencies (including PyTorch CPU and Open3D).

```bash
./setup_grasping.sh
```

### 2. Verify Installation
Ensure the ROS2 node can find the modules:

```bash
source ros2_ws/setup_env.bash
ros2 run tidybot_bringup pointnet_gpd_node.py --help
```

## Usage

### 1. Launch the Bridge
The bridge handles point cloud generation and the GPD node. It supports both simulation and real hardware.

**For Simulation (MuJoCo):**
```bash
ros2 launch tidybot_bringup gpd_bridge.launch.py use_sim:=true
```

**For Real Robot (RealSense):**
```bash
ros2 launch tidybot_bringup gpd_bridge.launch.py use_sim:=false
```
*Note: On the real robot, ensure `realsense2_camera` is launched with `align_depth.enable:=true`.*

### 2. Trigger a Grasp
To trigger a grasp detection, publish a `RegionOfInterest` message. This crops the point cloud to the specified pixel coordinates before processing, which is essential for focusing on a specific object and improving inference speed.

```bash
ros2 topic pub --once /grasp_pose_request_roi sensor_msgs/msg/RegionOfInterest "{x_offset: 200, y_offset: 200, width: 200, height: 200, do_rectify: false}"
```

### 3. Visualize
Open RViz and visualize the `/detected_grasps/pose` topic to see the original detected pose, and monitor the terminal for the motion planner's success message.

## Technical Details (For Maintainers)

**Patches Applied to PointNetGPD (`lianghongzhuo/PointNetGPD`):**
1.  **`scipy.random` deprecation**: Replaced with `np.random`.
2.  **Open3D Vector3dVector**: Fixed type mismatch (float32 vs float64).
3.  **Depth Alignment**: The MuJoCo bridge (`mujoco_bridge_node.py`) includes a custom OpenCV warping step to ensure simulated depth is perfectly registered to the RGB frame, mimicking hardware alignment.
