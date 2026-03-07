# Simple Centroid-Based Grasping

This project provides a simple, heuristic-based grasp planner as an alternative to more complex ML-based approaches like PointNetGPD. It calculates an object's centroid from a point cloud and generates either a top-down or side-approach grasp pose.

## Advantages
- **Lightweight**: No heavy machine learning models or GPU requirements.
- **Fast**: Near-instantaneous pose generation.
- **Deterministic**: Easy to debug and predict behavior.
- **Robust**: Works well for well-isolated objects on flat surfaces.

## Architecture

The simple grasping pipeline consists of:
1.  **Point Cloud Generation**: Converts RGB-D data from the RealSense camera into a `PointCloud2` using `depth_image_proc`.
2.  **Foreground Segmentation**: Filters points based on height relative to the table surface to isolate the object.
3.  **Centroid Calculation**: Computes the 3D mean of the isolated object points.
4.  **Heuristic Application**: Generates a grasp pose based on the chosen strategy (`top` or `side`).
5.  **Motion Planning**: Optionally calls the `/plan_to_target` service to execute the grasp.

If desired, approach can also be augmented to accept segmentation input rather than bounding box.

## Installation

Ensure you have the necessary dependencies. The `scipy` library is required for coordinate transformations.

```bash
uv add scipy
```

## Usage

### 1. Launch the Simple Grasping Pipeline
This launch file starts the point cloud processing and the simple grasp planner node.

**For Simulation:**
```bash
ros2 launch tidybot_bringup simple_grasp.launch.py use_sim:=true grasp_type:=top
```

**For Real Robot:**
```bash
ros2 launch tidybot_bringup simple_grasp.launch.py use_sim:=false grasp_type:=side
```

### 2. Trigger a Grasp
To trigger the grasp generation, publish a `RegionOfInterest` (ROI) message. This tells the planner to process the current point cloud.

```bash
ros2 topic pub --once /grasp_pose_request_roi sensor_msgs/msg/RegionOfInterest "{x_offset: 240, y_offset: 200, width: 160, height: 160, do_rectify: false}"
```

### 3. Parameters
You can customize the behavior of the planner via ROS 2 parameters:
- `grasp_type`: `top` (default) or `side`.
- `table_height_buffer`: Height offset (meters) above the table to consider points as part of the object.
- `height_adjust`: Z-offset to adjust the final gripper height.
- `send_plan_request`: If `true`, the node will automatically call the motion planner after generating a pose.

## Comparison with PointNetGPD
While PointNetGPD is better at finding optimal grasp points on complex geometries, the Simple Grasp Planner is often more reliable for standard pick-and-place tasks with simple objects, as it avoids the potential failures of neural network inference.
