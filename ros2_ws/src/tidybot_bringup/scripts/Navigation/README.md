# ME 326 Navigator — Setup Instructions for Claude Code

## Context

This is for the ME 326 Collaborative Robotics class at Stanford. We're building a Navigator node (Node 2) for the TidyBot2 robot. The class repo is at:
https://github.com/armlabstanford/collaborative-robotics-2026

## What Needs to Happen (in order)

### Step 1: GitHub Setup

1. Install GitHub CLI if not present: `sudo apt install gh`
2. Authenticate: `gh auth login` (follow prompts, use browser auth)
3. Fork and clone: `gh repo fork armlabstanford/collaborative-robotics-2026 --clone`
4. `cd collaborative-robotics-2026`
5. Set up upstream: `git remote add upstream https://github.com/armlabstanford/collaborative-robotics-2026.git` (gh may do this automatically)
6. Create navigation branch: `git checkout -b navigation`

### Step 2: Install Navigator Files

Copy the following files into `ros2_ws/src/tidybot_bringup/scripts/`:

- `navigator.py` — the Navigator ROS 2 node
- `vision_nav.py` — the Vision node for sim testing

Make both executable:
```bash
chmod +x ros2_ws/src/tidybot_bringup/scripts/navigator.py
chmod +x ros2_ws/src/tidybot_bringup/scripts/vision_nav.py
```

### Step 3: Update CMakeLists.txt

Edit `ros2_ws/src/tidybot_bringup/CMakeLists.txt`.
Find the `install(PROGRAMS ...)` block and add these two lines:
```
scripts/navigator.py
scripts/vision_nav.py
```

### Step 4: Build and Source

```bash
cd ros2_ws
colcon build --packages-select tidybot_bringup
source setup_env.bash
```

### Step 5: First Test (Navigator only, fake detections)

Terminal 1:
```bash
cd ros2_ws && source setup_env.bash
ros2 launch tidybot_bringup sim.launch.py
```

Terminal 2:
```bash
cd ros2_ws && source setup_env.bash
ros2 run tidybot_bringup navigator.py
```

Terminal 3 — test scan:
```bash
cd ros2_ws && source setup_env.bash
ros2 topic pub --once /navigator/command std_msgs/msg/String "{data: 'SCAN'}"
```

Terminal 3 — test approach with fake detection:
```bash
ros2 topic pub /object_detection geometry_msgs/msg/Point "{x: 400.0, y: 240.0, z: 5000.0}" --rate 10 &
ros2 topic pub --once /navigator/command std_msgs/msg/String "{data: 'APPROACH'}"
```

### Step 6: Commit

```bash
git add ros2_ws/src/tidybot_bringup/scripts/navigator.py
git add ros2_ws/src/tidybot_bringup/scripts/vision_nav.py
git add ros2_ws/src/tidybot_bringup/CMakeLists.txt
git commit -m "add navigator and vision_nav nodes for sim testing"
git push origin navigation
```

## File Descriptions

- **navigator.py**: ROS 2 node that drives the TidyBot2 base. Subscribes to /object_detection (pixel coords from Vision) and /apriltag_pose (PnP pose). Publishes /cmd_vel (base velocity). State machine: IDLE → SCAN → APPROACH → ARRIVED → RETURN. Built on HW2 P7 proportional controller pattern.

- **vision_nav.py**: ROS 2 node for sim testing only. Subscribes to camera images, runs HSV color detection (HW2 P5) and ArUco/AprilTag detection + solvePnP (HW2 P6). Publishes /object_detection and /apriltag_pose. Will be replaced by teammates' YOLO/Gemini pipeline on real robot.

## ROS 2 Topic Interface

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| /cmd_vel | geometry_msgs/Twist | Navigator publishes | Base velocity |
| /odom | nav_msgs/Odometry | Navigator subscribes | Robot pose feedback |
| /navigator/command | std_msgs/String | Brain publishes | SCAN, APPROACH, RETURN, STOP |
| /navigator/status | std_msgs/String | Navigator publishes | SCANNING, ARRIVED, RETURNED, etc |
| /object_detection | geometry_msgs/Point | Vision publishes | x=pixel_x, y=pixel_y, z=bbox_area |
| /apriltag_pose | geometry_msgs/Pose | Vision publishes | PnP pose (position.z = depth in m) |
| /camera/color/image_raw | sensor_msgs/Image | Camera publishes | RGB frames |

## Important Notes

- The camera topic name may differ in sim. Run `ros2 topic list | grep -i cam` to find it. Update the subscription in vision_nav.py if needed.
- HSV ranges in vision_nav.py are defaults. May need tuning for whatever objects are in the sim scene.
- Camera intrinsics (CAMERA_MATRIX) in vision_nav.py are RealSense D435 defaults. May need adjustment for the sim camera.
- All gains (KP_ANGULAR, KP_FORWARD, etc.) will need tuning. Start with defaults, adjust based on observed behavior.
