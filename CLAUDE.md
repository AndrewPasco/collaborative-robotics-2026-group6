# CLAUDE.md - Development Guide for TidyBot2

## Development Rules for Claude

1.  **Always commit and push changes** - After making code changes, add, commit, and push to the repository (unless the user specifies otherwise).
2.  **Use descriptive commit messages** - Summarize what was changed and why.
3.  **Test before committing** - Ensure changes work before pushing.
4.  **Update documentation** - Keep the project's `README.md` and this `CLAUDE.md` file up to date with any significant changes.

## 1. Project Goal: ME 326 Collaborative Robot

The goal of this project is to make the TidyBot2 mobile robot an effective collaborator and assistant for a human counterpart for Stanford's ME 326 course. You will use natural language to communicate tasks to the robot, which will then use perception to understand its environment and control its motion to complete the task.

The code for this project is located in the `collaborative-robotics-2026` submodule.

### 1.1 Core Tasks

Every team must complete three tasks:

1.  **Object Retrieval:** A user will ask the robot to retrieve a specific object and bring it to a location (e.g., "locate the apple in the scene and retrieve it").
2.  **Sequential Task:** A user will ask the robot to perform a sequence of actions, like placing one object into another (e.g., "find the red apple and place it in the basket").
3.  **Group Chosen Task:** A team-defined task that involves perception, planning, and control. An example is having the robot transport a "liquid" from one vessel to another, which might involve unscrewing a lid, pouring, and measuring.

## 2. System Overview: TidyBot2 Bimanual Manipulator

### 2.1 Robot Configuration

-   **Mobile Base:** 3 DOF Holonomic Base (x, y, theta)
-   **Arms:** 2x WidowX WX250s (6 DOF each)
-   **Grippers:** 2x Interbotix Gripper (Prismatic Fingers)
-   **Camera:** Pan-tilt Intel RealSense D435 (RGB + Depth)
-   **LiDAR:** RPLIDAR for localization and navigation.

### 2.2 Software Stack

-   **OS:** Ubuntu 22.04
-   **Middleware:** ROS 2 Humble
-   **Simulation:** MuJoCo
-   **Python Environment:** `uv`

## 3. Getting Started

All commands should be run from within the `collaborative-robotics-2026` directory.

### 3.1 Option A: Docker (Recommended for Any OS)

Use the pre-built Docker image with VNC desktop access.

```bash
# 1. Navigate into the submodule
cd collaborative-robotics-2026

# 2. Pull the pre-built image
docker pull peasant98/tidybot2:humble

# 3. Run with the repo mounted to sync local changes
# Note: We are mounting the parent directory to make both the main repo and submodule available
docker run -p 6080:80 --shm-size=2g 
    -v $(pwd)/..:/home/ubuntu/Desktop/collaborative 
    peasant98/tidybot2:humble

# 4. Access the container's desktop via browser: http://127.0.0.1:6080/

# 5. In the container's terminal, navigate to the project and run the setup script
cd /home/ubuntu/Desktop/collaborative/collaborative-robotics-2026
./setup.sh
```

### 3.2 Option B: Native Ubuntu 22.04

```bash
# 1. Navigate into the submodule
cd collaborative-robotics-2026

# 2. Run the setup script
# This will install ROS2 Humble, system dependencies, Python packages, and build the workspace.
./setup.sh
```
**Important:** After setup, source the environment in every new terminal:
```bash
cd collaborative-robotics-2026/ros2_ws
source setup_env.bash
```

## 4. Running the Code

### 4.1 Standalone MuJoCo Simulation (No ROS2)
```bash
cd collaborative-robotics-2026/simulation/scripts

# Bimanual arm demo with camera control
uv run python test_move.py

# Object manipulation demo
uv run python pick_up_block.py
```

### 4.2 Full ROS2 Simulation

**Terminal 1 - Launch simulation:**
```bash
cd collaborative-robotics-2026/ros2_ws
source setup_env.bash
ros2 launch tidybot_bringup sim.launch.py
```
This opens RViz2 and the MuJoCo viewer.

**Terminal 2 - Run test scripts:**
```bash
cd collaborative-robotics-2026/ros2_ws
source setup_env.bash

# Test base, arms, camera, or run a state machine example
ros2 run tidybot_bringup test_base_sim.py
ros2 run tidybot_bringup test_arms_sim.py
ros2 run tidybot_bringup example_state_machine.py
```

## 5. Technical Deep Dive

### 5.1 Project Structure
```
collaborative-robotics-2026/
├── simulation/              # Standalone MuJoCo simulation & assets
└── ros2_ws/                 # ROS2 workspace
    ├── setup_env.bash       # Environment setup (source this!)
    └── src/
        ├── tidybot_bringup/     # Launch files & test scripts
        ├── tidybot_client/      # Remote client utilities & DDS configs
        ├── tidybot_control/     # Arm/base/pan-tilt controllers
        ├── tidybot_description/ # URDF/XACRO robot model
        ├── tidybot_ik/          # Motion planning & IK (using mink)
        ├── tidybot_msgs/        # Custom messages & services
        └── tidybot_mujoco_bridge/ # MuJoCo-ROS2 bridge
```

### 5.2 Key ROS2 Topics & Messages

| Topic | Type | Description |
|---|---|---|
| `/cmd_vel` | geometry_msgs/Twist | Base velocity commands |
| `/left_arm/command` | ArmCommand | Left arm joint positions + duration |
| `/right_arm/command` | ArmCommand | Right arm joint positions + duration |
| `/left_gripper/command` | GripperCommand | Left gripper (0=open, 1=closed) |
| `/right_gripper/command`| GripperCommand | Right gripper |
| `/joint_states` | sensor_msgs/JointState | All joint positions/velocities |

**`ArmCommand.msg`**:
```
float64[6] joint_positions  # [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
float64 duration            # Movement time in seconds
```

### 5.3 Core Software Libraries

-   **Perception (VLMs):** You are encouraged to use Visual Language Models for object detection from language commands.
    -   Google Gemini can be used with provided cloud credits.
    -   Local models like `YOLOv11` with `CLIP` are also excellent options.
-   **Grasping:** `GraspAnything` can be used to propose candidate grasp poses on objects.
-   **Motion Planning (IK):** The `tidybot_ik` package uses `mink` for lightweight inverse kinematics and trajectory optimization, as an alternative to MoveIt2.

## 6. Development Workflow

### 6.1 Common Commands
```bash
# Rebuild workspace after code changes
cd collaborative-robotics-2026/ros2_ws && colcon build

# Rebuild a specific package
cd collaborative-robotics-2026/ros2_ws && colcon build --packages-select tidybot_control

# Check ROS2 topics and messages
ros2 topic list
ros2 topic echo /joint_states
```

### 6.2 Remote Hardware Access
To connect to the physical robot:

1.  **Install Cyclone DDS (one time):**
    ```bash
    sudo apt install ros-humble-rmw-cyclonedds-cpp
    ```
2.  **Configure Environment:** Add these lines to your `~/.bashrc` file. The `ROS_DOMAIN_ID` will be provided by the teaching team.
    ```bash
    export ROS_DOMAIN_ID=<ID>
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ```
3.  **SSH into Robot:** The password is `locobot`.
    ```bash
    ssh -X locobot@<ROBOT_IP>
    ```
4.  **Start Robot Drivers:** On the robot, run `ros2 launch tidybot_bringup robot.launch.py`.
5.  **Verify Connection:** On your machine, run `ros2 topic list` and check for robot topics.

## 7. Resources & Authors

-   **Authors:** Alex Qiu & Matt Strong (Stanford ARM Lab)
-   **Course:** ME 326: Collaborative Robotics, Winter 2026 (Prof. Monroe Kennedy)
-   **Key Resources:**
    -   [TidyBot2 Project Page](https://tidybot2.github.io/)
    -   [MuJoCo Documentation](https://mujoco.readthedocs.io/)
    -   [uv Documentation](https://docs.astral.sh/uv/)
    -   [Interbotix WX250s Docs](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250s.html)
