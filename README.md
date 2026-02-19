# ME 326: Collaborative Robotics

This repository contains the project for Stanford's ME 326: Collaborative Robotics course (Winter 2026). The goal of this project is to make the TidyBot2 mobile robot an effective collaborator and assistant for a human counterpart. You will use natural language to communicate tasks to the robot, which will then use perception to understand its environment and control its motion to complete the task.

## 1. Project Goal & Core Tasks

The goal of this project is to make a mobile robot an effective collaborator and assistant for a human counterpart. The platform used is the TidyBot++ base with a custom bimanual manipulation setup using 6-DoF WidowX Arms, an Intel RealSense depth camera, and a LiDAR for localization.

Students will use natural communication methods (e.g., natural language) to communicate tasks to the robot. The robot will use perception to understand the environment and control its motion to complete the requested task.

Every team must complete three tasks:

1.  **Object Retrieval:** Use language to request the robot to retrieve a specific object and bring it to a location (e.g., "locate the apple in the scene and retrieve it").
2.  **Sequential Task:** Use language to request a series of actions, like placing one object into another (e.g., "find the red apple and place it in the basket").
3.  **Group Chosen Task:** A team-defined task that involves perception, planning, and control.

### Group Chosen Task: Liquid Transport

For our third task, the Tidybot will transport a “liquid” from one vessel to another, using voice commands.

*   **Baseline:** Pick up a known container and pour a liquid (or stand-in material like sand) into a known second container.
*   **Potential Extensions:** Unscrew a lid before pouring; pour a specific amount of liquid based on language input.

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
# 1. Clone the repo (if you are in windows, we highly, highly recommend WSL Bash)
git clone https://github.com/armlabstanford/collaborative-robotics-2026.git
cd collaborative-robotics-2026

# 2. Pull the pre-built image
docker pull peasant98/tidybot2:humble

# 3. Run with the repo mounted to sync local changes
# Note: We are mounting the parent directory to make both the main repo and submodule available
docker run -p 6080:80 --shm-size=2g \
    -v $(pwd):/home/ubuntu/Desktop/collaborative \
    peasant98/tidybot2:humble

# 4. Access the container's desktop via browser: http://127.0.0.1:6080/

# 5. In the container's terminal, navigate to the project and run the setup script
cd /home/ubuntu/Desktop/collaborative/collaborative-robotics-2026
./setup.sh
```

**Syncing updates:** With the volume mount, any `git pull` on your host machine will immediately reflect inside the running container.

**Available commands in container:**
| Command | Description |
|---------|-------------|
| `tidybot-sim` | MuJoCo standalone simulation |
| `tidybot-ros` | ROS2 + RViz + MuJoCo |
| `tidybot-ros-no-rviz` | ROS2 + MuJoCo (no RViz) |
| `tidybot-test-base` | Test base movement |
| `tidybot-test-arms` | Test arm control |

### 3.2 Option B: Native Ubuntu 22.04

If you have Ubuntu 22.04 (native install, dual-boot, or VM), use the setup script:

```bash
# 1. Clone the repository
git clone https://github.com/armlabstanford/collaborative-robotics-2026.git
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

## Quick Start SIMULATION

### Option 1: Standalone MuJoCo Simulation (No ROS2)

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
**Launch Options:**
```bash
# Disable RViz
ros2 launch tidybot_bringup sim.launch.py use_rviz:=false

# Disable MuJoCo viewer
ros2 launch tidybot_bringup sim.launch.py show_mujoco_viewer:=false
```

## Quick Start REAL

NOTE: You do not need to re-run "colcon build" for every new terminal, re-build is only necessary whenver source code was modified. Make sure to re-source for new terminals.

**Terminal 1: Launch Initialization of TidyBot**
```bash
cd ros2_ws
source setup_env.bash 

# Launch bringup
ros2 launch tidybot_bringup real.launch.py
```

**Terminal 2: Run Command Script**
```bash
cd ros2_ws
source setup_env.bash 

# Test base movement
ros2 run tidybot_bringup test_base_real.py

# Test bimanual arms
ros2 run tidybot_bringup test_arms_real.py
```

## Repository Structure

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
-   **Grasping:** `PointNetGPD` is used for grasp pose detection on the CPU. Run `./setup_grasping.sh` to install dependencies. See [docs/grasp_detection_setup.md](docs/grasp_detection_setup.md) for details.
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

### 6.2 Adding New Scripts and Launch Files

To maintain project organization and ensure new ROS2 nodes and launch files are correctly integrated:

#### 6.2.1 Python Scripts (Nodes)

1.  **Placement**: New Python scripts (ROS2 nodes) should be placed in subdirectories within `ros2_ws/src/tidybot_bringup/scripts/`. For example:
    ```
    ros2_ws/src/tidybot_bringup/scripts/
    ├── vision/
    │   └── my_vision_node.py
    └── manipulation/
        └── my_manipulation_node.py
    ```
2.  **Executable Configuration**: To make your new Python script executable as a ROS2 run command (`ros2 run tidybot_bringup <script_name>.py`), you **must** add it to `ros2_ws/src/tidybot_bringup/CMakeLists.txt`. Locate the `install(PROGRAMS ...)` block and add your script's relative path:
    ```cmake
    install(PROGRAMS
      scripts/vision/my_vision_node.py
      scripts/manipulation/my_manipulation_node.py
      # ... existing scripts ...
      DESTINATION lib/${PROJECT_NAME}
    )
    ```
    After modifying `CMakeLists.txt`, rebuild your workspace: `cd ros2_ws && colcon build`.

#### 6.2.2 Launch Files

1.  **Placement**: New launch files (e.g., `my_task_launch.py`) should be placed in `ros2_ws/src/tidybot_bringup/launch/`.
2.  **Example Structure**: You can adapt existing launch files (e.g., `sim.launch.py`) as a template. A minimal launch file looks like this:
    ```python
    from launch import LaunchDescription
    from launch_ros.actions import Node

    def generate_launch_description():
        return LaunchDescription([
            Node(
                package='your_package_name', # e.g., 'tidybot_bringup'
                executable='your_node_script_name.py', # e.g., 'my_vision_node.py'
                name='my_node_name',
                output='screen',
                parameters=[{'param_name': 'param_value'}] # Optional parameters
            )
        ])
    ```
    To run your new launch file: `ros2 launch tidybot_bringup my_task_launch.py`.

### 6.3 Remote Hardware Access
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

## 7. Troubleshooting

**MuJoCo rendering issues:**
```bash
sudo apt install -y mesa-utils
glxinfo | grep "OpenGL version"
```

**Python import errors:**
```bash
# Always source environment first
source ros2_ws/setup_env.bash
```

**colcon build fails:**
```bash
# Ensure ROS2 is sourced before building
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build
```

## 8. Resources & Authors

**Authors:**
*   Mete Gumusayak
*   Ottavia Personeni
*   Luke Yuen
*   Aditya Kothari
*   Yash Rampuria
*   Rohit Arumugam
*   Andrew Pasco
*   Alex Qiu & Matt Strong (Stanford ARM Lab)

**Course:** ME 326: Collaborative Robotics, Winter 2026 (Prof. Monroe Kennedy)

**Key Resources:**
-   [TidyBot2 Project Page](https://tidybot2.github.io/)
-   [MuJoCo Documentation](https://mujoco.readthedocs.io/)
-   [uv Documentation](https://docs.astral.sh/uv/)
-   [Interbotix WX250s Docs](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250s.html)
-   [TidyBot Paper](https://arxiv.org/abs/2305.05658)
-   [TidyBot2 Paper](https://arxiv.org/pdf/2412.10447)

---
*This project is for ME 326 at Stanford University, Winter 2026.*
