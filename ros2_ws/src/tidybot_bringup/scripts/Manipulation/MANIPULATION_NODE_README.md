# manipulation_node.py — Detailed Flow

ME326 Collaborative Robotics — Group 6

## Overview

`manipulation_node.py` is a clean grasp-and-lift executor that works for **both simulation (MuJoCo) and real hardware** using a single unified code path. It replaces the older `modular_manipulation_executor_node.py` which had divergent sim/real paths, wrong topics for real hardware, and accumulated dead code.

### Why One Code Path Works

Both the MuJoCo bridge (sim) and `arm_wrapper_node` (real) listen on the **same topics**:

| Topic | Message Type | Description |
|---|---|---|
| `/{arm}_arm/joint_cmd` | `Float64MultiArray` | 6 joint positions (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate) |
| `/{arm}_gripper/cmd` | `Float64MultiArray` | Gripper command: `0.0` = open, `1.0` = closed |

No `sim_mode` flag is needed. The same publish call reaches MuJoCo in sim or the Dynamixel servos on the real robot.

---

## System Architecture

```
                          ┌──────────────────┐
                          │    Brain Node     │
                          │  (Mete's node)    │
                          └──┬──────────┬─────┘
                             │          │
                  /arm_status│          │/task_status
                   (Int32)   │          │ (Int32)
                             ▼          ▼
┌──────────────┐    ┌────────────────────────────┐    ┌─────────────────┐
│  PointNetGPD │───▶│   manipulation_node.py      │───▶│  Motion Planner │
│  (grasp det) │    │   (ManipulationExecutor)    │◀───│ /plan_to_target │
└──────────────┘    └──────┬──────────┬──────────┘    └─────────────────┘
  /detected_grasps/pose    │          │
                           │          │
              /{arm}_arm/  │          │ /{arm}_gripper/
              joint_cmd    │          │ cmd
                           ▼          ▼
              ┌────────────────────────────────┐
              │  MuJoCo Bridge (sim)           │
              │       — OR —                   │
              │  arm_wrapper_node (real)        │
              └────────────────────────────────┘
```

---

## ROS2 Interface

### Subscribed Topics

| Topic | Type | Source | Description |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | MuJoCo bridge or robot drivers | Joint position feedback for all joints |
| `/arm_status` | `std_msgs/Int32` | Brain node | Which arm to use: `0` = left, `1` = right |
| `/task_status` | `std_msgs/Int32` | Brain node | Which task: `0` = task 1, `1` = task 3 |
| `/detected_grasps/pose` | `geometry_msgs/PoseStamped` | PointNetGPD / manual | Grasp target pose (in `base_link` frame) |
| `/grasp_planner/grasp_pose` | `geometry_msgs/PoseStamped` | Grasp planner | Alternate grasp target topic |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/right_arm/joint_cmd` | `Float64MultiArray` | Right arm 6-DOF joint positions |
| `/left_arm/joint_cmd` | `Float64MultiArray` | Left arm 6-DOF joint positions |
| `/right_gripper/cmd` | `Float64MultiArray` | Right gripper (0.0=open, 1.0=closed) |
| `/left_gripper/cmd` | `Float64MultiArray` | Left gripper (0.0=open, 1.0=closed) |
| `/manipulation/task_status` | `String` | Result published to brain node ("SUCCESS", "IK_FAIL", "TIMEOUT") |

### Service Clients

| Service | Type | Description |
|---|---|---|
| `/plan_to_target` | `tidybot_msgs/PlanToTarget` | IK solver. Called with `execute=False` (IK only — we publish joint commands ourselves) |

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `arm_name` | string | `"right"` | Initial arm (overridden by `/arm_status` from brain node) |
| `use_hardcoded_pose` | bool | `True` | If true, publishes a test grasp pose after 3s for self-testing |
| `use_motion_planner` | bool | `True` | If true, uses `/plan_to_target` for IK. If false, uses hardcoded fallback IK |

---

## State Machine Flow

The executor runs a 7-state grasp sequence. Each state advances based on **joint state feedback** (not blind timers), ensuring the arm has physically arrived before proceeding.

```
    ┌──────┐
    │ IDLE │◀─────────────────────────────────────────┐
    └──┬───┘                                          │
       │ grasp pose received                          │
       ▼                                              │
┌──────────────┐                                      │
│ OPEN_GRIPPER │  Set gripper to 0.0 (open)           │
│   Step 1/7   │  Advance when: fingers confirm open  │
│              │    OR 3s timeout                      │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│MOVE_PREGRASP │  IK solve for grasp + 8cm Z offset   │
│   Step 2/7   │  Publish joints to /{arm}_arm/joint_cmd
│              │  Advance when: joints match target    │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│  MOVE_GRASP  │  IK solve for grasp + 1.5cm offset   │
│   Step 3/7   │  Publish joints to /{arm}_arm/joint_cmd
│              │  Advance when: joints match target    │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│PAUSE_AT_GRASP│  Hold position, gripper still open    │
│   Step 4/7   │  Advance when: 1.0s elapsed           │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│CLOSE_GRIPPER │  Set gripper to 1.0 (close)           │
│   Step 5/7   │  Uses STALL DETECTION (see below)     │
│              │  Advance when: stall + firm grip hold  │
│              │    OR fully closed OR 20s timeout      │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│  MOVE_LIFT   │  IK solve for grasp + 15cm Z offset   │
│   Step 6/7   │  Gripper stays closed                  │
│              │  Advance when: joints match target     │
└──────┬───────┘                                      │
       ▼                                              │
┌──────────────┐                                      │
│     DONE     │  Publish "SUCCESS" to brain node      │
│   Step 7/7   │  Gripper stays closed                  │
│              │  Can accept new pose (for task 3)──────┘
└──────────────┘
```

---

## Gripper Stall Detection (CLOSE_GRIPPER state)

The old executor used a blind 35-second timeout for gripper closing. This node uses **stall detection** — it monitors the finger gap and detects when the gripper has stopped moving (because it's gripping an object).

### How it works:

1. **Record** finger gap (sum of both finger positions) every tick (50 Hz) into a ring buffer
2. **Wait** at least `STALL_CHECK_DELAY` (3s) before checking — gives grippers time to start moving
3. **Compare** current gap to the gap from `STALL_WINDOW` (2s) ago
4. **If** the change is less than `STALL_THRESHOLD` (0.001m) — the gripper has **stalled** on an object
5. **Hold** for `FIRM_GRIP_HOLD` (5s) while continuing to send close commands — ensures firm grip
6. **Advance** to MOVE_LIFT

### Exit conditions (whichever happens first):
- **Stall detected + firm grip hold complete** — normal case when gripping an object
- **Gripper fully closed** (`< 0.018m`) — empty grasp, closes all the way
- **Safety timeout** (20s) — fallback if stall detection never triggers

### Constants:
```python
STALL_CHECK_DELAY = 3.0      # seconds before checking for stalls
STALL_THRESHOLD = 0.001      # meters — change below this = stalled
STALL_WINDOW = 2.0           # seconds — comparison window
FIRM_GRIP_HOLD = 5.0         # seconds — hold after stall detected
GRIPPER_SAFETY_TIMEOUT = 20.0  # seconds — max wait
```

---

## Dual-Arm Support (Task 3)

For task 3 (bottle + cap), the executor supports sequential dual-arm operation:

### Flow:
1. Brain node publishes `/arm_status = 0` (left arm)
2. Brain/grasp planner publishes grasp pose for bottle
3. Executor runs full state machine with **left arm** → reaches DONE
4. Left gripper stays closed (holding bottle) — `gripper_values["left"] = 1.0` persists
5. Brain node publishes `/arm_status = 1` (right arm)
6. Brain/grasp planner publishes grasp pose for bottle cap
7. Executor runs full state machine with **right arm** → reaches DONE
8. Right gripper closes on cap — left gripper still closed on bottle

### Why this works:
- `gripper_values` is a **per-arm dict**: `{"right": 0.0, "left": 0.0}`
- The state machine only modifies `gripper_values[active_arm]`
- **Both arms' gripper values are published every tick** (50 Hz) — so the inactive arm maintains its grip
- `_arm_status_cb` only accepts arm switches when state is `IDLE` or `DONE`
- `_grasp_pose_cb` accepts new poses when state is `IDLE` or `DONE` (allows second run)

---

## Joint State Lookup

The node supports two joint state formats for sim/real compatibility:

### Index-based (sim — aggregated `/joint_states`):
```
[0]  camera_pan        [1]  camera_tilt
[2]  right_waist       [3]  right_shoulder    [4]  right_elbow
[5]  right_forearm_roll [6] right_wrist_angle [7]  right_wrist_rotate
[8]  right_left_finger [9]  right_right_finger
[10] left_waist        [11] left_shoulder     [12] left_elbow
[13] left_forearm_roll [14] left_wrist_angle  [15] left_wrist_rotate
[16] left_left_finger  [17] left_right_finger
```

### Name-based (real — per-arm `/joint_states`):
Falls back to looking up joints by name (e.g., `right_waist`, `right_shoulder`, ...) when index-based lookup fails. This handles the case where the real robot publishes separate joint state messages per arm.

---

## Motion Planning

We use the **built-in motion planner** (`/plan_to_target` service from `tidybot_ik`) for all IK solving. The planner handles inverse kinematics, singularity checks, and collision validation. The only difference from the default usage is that we set `execute=False` so the planner returns the joint solution to us instead of publishing it directly. This lets us publish on the unified `Float64MultiArray` topic that works for both sim and real (the planner's built-in publish uses `ArmCommand`, which only works in sim).

### With planner (`use_motion_planner=true`):
1. Calls `/plan_to_target` with `execute=False` — planner solves IK and returns joint positions
2. Extracts 6 joint positions from response (same solution the planner would have published)
3. We publish as `Float64MultiArray` to `/{arm}_arm/joint_cmd` (unified sim/real topic)
4. In sim: MuJoCo position actuators provide physics-based smoothing
5. On real: `arm_wrapper_node` handles velocity limiting (max 1.0 rad/s) at 50 Hz

### Without planner (`use_motion_planner=false`):
Uses `_hardcoded_ik()` — simple z-based joint selection with arm-aware waist angle:
- Right arm waist: `-pi/2` (points left)
- Left arm waist: `+pi/2` (points right)

---

## Usage

### Sim with hardcoded test pose:
```bash
ros2 launch tidybot_bringup sim.launch.py scene:=scene_pickup.xml
ros2 run tidybot_bringup manipulation_node.py --ros-args -p use_hardcoded_pose:=true -p use_motion_planner:=true
```

### Sim with grasp planner:
```bash
ros2 launch tidybot_bringup sim.launch.py scene:=scene_pickup.xml
ros2 run tidybot_bringup manipulation_node.py --ros-args -p use_hardcoded_pose:=false -p use_motion_planner:=true
# Then trigger grasp detection or publish manually:
ros2 topic pub --once /detected_grasps/pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'base_link'}, pose: {position: {x: -0.10, y: -0.50, z: 0.29}, orientation: {w: 0.1228, x: 0.6964, y: 0.1228, z: -0.6964}}}"
```

### Real robot:
```bash
ros2 launch tidybot_bringup real.launch.py
ros2 run tidybot_bringup manipulation_node.py --ros-args -p use_hardcoded_pose:=false -p use_motion_planner:=true
```

### Switch arm via brain node:
```bash
# Switch to left arm
ros2 topic pub --once /arm_status std_msgs/msg/Int32 "{data: 0}"
# Switch to right arm
ros2 topic pub --once /arm_status std_msgs/msg/Int32 "{data: 1}"
```

### Set task:
```bash
# Task 1 (single arm retrieval)
ros2 topic pub --once /task_status std_msgs/msg/Int32 "{data: 0}"
# Task 3 (bottle + cap, dual arm)
ros2 topic pub --once /task_status std_msgs/msg/Int32 "{data: 1}"
```

---

## Key Differences from modular_manipulation_executor_node.py

| Feature | Old Executor | New Executor |
|---|---|---|
| Sim/real code path | Separate (`if sim_mode`) | Unified (same topics) |
| Arm topics | `/right_arm/cmd` (ArmCommand) | `/{arm}_arm/joint_cmd` (Float64MultiArray) |
| Gripper convention | -1.0 open / 1.0 closed | 0.0 open / 1.0 closed |
| Gripper close logic | 35s blind timeout | Stall detection + firm grip hold |
| Motion planner | `execute=True` (planner publishes) | `execute=False` (we publish) |
| Real hardware msgs | JointGroupCommand (interbotix) | Float64MultiArray (same as sim) |
| `sim_mode` param | Required | Removed |
| Line count | ~836 lines | ~686 lines |
| Dead code | CHANGE 1-15 markers, commented blocks | None |
