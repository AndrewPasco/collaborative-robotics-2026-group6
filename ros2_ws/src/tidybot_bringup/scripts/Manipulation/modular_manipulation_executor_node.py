#!/usr/bin/env python3
"""
manipulation_executor_node.py

ME326 Collaborative Robotics – Group 6
Authors: LY, OP (Vision + Manipulation subteam)

This ROS 2 node is the "brain calls this to grasp" entry point.
It receives a target grasp pose (PoseStamped) and executes a full
grasp → lift sequence using a STATE MACHINE driven by a 50 Hz timer
(same pattern as test_arms_sim.py).

ARCHITECTURE:
  50 Hz control loop publishes gripper every tick and checks state.
  State transitions are based on JOINT STATE FEEDBACK, not timers:
    - Arm moves advance when actual joint positions match the target
    - Gripper close advances when finger joints confirm closure
  This guarantees the arm has physically arrived before we proceed.

  States:
    IDLE            – waiting for a grasp pose
    OPEN_GRIPPER    – opening gripper before approach
    MOVE_GRASP      – arm moving to grasp position
    PAUSE_AT_GRASP  – settling at grasp pose (gripper still open)
    CLOSE_GRIPPER   – closing gripper on object
    MOVE_LIFT       – lifting object
    DONE            – sequence complete, publish status

MODES:
  sim_mode=True   → hardcoded IK fallback + ArmCommand on /right_arm/cmd
  sim_mode=False  → PlanToTarget service + JointGroupCommand on real hardware

  use_hardcoded_pose=True  → publishes a test pose after 3s
  use_hardcoded_pose=False → listens to /grasp_planner/grasp_pose from grasp_planner_node.py

TOPICS USED:
  - /right_arm/cmd                     (tidybot_msgs/ArmCommand)     – sim arm commands
  - /right_arm/commands/joint_group    (JointGroupCommand)           – real arm commands
  - /right_gripper/cmd                 (std_msgs/Float64MultiArray)  – gripper commands
  - /joint_states                      (sensor_msgs/JointState)      – feedback
  - /planned_grasp                     (geometry_msgs/PoseStamped)   – grasp target (legacy)
  - /grasp_planner/grasp_pose          (geometry_msgs/PoseStamped)   – grasp target (from grasp_planner_node)
  - /manipulation/task_status          (std_msgs/String)             – result

GRIPPER CONVENTION:
  -1.0 = fully OPEN
   1.0 = fully CLOSED

USAGE:
  # Sim mode with hardcoded pose (default):
  python3 manipulation_executor_node.py

  # Sim mode, listening to grasp planner:
  python3 manipulation_executor_node.py --ros-args -p use_hardcoded_pose:=false

  # Real mode with planner + hardcoded test pose:
  python3 manipulation_executor_node.py --ros-args -p sim_mode:=false -p use_motion_planner:=true

  # Real mode with planner + grasp planner node:
  python3 manipulation_executor_node.py --ros-args -p sim_mode:=false -p use_motion_planner:=true -p use_hardcoded_pose:=false
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import time
import copy

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray
from tidybot_msgs.msg import ArmCommand
from tidybot_msgs.srv import PlanToTarget

# ── CHANGE 1: Import real hardware message type (safe import) ────────────────
try:
    from interbotix_xs_msgs.msg import JointGroupCommand
    HAS_INTERBOTIX = True
except ImportError:
    HAS_INTERBOTIX = False
# ── END CHANGE 1 ────────────────────────────────────────────────────────────


# =============================================================================
#  CONSTANTS
# =============================================================================

DEFAULT_ARM = "right"

# Gripper values (tested manually on the MuJoCo sim)
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# Vertical offsets
PREGRASP_Z_OFFSET = 0.08   # 8 cm above the grasp pose
LIFT_HEIGHT = 0.15          # 15 cm lift after grasping

# Arm command duration (interpolation time sent to arm_controller)
MOVE_DURATION = 2.0

# ── CHANGE 2: Add real hardware duration ─────────────────────────────────────
MOVE_DURATION_REAL = 3.0    # slower for real hardware safety
# ── END CHANGE 2 ────────────────────────────────────────────────────────────

# ── Joint state feedback thresholds ──────────────────────────────────────────
ARM_ARRIVAL_TOLERANCE = 0.05     # rad – how close joints must be to target
GRIPPER_OPEN_THRESHOLD = 0.03    # finger position above this = open enough
GRIPPER_CLOSED_THRESHOLD = 0.018 # finger position below this = fully closed
PAUSE_AT_GRASP_SECS = 1        # seconds to hold still at grasp before closing

# ── CHANGE 3: Add gripper min wait to prevent false confirmations ────────────
GRIPPER_MIN_WAIT = 2.0           # min seconds before checking gripper feedback
# ── END CHANGE 3 ────────────────────────────────────────────────────────────

# Safety timeouts (disabled – set very high so we rely on joint feedback only)
ARM_MOVE_TIMEOUT = 10000.0
GRIPPER_TIMEOUT = 10000.0

# ── Joint state indices (SIM ONLY – real mode uses named joints) ─────────────
# [0] camera_pan  [1] camera_tilt
# [2] right_waist [3] right_shoulder [4] right_elbow
# [5] right_forearm_roll [6] right_wrist_angle [7] right_wrist_rotate
# [8] right_left_finger  [9] right_right_finger
RIGHT_ARM_INDICES = [2, 3, 4, 5, 6, 7]
RIGHT_FINGER_LEFT = 8
RIGHT_FINGER_RIGHT = 9

# ── CHANGE 4: Add real hardware joint names + orientation constant ───────────
RIGHT_ARM_JOINT_NAMES = [
    'right_waist', 'right_shoulder', 'right_elbow',
    'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate'
]

# Fingers-down orientation from test_planner_sim.py / test_planner_real.py
ORIENT_FINGERS_DOWN = (0.5, 0.5, 0.5, -0.5)  # (qw, qx, qy, qz)
# ── END CHANGE 4 ────────────────────────────────────────────────────────────

# Reason codes
REASON_SUCCESS = "SUCCESS"
REASON_IK_FAIL = "IK_FAIL"
REASON_TIMEOUT = "TIMEOUT"
REASON_EXEC_FAIL = "EXEC_FAIL"


# =============================================================================
#  NODE
# =============================================================================

class ManipulationExecutorNode(Node):

    def __init__(self):
        super().__init__("manipulation_executor_node")

        self.cb_group = ReentrantCallbackGroup()

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("arm_name", DEFAULT_ARM)
        self.declare_parameter("use_hardcoded_pose", True)
        self.declare_parameter("sim_mode", True)
        self.declare_parameter("use_motion_planner", True)

        self.arm_name = self.get_parameter("arm_name").value
        self.use_hardcoded = self.get_parameter("use_hardcoded_pose").value
        self.sim_mode = self.get_parameter("sim_mode").value
        self.use_planner = self.get_parameter("use_motion_planner").value

        # ── CHANGE 5: Set duration based on mode ─────────────────────────
        self.move_duration = MOVE_DURATION if self.sim_mode else MOVE_DURATION_REAL
        # ── END CHANGE 5 ────────────────────────────────────────────────

        self.get_logger().info(
            f"ManipulationExecutor starting  |  arm={self.arm_name}  "
            f"sim={self.sim_mode}  hardcoded={self.use_hardcoded}  "
            f"planner={self.use_planner}  duration={self.move_duration}s"
        )

        # ─── State machine ──────────────────────────────────────────────
        self.state = "IDLE"
        self.state_start_time = None
        self.logged_state = None
        self.gripper_value = GRIPPER_OPEN
        self.grasp_pose = None
        self.current_arm_target = None   # the 6-element joint target we're tracking
        self.arm_cmd_sent = False        # have we sent the ArmCommand for this state?

        # ─── Joint state cache ──────────────────────────────────────────
        self.current_joint_state = None
        # ── CHANGE 6: Add name-based joint cache for real hardware ───────
        self.joint_positions_by_name = {}
        # ── END CHANGE 6 ────────────────────────────────────────────────

        # ================================================================
        #  SUBSCRIBERS
        # ================================================================

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        # ── CHANGE 7: Subscribe to BOTH grasp pose topics ───────────────
        # Legacy topic (for manual testing / backward compatibility)
        self.grasp_pose_sub = self.create_subscription(
            PoseStamped, "/planned_grasp", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )

        # Grasp planner node topic (from grasp_planner_node.py)
        self.grasp_planner_sub = self.create_subscription(
            PoseStamped, "/grasp_planner/grasp_pose", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )
        self.get_logger().info(
            "Listening on /planned_grasp AND /grasp_planner/grasp_pose"
        )
        # ── END CHANGE 7 ────────────────────────────────────────────────

        # ================================================================
        #  PUBLISHERS
        # ================================================================

        # Sim arm command
        self.arm_cmd_pub = self.create_publisher(
            ArmCommand, f"/{self.arm_name}_arm/cmd", 10,
        )

        # ── CHANGE 8: Add real hardware arm publisher ────────────────────
        if not self.sim_mode and HAS_INTERBOTIX:
            self.real_arm_cmd_pub = self.create_publisher(
                JointGroupCommand,
                f"/{self.arm_name}_arm/commands/joint_group", 10,
            )
            self.get_logger().info(
                f"Real mode: publishing arm commands to "
                f"/{self.arm_name}_arm/commands/joint_group"
            )
        elif not self.sim_mode and not HAS_INTERBOTIX:
            self.get_logger().error(
                "Real mode requested but interbotix_xs_msgs not available! "
                "Install interbotix packages."
            )
        # ── END CHANGE 8 ────────────────────────────────────────────────

        self.gripper_pub = self.create_publisher(
            Float64MultiArray, f"/{self.arm_name}_gripper/cmd", 10,
        )

        self.status_pub = self.create_publisher(
            String, "/manipulation/task_status", 10,
        )

        # ================================================================
        #  SERVICE CLIENT – motion planner
        # ================================================================

        self.plan_client = self.create_client(
          PlanToTarget,
          '/plan_to_target',
          callback_group=self.cb_group
        )

        # Wait for planner service (with timeout)
        if self.use_planner:
            self.get_logger().info('Waiting for /plan_to_target service...')
            service_available = self.plan_client.wait_for_service(timeout_sec=5.0)
            
            if not service_available:
                self.get_logger().warn(
                    'Motion planner service not available! '
                    'Falling back to hardcoded IK.'
                )
                self.use_planner = False
            else:
                self.get_logger().info('Motion planner service connected!')

        # ================================================================
        #  50 Hz CONTROL LOOP
        # ================================================================
        self.control_timer = self.create_timer(0.02, self._control_loop)

        # ================================================================
        #  STARTUP: hardcoded pose after 3 seconds
        # ================================================================
        if self.use_hardcoded:
            self.startup_timer = self.create_timer(
                3.0, self._publish_hardcoded_pose, callback_group=self.cb_group,
            )
            self.get_logger().info("Will publish hardcoded test pose in 3s …")
        else:
            self.get_logger().info(
                "Waiting for grasp pose from grasp_planner_node "
                "(topic: /grasp_planner/grasp_pose) …"
            )

        self.get_logger().info("ManipulationExecutor ready.")

    # =====================================================================
    #  JOINT STATE HELPERS
    # =====================================================================

    # ── CHANGE 9: Replace _get_arm_positions to support sim + real ───────
    def _get_arm_positions(self):
        """Return current 6-DOF arm joint positions, or None."""
        if self.sim_mode:
            # Sim: use fixed index positions
            if self.current_joint_state is None:
                return None
            return [self.current_joint_state.position[i] for i in RIGHT_ARM_INDICES]
        else:
            # Real: use named joint lookup
            positions = []
            for name in RIGHT_ARM_JOINT_NAMES:
                if name not in self.joint_positions_by_name:
                    return None
                positions.append(self.joint_positions_by_name[name])
            return positions
    # ── END CHANGE 9 ────────────────────────────────────────────────────

    # ── CHANGE 10: Replace _get_finger_positions to support sim + real ───
    def _get_finger_positions(self):
        """Return (left_finger, right_finger) positions, or None."""
        if self.current_joint_state is None:
            return None
        if self.sim_mode:
            # Sim: use fixed indices
            return (
                self.current_joint_state.position[RIGHT_FINGER_LEFT],
                self.current_joint_state.position[RIGHT_FINGER_RIGHT],
            )
        else:
            # Real: use named lookup
            left = self.joint_positions_by_name.get('right_left_finger')
            right = self.joint_positions_by_name.get('right_right_finger')
            if left is None or right is None:
                return None
            return (left, right)
    # ── END CHANGE 10 ───────────────────────────────────────────────────

    def _arm_at_target(self):
        """Check if arm joints are within tolerance of current_arm_target."""
        if self.current_arm_target is None:
            return False
        positions = self._get_arm_positions()
        if positions is None:
            return False
        error = np.linalg.norm(
            np.array(positions) - np.array(self.current_arm_target)
        )
        return error < ARM_ARRIVAL_TOLERANCE

    def _gripper_is_closed(self):
        """Check if finger joints confirm gripper is closed."""
        fingers = self._get_finger_positions()
        if fingers is None:
            return False
        left, right = fingers
        return left < GRIPPER_CLOSED_THRESHOLD and right < GRIPPER_CLOSED_THRESHOLD

    def _gripper_is_open(self):
        """Check if finger joints confirm gripper is open."""
        fingers = self._get_finger_positions()
        if fingers is None:
            return False
        left, right = fingers
        return left > GRIPPER_OPEN_THRESHOLD and right > GRIPPER_OPEN_THRESHOLD

    # =====================================================================
    #  50 Hz CONTROL LOOP
    # =====================================================================

    def _control_loop(self):
        """
        Runs at 50 Hz.  Every tick:
          1. Publish gripper command (keeps gripper locked at desired value)
          2. Check state + joint feedback → advance when conditions are met
        """
        now = time.time()

        # ── Always publish gripper at 50 Hz ─────────────────────────────
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(self.gripper_value)]
        self.gripper_pub.publish(gripper_msg)

        # ── If idle, nothing to do ──────────────────────────────────────
        if self.state == "IDLE":
            return

        # ── Initialize timing on first tick of a new state ──────────────
        if self.state_start_time is None:
            self.state_start_time = now

        elapsed = now - self.state_start_time

        # ── Log state transitions once ──────────────────────────────────
        if self.state != self.logged_state:
            self._log_state()
            self.logged_state = self.state

        # =================================================================
        #  STATE: OPEN_GRIPPER
        #  Condition to advance: gripper fingers confirm open
        # ── CHANGE 11: Add GRIPPER_MIN_WAIT before checking feedback ─────
        # =================================================================
        if self.state == "OPEN_GRIPPER":
            self.gripper_value = GRIPPER_OPEN
            if elapsed > GRIPPER_MIN_WAIT and self._gripper_is_open():
                self.get_logger().info("  Gripper confirmed open.")
                self._advance("MOVE_GRASP")
        # ── END CHANGE 11 ───────────────────────────────────────────────

        # =================================================================
        #  STATE: MOVE_PREGRASP
        #  Send arm command once, then wait for joints to arrive
        # =================================================================
        elif self.state == "MOVE_PREGRASP":
            self.gripper_value = GRIPPER_OPEN
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=PREGRASP_Z_OFFSET)
                self.arm_cmd_sent = True
            if self._arm_at_target():
                self.get_logger().info("  Arm arrived at pre-grasp.")
                self._advance("MOVE_GRASP")

        # =================================================================
        #  STATE: MOVE_GRASP
        #  Send arm command once, then wait for joints to arrive
        # =================================================================
        elif self.state == "MOVE_GRASP":
            self.gripper_value = GRIPPER_OPEN
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=0.0)
                self.arm_cmd_sent = True
            if self._arm_at_target():
                self.get_logger().info("  Arm arrived at grasp pose.")
                self._advance("PAUSE_AT_GRASP")

        # =================================================================
        #  STATE: PAUSE_AT_GRASP
        #  Gripper stays OPEN.  Wait a fixed time so the arm fully settles.
        # =================================================================
        elif self.state == "PAUSE_AT_GRASP":
            self.gripper_value = GRIPPER_OPEN
            if elapsed > PAUSE_AT_GRASP_SECS:
                self.get_logger().info(
                    f"  Paused {PAUSE_AT_GRASP_SECS}s at grasp pose. Closing gripper …"
                )
                self._advance("CLOSE_GRIPPER")

        # =================================================================
        #  STATE: CLOSE_GRIPPER
        #  Condition to advance: finger joints confirm closed
        # ── CHANGE 12: Add GRIPPER_MIN_WAIT before checking feedback ─────
        # =================================================================
        elif self.state == "CLOSE_GRIPPER":
            self.gripper_value = GRIPPER_CLOSE
            if elapsed > GRIPPER_MIN_WAIT and self._gripper_is_closed():
                fingers = self._get_finger_positions()
                self.get_logger().info(
                    f"  Gripper confirmed closed (fingers: {fingers})."
                )
                self._advance("MOVE_LIFT")
        # ── END CHANGE 12 ───────────────────────────────────────────────

        # =================================================================
        #  STATE: MOVE_LIFT
        #  Send arm command once, then wait for joints to arrive
        # =================================================================
        elif self.state == "MOVE_LIFT":
            self.gripper_value = GRIPPER_CLOSE
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=LIFT_HEIGHT)
                self.arm_cmd_sent = True
            if self._arm_at_target():
                self.get_logger().info("  Arm arrived at lift pose.")
                self._advance("DONE")

        # =================================================================
        #  STATE: DONE
        # =================================================================
        elif self.state == "DONE":
            # if not self.arm_cmd_sent:
            #     fingers = self._get_finger_positions()
            #     self.get_logger().info(f"  Final gripper state (fingers: {fingers}).")
            #     self._publish_status(REASON_SUCCESS)
            #     self.get_logger().info(f"Pick sequence complete – result: {REASON_SUCCESS}")
            #     self.arm_cmd_sent = True
            # Keep publishing arm command to prevent drift
            if self.current_arm_target is not None and self.sim_mode:
                cmd = ArmCommand()
                cmd.joint_positions = list(self.current_arm_target)
                cmd.duration = 0.02  # short duration = hold position
                self.arm_cmd_pub.publish(cmd)

            if not self.arm_cmd_sent:
                fingers = self._get_finger_positions()
                self.get_logger().info(f"  Final gripper state (fingers: {fingers}).")
                self._publish_status(REASON_SUCCESS)
                self.get_logger().info(f"Pick sequence complete – result: {REASON_SUCCESS}")
                self.arm_cmd_sent = True

    # =====================================================================
    #  STATE HELPERS
    # =====================================================================

    def _advance(self, new_state: str):
        """Transition to a new state, reset timing and command flag."""
        self.state = new_state
        self.state_start_time = None
        self.arm_cmd_sent = False

    def _log_state(self):
        """Log a message when entering a new state."""
        messages = {
            "OPEN_GRIPPER":   "Step 1/5 – Opening gripper (waiting for confirmation) …",
            "MOVE_PREGRASP":  "Step 2/5 – Moving to PRE-GRASP (waiting for arm arrival) …",
            "MOVE_GRASP":     "Step 3/5 – Descending to GRASP (waiting for arm arrival) …",
            "PAUSE_AT_GRASP": f"Step 4/5 – At grasp pose. Holding open for {PAUSE_AT_GRASP_SECS}s …",
            "CLOSE_GRIPPER":  "Step 4/5 – Closing gripper (waiting for confirmation) …",
            "MOVE_LIFT":      "Step 5/5 – LIFTING (waiting for arm arrival) …",
            "DONE":           "Done!",
        }
        self.get_logger().info(messages.get(self.state, f"State: {self.state}"))

    def _publish_status(self, reason: str):
        """Helper to publish task status."""
        status_msg = String()
        status_msg.data = reason
        self.status_pub.publish(status_msg)

    # =====================================================================
    #  ARM COMMAND
    # =====================================================================

    # ── CHANGE 13: Replace _send_arm_to_pose to support sim + real ───────
    def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
        """
        Send arm to target pose.
        - If planner is available: call /plan_to_target service (execute=True)
        - Otherwise: use hardcoded IK + direct arm command
        """
        target_pose = copy.deepcopy(grasp_pose)
        target_pose.position.z += z_offset

        if self.use_planner:
            # ── Planner path (sim or real) ──────────────────────────────
            joint_targets = self._call_planner_service(target_pose)

            if joint_targets is None:
                self.get_logger().error(
                    f"IK failed for pose at z={target_pose.position.z:.3f}. "
                    "Aborting grasp sequence."
                )
                self._publish_status(REASON_IK_FAIL)
                self._advance("DONE")
                return

            # Planner with execute=True already sent the arm command.
            # We just store the target for arrival tracking.
            self.current_arm_target = list(joint_targets)
            self.get_logger().info(
                f"  Planner executed: {[f'{j:.2f}' for j in joint_targets]}  "
                f"(z={target_pose.position.z:.3f})"
            )
        else:
            # ── Hardcoded IK fallback ───────────────────────────────────
            joint_targets = self._hardcoded_ik(target_pose)
            self.current_arm_target = list(joint_targets)

            if self.sim_mode:
                # Sim: use ArmCommand on /{arm}_arm/cmd
                cmd = ArmCommand()
                cmd.joint_positions = list(joint_targets)
                cmd.duration = self.move_duration
                self.arm_cmd_pub.publish(cmd)
            elif HAS_INTERBOTIX:
                # Real: use JointGroupCommand on /{arm}_arm/commands/joint_group
                cmd = JointGroupCommand()
                cmd.name = f'{self.arm_name}_arm'
                cmd.cmd = list(joint_targets)
                self.real_arm_cmd_pub.publish(cmd)
            else:
                self.get_logger().error("No arm command method available!")
                return

            self.get_logger().info(
                f"  ArmCommand: {[f'{j:.2f}' for j in joint_targets]}  "
                f"duration={self.move_duration}s  (z={target_pose.position.z:.3f})"
            )
    # ── END CHANGE 13 ───────────────────────────────────────────────────

    def _call_planner_service(self, target_pose: Pose):
        """
        Call the /plan_to_target service to compute IK.
        Uses execute=True so the planner also sends the arm command.

        Returns:
            List of 6 joint positions, or None if planning failed
        """
        state_label = {
            "MOVE_PREGRASP": "pre-grasp",
            "MOVE_GRASP": "grasp",
            "MOVE_LIFT": "lift"
        }.get(self.state, self.state)

        request = PlanToTarget.Request()
        request.arm_name = self.arm_name
        request.target_pose = target_pose
        request.use_orientation = False      # position-only for now
        request.execute = True               # let planner send the arm command
        request.duration = self.move_duration
        request.max_condition_number = 100.0

        self.get_logger().info(
            f"  Calling planner for {state_label} pose: "
            f"({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, "
            f"{target_pose.position.z:.3f})"
        )

        # Call service (blocking with timeout)
        future = self.plan_client.call_async(request)

        timeout = 10.0
        start_time = time.time()

        while not future.done():
            if time.time() - start_time > timeout:
                self.get_logger().error(
                    f"  Planner service call timed out after {timeout}s!"
                )
                return None
            time.sleep(0.01)

        # Get result
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"  Planner service exception: {e}")
            return None

        # Check success
        if not response.success:
            self.get_logger().error(f"  IK planning failed: {response.message}")
            return None

        # Log success
        self.get_logger().info(f"  IK success: {response.message}")
        if hasattr(response, 'position_error') and response.position_error is not None:
            self.get_logger().info(
                f"    Position error: {response.position_error:.4f}m, "
                f"Orientation error: {response.orientation_error:.4f}rad "
                f"({np.degrees(response.orientation_error):.1f}°)"
            )

        return list(response.joint_positions)

    def _hardcoded_ik(self, target_pose: Pose):
        """Fallback hardcoded IK (for testing without planner)."""
        z = target_pose.position.z

        if z > 0.20:
            joint_targets = [0.0, -0.5, 0.3, 0.0, -0.8, 0.0]
        elif z > 0.10:
            joint_targets = [0.0, 0.2, 0.3, 0.0, -0.5, 0.0]
        else:
            joint_targets = [0.0, 0.5, 0.5, 0.0, -0.2, 0.0]

        self.get_logger().warn(
            f"  Using HARDCODED fallback IK for z={z:.3f}"
        )

        return joint_targets

    # =====================================================================
    #  CALLBACKS
    # =====================================================================

    # ── CHANGE 14: Update _joint_state_cb to cache by name ───────────────
    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg
        # Also store by name (needed for real hardware)
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions_by_name[name] = msg.position[i]
    # ── END CHANGE 14 ───────────────────────────────────────────────────

    def _grasp_pose_cb(self, msg: PoseStamped):
        """
        ────────────────────────────────────────────────────────────────
        MAIN ENTRY POINT.  Receives grasp pose from:
          - /planned_grasp           (legacy / manual testing)
          - /grasp_planner/grasp_pose (from grasp_planner_node.py)
          - _publish_hardcoded_pose() (self-test)
        Kicks off the state machine.
        ────────────────────────────────────────────────────────────────
        """
        if self.state != "IDLE":
            self.get_logger().warn("Already executing – ignoring new pose.")
            return

        self.grasp_pose = msg.pose
        self.get_logger().info(
            f"Received grasp pose in '{msg.header.frame_id}':  "
            f"pos=({self.grasp_pose.position.x:.3f}, "
            f"{self.grasp_pose.position.y:.3f}, "
            f"{self.grasp_pose.position.z:.3f})  "
            f"orient=({self.grasp_pose.orientation.w:.3f}, "
            f"{self.grasp_pose.orientation.x:.3f}, "
            f"{self.grasp_pose.orientation.y:.3f}, "
            f"{self.grasp_pose.orientation.z:.3f})"
        )
        self._advance("OPEN_GRIPPER")

    # =====================================================================
    #  HARDCODED SIM/REAL POSE
    # =====================================================================

    # ── CHANGE 15: Separate sim vs real hardcoded poses ──────────────────
    def _publish_hardcoded_pose(self):
        """
        Publish a test pose (fires once after 3s).
        Sim and real use different target positions.
        """
        self.startup_timer.cancel()

        hardcoded = PoseStamped()
        hardcoded.header.stamp = self.get_clock().now().to_msg()
        hardcoded.header.frame_id = "base_link"

        if self.sim_mode:
            # ── Sim test pose ───────────────────────────────────────────
            # Uses hardcoded IK (z-based joint lookup)
            hardcoded.pose.position.x = 0.30
            hardcoded.pose.position.y = -0.15
            hardcoded.pose.position.z = 0.10

            hardcoded.pose.orientation.w = 1.0
            hardcoded.pose.orientation.x = 0.0
            hardcoded.pose.orientation.y = 0.0
            hardcoded.pose.orientation.z = 0.0
        else:
            # ── Real test pose ──────────────────────────────────────────
            # From test_planner_real.py – known reachable with planner
            qw, qx, qy, qz = ORIENT_FINGERS_DOWN
            hardcoded.pose.position.x = -0.10
            hardcoded.pose.position.y = -0.35
            hardcoded.pose.position.z = 0.55

            hardcoded.pose.orientation.w = qw
            hardcoded.pose.orientation.x = qx
            hardcoded.pose.orientation.y = qy
            hardcoded.pose.orientation.z = qz

        mode_str = "SIM" if self.sim_mode else "REAL"
        self.get_logger().info(
            f"Publishing {mode_str} hardcoded grasp pose for testing …"
        )
        self._grasp_pose_cb(hardcoded)
    # ── END CHANGE 15 ───────────────────────────────────────────────────


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ManipulationExecutor …")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
