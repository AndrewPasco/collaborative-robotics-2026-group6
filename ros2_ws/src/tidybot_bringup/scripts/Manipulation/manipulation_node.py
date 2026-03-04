#!/usr/bin/env python3
"""
manipulation_node.py

ME326 Collaborative Robotics – Group 6
Authors: LY, OP (Vision + Manipulation subteam)

Clean manipulation executor that works for BOTH sim and real hardware
using the same code path. Based on test_arms_real.py topic pattern.

ARCHITECTURE:
  50 Hz control loop publishes gripper every tick and checks state.
  State transitions use JOINT STATE FEEDBACK (not blind timers).
  Gripper close uses STALL DETECTION: when finger gap stops changing,
  the gripper has gripped an object — then holds for a firm grip period.

  States:
    IDLE            – waiting for a grasp pose
    OPEN_GRIPPER    – opening gripper before approach
    MOVE_PREGRASP   – arm moving to pre-grasp (above grasp pose)
    MOVE_GRASP      – arm moving to grasp position
    PAUSE_AT_GRASP  – settling at grasp pose (gripper still open)
    CLOSE_GRIPPER   – closing gripper on object (stall detection)
    MOVE_LIFT       – lifting object
    DONE            – sequence complete, publish status

TOPICS SUBSCRIBED:
  - /joint_states                      (sensor_msgs/JointState)       – feedback
  - /arm_status                        (std_msgs/Int32)               – from brain: 0=left, 1=right
  - /task_status                       (std_msgs/Int32)               – from brain: 0=task1, 1=task3
  - /detected_grasps/pose              (geometry_msgs/PoseStamped)    – grasp target (legacy)
  - /grasp_planner/grasp_pose          (geometry_msgs/PoseStamped)    – grasp target (from grasp_planner_node)

TOPICS PUBLISHED:
  - /{arm}_arm/joint_cmd               (std_msgs/Float64MultiArray)   – arm joint commands
  - /{arm}_gripper/cmd                 (std_msgs/Float64MultiArray)   – gripper commands
  - /manipulation/task_status          (std_msgs/String)              – result to brain

SERVICE CLIENTS:
  - /plan_to_target                    (tidybot_msgs/PlanToTarget)    – IK solver

GRIPPER CONVENTION (matches MuJoCo bridge and arm_wrapper_node):
  0.0 = fully OPEN
  1.0 = fully CLOSED

USAGE:
  # Sim (with hardcoded test pose):
  ros2 run tidybot_bringup manipulation_node.py --ros-args -p use_hardcoded_pose:=true

  # Sim (listening to grasp planner):
  ros2 run tidybot_bringup manipulation_node.py

  # Real robot (same command — no sim_mode flag needed):
  ros2 run tidybot_bringup manipulation_node.py --ros-args -p use_motion_planner:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import copy
from collections import deque

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray, Int32
from tidybot_msgs.srv import PlanToTarget


# =============================================================================
#  CONSTANTS
# =============================================================================

# Gripper values (0.0=open, 1.0=closed — matches MuJoCo bridge & arm_wrapper)
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0

# Vertical offsets (meters)
PREGRASP_Z_OFFSET = 0.08   # 8 cm above the grasp pose
LIFT_HEIGHT = 0.15          # 15 cm lift after grasping

# Joint state feedback thresholds
ARM_ARRIVAL_TOLERANCE = 0.15     # rad – how close joints must be to target
GRIPPER_OPEN_THRESHOLD = 0.03    # finger position above this = open enough
GRIPPER_CLOSED_THRESHOLD = 0.018 # finger position below this = fully closed
PAUSE_AT_GRASP_SECS = 1.0       # seconds to hold still at grasp before closing

# Gripper stall detection (for gripping objects that prevent full closure)
STALL_CHECK_DELAY = 3.0      # seconds – don't check for stalls until grippers have had time to move
STALL_THRESHOLD = 0.001      # meters – if finger gap changes less than this, stalled
STALL_WINDOW = 2.0           # seconds – compare current gap to gap this long ago
FIRM_GRIP_HOLD = 5.0         # seconds – keep closing after stall detected
GRIPPER_SAFETY_TIMEOUT = 20.0  # seconds – max wait if stall detection never triggers

# Fingers-down orientation (qw, qx, qy, qz)
ORIENT_FINGERS_DOWN = (0.5, 0.5, 0.5, -0.5)

# Reason codes for task_status publisher
REASON_SUCCESS = "SUCCESS"
REASON_IK_FAIL = "IK_FAIL"
REASON_TIMEOUT = "TIMEOUT"

# ── Joint state indices (from MuJoCo bridge JOINT_NAMES order) ───────────
# [0] camera_pan  [1] camera_tilt
# [2] right_waist [3] right_shoulder [4] right_elbow
# [5] right_forearm_roll [6] right_wrist_angle [7] right_wrist_rotate
# [8] right_left_finger  [9] right_right_finger
# [10] left_waist [11] left_shoulder [12] left_elbow
# [13] left_forearm_roll [14] left_wrist_angle [15] left_wrist_rotate
# [16] left_left_finger  [17] left_right_finger

ARM_INDICES = {
    "right": [2, 3, 4, 5, 6, 7],
    "left":  [10, 11, 12, 13, 14, 15],
}

FINGER_INDICES = {
    "right": (8, 9),    # (left_finger, right_finger)
    "left":  (16, 17),
}

ARM_JOINT_NAMES = {
    "right": [
        'right_waist', 'right_shoulder', 'right_elbow',
        'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate',
    ],
    "left": [
        'left_waist', 'left_shoulder', 'left_elbow',
        'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate',
    ],
}


# =============================================================================
#  NODE
# =============================================================================

class ManipulationExecutor(Node):
    """Grasp-and-lift executor. Works for both sim and real hardware."""

    def __init__(self):
        super().__init__("manipulation_executor")

        self.cb_group = ReentrantCallbackGroup()

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("arm_name", "right")
        self.declare_parameter("use_hardcoded_pose", False)
        self.declare_parameter("use_motion_planner", True)

        self.arm_name = self.get_parameter("arm_name").value
        self.use_hardcoded = self.get_parameter("use_hardcoded_pose").value
        self.use_planner = self.get_parameter("use_motion_planner").value

        self.get_logger().info(
            f"ManipulationExecutor starting  |  arm={self.arm_name}  "
            f"hardcoded={self.use_hardcoded}  planner={self.use_planner}"
        )

        # ─── State machine ──────────────────────────────────────────────
        self.state = "IDLE"
        self.state_start_time = None
        self.logged_state = None
        self.grasp_pose = None          # PoseStamped from grasp planner
        self.current_arm_target = None  # 6-element joint target we're tracking
        self.arm_cmd_sent = False       # have we published joints for this state?

        # Per-arm gripper values (both arms publish every tick)
        self.gripper_values = {"right": GRIPPER_OPEN, "left": GRIPPER_OPEN}

        # Stall detection state (for CLOSE_GRIPPER)
        self.grip_gap_history = deque()  # list of (timestamp, finger_gap)
        self.stall_detected_time = None  # when stall was first detected

        # Task info from brain node
        self.current_arm_status = 1     # default: right arm (0=left, 1=right)
        self.current_task = 0           # default: task1 (0=task1, 1=task3)

        # ─── Joint state cache ──────────────────────────────────────────
        self.current_joint_state = None
        self.joint_positions_by_name = {}

        # ================================================================
        #  SUBSCRIBERS
        # ================================================================

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        # From brain node (Mete: use std_msgs/Int32 for both of these)
        self.arm_status_sub = self.create_subscription(
            Int32, "/arm_status", self._arm_status_cb, 10,
            callback_group=self.cb_group,
        )
        self.task_status_sub = self.create_subscription(
            Int32, "/task_status", self._task_status_cb, 10,
            callback_group=self.cb_group,
        )

        # Grasp pose topics (both point to the same callback)
        self.create_subscription(
            PoseStamped, "/detected_grasps/pose", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            PoseStamped, "/grasp_planner/grasp_pose", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )
        self.get_logger().info(
            "Listening on /detected_grasps/pose AND /grasp_planner/grasp_pose"
        )

        # ================================================================
        #  PUBLISHERS (both arms — Float64MultiArray on joint_cmd/gripper)
        # ================================================================

        self.arm_pubs = {}
        self.gripper_pubs = {}
        for arm in ("right", "left"):
            self.arm_pubs[arm] = self.create_publisher(
                Float64MultiArray, f"/{arm}_arm/joint_cmd", 10,
            )
            self.gripper_pubs[arm] = self.create_publisher(
                Float64MultiArray, f"/{arm}_gripper/cmd", 10,
            )

        self.status_pub = self.create_publisher(
            String, "/manipulation/task_status", 10,
        )

        # ================================================================
        #  SERVICE CLIENT — motion planner
        # ================================================================

        self.plan_client = self.create_client(
            PlanToTarget, '/plan_to_target',
            callback_group=self.cb_group,
        )

        if self.use_planner:
            self.get_logger().info("Waiting for /plan_to_target service...")
            if self.plan_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("Motion planner service connected!")
            else:
                self.get_logger().warn(
                    "Motion planner not available! Falling back to hardcoded IK."
                )
                self.use_planner = False

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
            self.get_logger().info("Will publish hardcoded test pose in 3s...")
        else:
            self.get_logger().info(
                "Waiting for grasp pose on /grasp_planner/grasp_pose..."
            )

        self.get_logger().info("ManipulationExecutor ready.")

    # =====================================================================
    #  JOINT STATE HELPERS
    # =====================================================================

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions_by_name[name] = msg.position[i]

    def _get_arm_positions(self, arm: str):
        """Return current 6-DOF joint positions for the given arm, or None."""
        # Try index-based lookup first (works when /joint_states has all joints)
        js = self.current_joint_state
        if js is not None and len(js.position) > max(ARM_INDICES[arm]):
            return [js.position[i] for i in ARM_INDICES[arm]]
        # Fallback: name-based lookup (works with per-arm joint_states)
        names = ARM_JOINT_NAMES[arm]
        if all(n in self.joint_positions_by_name for n in names):
            return [self.joint_positions_by_name[n] for n in names]
        return None

    def _get_finger_positions(self, arm: str):
        """Return (left_finger, right_finger) positions, or None."""
        js = self.current_joint_state
        lf, rf = FINGER_INDICES[arm]
        if js is not None and len(js.position) > max(lf, rf):
            return js.position[lf], js.position[rf]
        return None

    def _arm_at_target(self, arm: str, target, tol=ARM_ARRIVAL_TOLERANCE):
        """Check if arm joints are within tolerance of target."""
        pos = self._get_arm_positions(arm)
        if pos is None or target is None:
            return False
        self.get_logger().info(f"arm positions: {pos}")
        self.get_logger().info(f"arm target: {target}")
        return all(abs(p - t) < tol for p, t in zip(pos, target))

    def _gripper_is_closed(self, arm: str):
        fingers = self._get_finger_positions(arm)
        if fingers is None:
            return False
        return max(fingers) < GRIPPER_CLOSED_THRESHOLD

    def _gripper_is_open(self, arm: str):
        fingers = self._get_finger_positions(arm)
        if fingers is None:
            return False
        return min(fingers) > GRIPPER_OPEN_THRESHOLD

    def _get_finger_gap(self, arm: str):
        """Return sum of both finger positions (proxy for gripper aperture)."""
        fingers = self._get_finger_positions(arm)
        if fingers is None:
            return None
        return fingers[0] + fingers[1]

    # =====================================================================
    #  50 Hz CONTROL LOOP
    # =====================================================================

    def _control_loop(self):
        """
        Runs at 50 Hz. Every tick:
          1. Publish gripper commands for BOTH arms (keeps grip on inactive arm)
          2. Check state + joint feedback → advance when conditions are met
        """
        import time
        now = time.time()

        # Always publish gripper for both arms at 50 Hz
        for arm in ("right", "left"):
            msg = Float64MultiArray()
            msg.data = [float(self.gripper_values[arm])]
            self.gripper_pubs[arm].publish(msg)

        if self.state == "IDLE":
            return

        # Initialize timing on first tick of a new state
        if self.state_start_time is None:
            self.state_start_time = now
        elapsed = now - self.state_start_time

        # Log state transitions once
        if self.state != self.logged_state:
            self._log_state()
            self.logged_state = self.state

        arm = self.arm_name

        # ── OPEN_GRIPPER ──────────────────────────────────────────────
        if self.state == "OPEN_GRIPPER":
            self.gripper_values[arm] = GRIPPER_OPEN
            if self._gripper_is_open(arm) or elapsed > 3.0:
                self.get_logger().info("  Gripper confirmed open.")
                self._advance("MOVE_PREGRASP")

        # ── MOVE_PREGRASP ─────────────────────────────────────────────
        elif self.state == "MOVE_PREGRASP":
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=PREGRASP_Z_OFFSET)
                self.arm_cmd_sent = True
            if self._arm_at_target(arm, self.current_arm_target):
                self.get_logger().info("  Arm arrived at pre-grasp.")
                self._advance("MOVE_GRASP")

        # ── MOVE_GRASP ────────────────────────────────────────────────
        elif self.state == "MOVE_GRASP":
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=0.015)
                self.arm_cmd_sent = True
            if self._arm_at_target(arm, self.current_arm_target):
                self.get_logger().info("  Arm arrived at grasp pose.")
                self._advance("PAUSE_AT_GRASP")

        # ── PAUSE_AT_GRASP ────────────────────────────────────────────
        elif self.state == "PAUSE_AT_GRASP":
            if elapsed > PAUSE_AT_GRASP_SECS:
                self.get_logger().info(
                    f"  Paused {PAUSE_AT_GRASP_SECS}s at grasp. Closing gripper..."
                )
                self._advance("CLOSE_GRIPPER")

        # ── CLOSE_GRIPPER (with stall detection) ─────────────────────
        elif self.state == "CLOSE_GRIPPER":
            self.gripper_values[arm] = GRIPPER_CLOSE

            # If gripper fully closed (empty grasp), advance immediately
            if self._gripper_is_closed(arm):
                self.get_logger().info("  Gripper fully closed.")
                self._advance("MOVE_LIFT")
                return

            # Record finger gap for stall detection
            gap = self._get_finger_gap(arm)
            if gap is not None:
                self.grip_gap_history.append((now, gap))
                # Trim entries older than STALL_WINDOW
                while self.grip_gap_history and (now - self.grip_gap_history[0][0]) > STALL_WINDOW:
                    self.grip_gap_history.popleft()

                # Check for stall (only after delay so grippers have time to start moving)
                if len(self.grip_gap_history) >= 2 and elapsed > STALL_CHECK_DELAY:
                    oldest_gap = self.grip_gap_history[0][1]
                    if abs(gap - oldest_gap) < STALL_THRESHOLD:
                        # Gripper has stalled — gripping an object
                        if self.stall_detected_time is None:
                            self.stall_detected_time = now
                            self.get_logger().info(
                                f"  Gripper stall detected (gap={gap:.4f}). "
                                f"Holding for {FIRM_GRIP_HOLD}s..."
                            )
                        elif (now - self.stall_detected_time) > FIRM_GRIP_HOLD:
                            fingers = self._get_finger_positions(arm)
                            self.get_logger().info(
                                f"  Firm grip hold complete (fingers: {fingers})."
                            )
                            self._advance("MOVE_LIFT")
                            return

            # Safety timeout
            if elapsed > GRIPPER_SAFETY_TIMEOUT:
                fingers = self._get_finger_positions(arm)
                self.get_logger().warn(
                    f"  Gripper safety timeout ({GRIPPER_SAFETY_TIMEOUT}s). "
                    f"Proceeding to lift (fingers: {fingers})."
                )
                self._advance("MOVE_LIFT")

        # ── MOVE_LIFT ─────────────────────────────────────────────────
        elif self.state == "MOVE_LIFT":
            self.gripper_values[arm] = GRIPPER_CLOSE
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=LIFT_HEIGHT)
                self.arm_cmd_sent = True
            if self._arm_at_target(arm, self.current_arm_target):
                self.get_logger().info("  Arm arrived at lift pose.")
                self._advance("DONE")

        # ── DONE ──────────────────────────────────────────────────────
        elif self.state == "DONE":
            self.gripper_values[arm] = GRIPPER_CLOSE
            if not self.arm_cmd_sent:
                fingers = self._get_finger_positions(arm)
                self.get_logger().info(f"  Final gripper state (fingers: {fingers}).")
                self._publish_status(REASON_SUCCESS)
                self.get_logger().info(f"Pick sequence complete — result: {REASON_SUCCESS}")
                self.arm_cmd_sent = True

    # =====================================================================
    #  STATE HELPERS
    # =====================================================================

    def _advance(self, new_state: str):
        """Transition to a new state, reset timing and command flag."""
        self.state = new_state
        self.state_start_time = None
        self.arm_cmd_sent = False
        # Reset stall detection
        self.grip_gap_history.clear()
        self.stall_detected_time = None

    def _log_state(self):
        messages = {
            "OPEN_GRIPPER":   f"Step 1/7 – Opening gripper ({self.arm_name} arm)...",
            "MOVE_PREGRASP":  "Step 2/7 – Moving to PRE-GRASP...",
            "MOVE_GRASP":     "Step 3/7 – Descending to GRASP...",
            "PAUSE_AT_GRASP": f"Step 4/7 – Holding at grasp for {PAUSE_AT_GRASP_SECS}s...",
            "CLOSE_GRIPPER":  "Step 5/7 – Closing gripper (stall detection)...",
            "MOVE_LIFT":      "Step 6/7 – LIFTING...",
            "DONE":           "Step 7/7 – Done!",
        }
        self.get_logger().info(messages.get(self.state, f"State: {self.state}"))

    def _publish_status(self, reason: str):
        msg = String()
        msg.data = reason
        self.status_pub.publish(msg)

    # =====================================================================
    #  ARM COMMAND
    # =====================================================================

    def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
        """
        Send active arm to a target pose (with optional z offset).
        1. If planner available: call /plan_to_target with execute=False (IK only)
        2. Otherwise: use hardcoded IK fallback
        Then publish joint positions as Float64MultiArray to /{arm}_arm/joint_cmd.
        """
        target_pose = copy.deepcopy(grasp_pose)
        target_pose.position.z += z_offset

        if self.use_planner:
            joint_targets = self._call_planner(target_pose)
            if joint_targets is None:
                self.get_logger().error(
                    f"IK failed for z={target_pose.position.z:.3f}. Aborting."
                )
                self._publish_status(REASON_IK_FAIL)
                self._advance("DONE")
                return
        else:
            joint_targets = self._hardcoded_ik(target_pose)

        # Store target for arrival tracking
        self.current_arm_target = list(joint_targets)

        # Publish directly to /{arm}_arm/joint_cmd (works for sim and real)
        msg = Float64MultiArray()
        msg.data = list(joint_targets)
        ## self.arm_pubs[self.arm_name].publish(msg)

        self.get_logger().info(
            f"  Sent joints to {self.arm_name} arm: "
            f"{[f'{j:.2f}' for j in joint_targets]}  "
            f"(z={target_pose.position.z:.3f})"
        )

    def _call_planner(self, target_pose: Pose):
        """
        Call /plan_to_target with execute=False (IK only).
        Returns list of 6 joint positions, or None on failure.
        """
        import time

        state_label = {
            "MOVE_PREGRASP": "pre-grasp",
            "MOVE_GRASP": "grasp",
            "MOVE_LIFT": "lift",
        }.get(self.state, self.state)

        request = PlanToTarget.Request()
        request.arm_name = self.arm_name
        request.target_pose = target_pose
        request.use_orientation = True
        request.execute = True  # IK only — we publish joint_cmd ourselves
        request.duration = 5.0
        request.max_condition_number = 100.0

        self.get_logger().info(
            f"  Calling planner for {state_label}: "
            f"({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, "
            f"{target_pose.position.z:.3f})"
        )

        future = self.plan_client.call_async(request)

        timeout = 10.0
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                self.get_logger().error(f"  Planner timed out after {timeout}s!")
                return None
            time.sleep(0.01)

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"  Planner exception: {e}")
            return None

        if not response.success:
            self.get_logger().error(f"  IK failed: {response.message}")
            return None

        self.get_logger().info(
            f"  IK success — pos_err={response.position_error:.4f}m, "
            f"orient_err={np.degrees(response.orientation_error):.1f}°"
        )
        return list(response.joint_positions)

    def _hardcoded_ik(self, target_pose: Pose):
        """Fallback IK for testing without planner. Arm-aware waist angle."""
        z = target_pose.position.z
        # Right arm waist points left (-π/2), left arm waist points right (+π/2)
        waist = -1.5708 if self.arm_name == "right" else 1.5708

        if z > 0.20:
            joints = [waist, -0.5, 0.3, 0.0, -0.8, 0.0]
        elif z > 0.10:
            joints = [waist, 0.2, 0.3, 0.0, -0.5, 0.0]
        else:
            joints = [waist, 0.5, 0.5, 0.0, -0.2, 0.0]

        self.get_logger().warn(f"  Using HARDCODED IK for z={z:.3f} ({self.arm_name})")
        return joints

    # =====================================================================
    #  CALLBACKS
    # =====================================================================

    def _arm_status_cb(self, msg: Int32):
        """From brain node: 0=left, 1=right. Only switches arm when IDLE or DONE."""
        if self.state not in ("IDLE", "DONE"):
            return
        old = self.arm_name
        self.arm_name = "left" if msg.data == 0 else "right"
        if self.arm_name != old:
            self.get_logger().info(f"Arm switched: {old} → {self.arm_name}")

    def _task_status_cb(self, msg: Int32):
        """From brain node: 0=task1, 1=task3."""
        self.current_task = msg.data
        self.get_logger().info(
            f"Task set to: {'task3 (bottle+cap)' if msg.data == 1 else 'task1 (single arm)'}"
        )

    def _grasp_pose_cb(self, msg: PoseStamped):
        """
        Receives grasp pose and kicks off the state machine.
        Accepts new poses when IDLE or DONE (so task3 can run a second arm).
        """
        if self.state not in ("IDLE", "DONE"):
            self.get_logger().warn("Already executing — ignoring new pose.")
            return

        self.grasp_pose = msg.pose
        self.get_logger().info(
            f"Grasp pose received ({self.arm_name} arm):  "
            f"pos=({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, "
            f"{msg.pose.position.z:.3f})  "
            f"orient=({msg.pose.orientation.w:.3f}, "
            f"{msg.pose.orientation.x:.3f}, "
            f"{msg.pose.orientation.y:.3f}, "
            f"{msg.pose.orientation.z:.3f})"
        )
        self._advance("OPEN_GRIPPER")

    def _publish_hardcoded_pose(self):
        """Publish a test pose (fires once after 3s startup delay)."""
        self.startup_timer.cancel()

        hardcoded = PoseStamped()
        hardcoded.header.stamp = self.get_clock().now().to_msg()
        hardcoded.header.frame_id = "base_link"

        # Default test pose (reachable by both arms with planner)
        qw, qx, qy, qz = ORIENT_FINGERS_DOWN
        hardcoded.pose.position.x = -0.10
        hardcoded.pose.position.y = -0.35
        hardcoded.pose.position.z = 0.55
        hardcoded.pose.orientation.w = qw
        hardcoded.pose.orientation.x = qx
        hardcoded.pose.orientation.y = qy
        hardcoded.pose.orientation.z = qz

        self.get_logger().info("Publishing hardcoded test pose...")
        self._grasp_pose_cb(hardcoded)


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationExecutor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
