#!/usr/bin/env python3
"""
manipulation_executor_node.py

ME326 Collaborative Robotics – Group 6
Authors: LY, OP (Vision + Manipulation subteam)

STATE MACHINE pick executor driven by a 50 Hz timer.
Gripper published every tick. State transitions use joint feedback.

Supports two modes:
  - SIM MODE:  hardcoded IK + ArmCommand on /right_arm/cmd
  - REAL MODE: PlanToTarget service (execute=True) + JointGroupCommand fallback

States:
  IDLE → OPEN_GRIPPER → MOVE_GRASP → PAUSE_AT_GRASP → CLOSE_GRIPPER → MOVE_LIFT → DONE

COORDINATE FRAME (base_link):
  -Y is forward, +X is left, +Z is up.
  Right arm shoulder at (-0.15, -0.12, 0.45).

GRIPPER: -1.0 = open, 1.0 = closed

Usage (sim):
  python3 manipulation_executor_node.py

Usage (real):
  python3 manipulation_executor_node.py --ros-args -p sim_mode:=false -p use_motion_planner:=true
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

# Real hardware arm command (only used in real mode)
try:
    from interbotix_xs_msgs.msg import JointGroupCommand
    HAS_INTERBOTIX = True
except ImportError:
    HAS_INTERBOTIX = False


# =============================================================================
#  CONSTANTS
# =============================================================================

DEFAULT_ARM = "right"

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

PREGRASP_Z_OFFSET = 0.08
LIFT_HEIGHT = 0.15

# Timing
MOVE_DURATION_SIM = 2.0
MOVE_DURATION_REAL = 3.0   # slower for real hardware safety

# Joint feedback thresholds
ARM_ARRIVAL_TOLERANCE = 0.15
GRIPPER_OPEN_THRESHOLD = 0.03
GRIPPER_CLOSED_THRESHOLD = 0.01
PAUSE_AT_GRASP_SECS = 2.0
GRIPPER_MIN_WAIT = 2.0     # min seconds before checking gripper feedback

# Safety timeouts
ARM_MOVE_TIMEOUT = 10.0
GRIPPER_TIMEOUT = 5.0

# Joint state indices (sim)
RIGHT_ARM_INDICES = [2, 3, 4, 5, 6, 7]
RIGHT_FINGER_LEFT = 8
RIGHT_FINGER_RIGHT = 9

# Real hardware joint names
RIGHT_ARM_JOINT_NAMES = [
    'right_waist', 'right_shoulder', 'right_elbow',
    'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate'
]

REASON_SUCCESS = "SUCCESS"
REASON_IK_FAIL = "IK_FAIL"

# Fingers-down orientation from test_planner_sim.py
ORIENT_FINGERS_DOWN = (0.5, 0.5, 0.5, -0.5)  # (qw, qx, qy, qz)


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

        self.move_duration = MOVE_DURATION_SIM if self.sim_mode else MOVE_DURATION_REAL

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
        self.current_arm_target = None
        self.arm_cmd_sent = False

        # ─── Joint state cache ──────────────────────────────────────────
        self.current_joint_state = None
        self.joint_positions_by_name = {}  # for real hardware

        # ================================================================
        #  SUBSCRIBERS
        # ================================================================

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        self.grasp_pose_sub = self.create_subscription(
            PoseStamped, "/planned_grasp", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )

        # ================================================================
        #  PUBLISHERS
        # ================================================================

        # Sim arm command
        self.arm_cmd_pub = self.create_publisher(
            ArmCommand, f"/{self.arm_name}_arm/cmd", 10,
        )

        # Real hardware arm command
        if HAS_INTERBOTIX:
            self.real_arm_cmd_pub = self.create_publisher(
                JointGroupCommand,
                f"/{self.arm_name}_arm/commands/joint_group", 10,
            )

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
            PlanToTarget, '/plan_to_target',
            callback_group=self.cb_group,
        )

        if self.use_planner:
            self.get_logger().info('Waiting for /plan_to_target service...')
            if not self.plan_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn(
                    'Motion planner not available! Falling back to hardcoded IK.'
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

        self.get_logger().info("ManipulationExecutor ready – waiting for grasp poses …")

    # =====================================================================
    #  JOINT STATE HELPERS
    # =====================================================================

    def _get_arm_positions(self):
        """Return current 6-DOF arm joint positions."""
        if self.sim_mode:
            if self.current_joint_state is None:
                return None
            return [self.current_joint_state.position[i] for i in RIGHT_ARM_INDICES]
        else:
            # Real hardware: use named joints
            positions = []
            for name in RIGHT_ARM_JOINT_NAMES:
                if name not in self.joint_positions_by_name:
                    return None
                positions.append(self.joint_positions_by_name[name])
            return positions

    def _get_finger_positions(self):
        if self.current_joint_state is None:
            return None
        if self.sim_mode:
            return (
                self.current_joint_state.position[RIGHT_FINGER_LEFT],
                self.current_joint_state.position[RIGHT_FINGER_RIGHT],
            )
        else:
            # Real hardware: look up by name
            left = self.joint_positions_by_name.get('right_left_finger')
            right = self.joint_positions_by_name.get('right_right_finger')
            if left is None or right is None:
                return None
            return (left, right)

    def _arm_at_target(self):
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
        fingers = self._get_finger_positions()
        if fingers is None:
            return False
        return fingers[0] < GRIPPER_CLOSED_THRESHOLD and fingers[1] < GRIPPER_CLOSED_THRESHOLD

    def _gripper_is_open(self):
        fingers = self._get_finger_positions()
        if fingers is None:
            return False
        return fingers[0] > GRIPPER_OPEN_THRESHOLD and fingers[1] > GRIPPER_OPEN_THRESHOLD

    # =====================================================================
    #  50 Hz CONTROL LOOP
    # =====================================================================

    def _control_loop(self):
        now = time.time()

        # Always publish gripper at 50 Hz
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(self.gripper_value)]
        self.gripper_pub.publish(gripper_msg)

        if self.state == "IDLE":
            return

        if self.state_start_time is None:
            self.state_start_time = now
        elapsed = now - self.state_start_time

        if self.state != self.logged_state:
            self._log_state()
            self.logged_state = self.state

        # ── OPEN_GRIPPER ────────────────────────────────────────────────
        if self.state == "OPEN_GRIPPER":
            self.gripper_value = GRIPPER_OPEN
            if elapsed > GRIPPER_MIN_WAIT and (self._gripper_is_open() or elapsed > GRIPPER_TIMEOUT):
                fingers = self._get_finger_positions()
                self.get_logger().info(f"  Gripper open (fingers: {fingers}).")
                self._advance("MOVE_GRASP")

        # ── MOVE_GRASP ─────────────────────────────────────────────────
        elif self.state == "MOVE_GRASP":
            self.gripper_value = GRIPPER_OPEN
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=0.0)
                self.arm_cmd_sent = True
            # Log error every 0.5s for debugging
            if int(elapsed * 2) != int((elapsed - 0.02) * 2):
                pos = self._get_arm_positions()
                if pos and self.current_arm_target:
                    error = np.linalg.norm(
                        np.array(pos) - np.array(self.current_arm_target)
                    )
                    self.get_logger().info(f"  Arm error: {error:.3f} rad")
            if self._arm_at_target():
                self.get_logger().info("  Arm arrived at grasp pose.")
                self._advance("PAUSE_AT_GRASP")
            elif elapsed > ARM_MOVE_TIMEOUT:
                self.get_logger().warn("  Grasp move timed out – proceeding.")
                self._advance("PAUSE_AT_GRASP")

        # ── PAUSE_AT_GRASP ─────────────────────────────────────────────
        elif self.state == "PAUSE_AT_GRASP":
            self.gripper_value = GRIPPER_OPEN
            if elapsed > PAUSE_AT_GRASP_SECS:
                self.get_logger().info(f"  Paused {PAUSE_AT_GRASP_SECS}s. Closing gripper …")
                self._advance("CLOSE_GRIPPER")

        # ── CLOSE_GRIPPER ──────────────────────────────────────────────
        elif self.state == "CLOSE_GRIPPER":
            self.gripper_value = GRIPPER_CLOSE
            if elapsed > GRIPPER_MIN_WAIT and (self._gripper_is_closed() or elapsed > GRIPPER_TIMEOUT):
                fingers = self._get_finger_positions()
                if self._gripper_is_closed():
                    self.get_logger().info(f"  Gripper closed (fingers: {fingers}).")
                else:
                    self.get_logger().warn(f"  Gripper close timed out (fingers: {fingers}).")
                self._advance("MOVE_LIFT")

        # ── MOVE_LIFT ──────────────────────────────────────────────────
        elif self.state == "MOVE_LIFT":
            self.gripper_value = GRIPPER_CLOSE
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=LIFT_HEIGHT)
                self.arm_cmd_sent = True
            if self._arm_at_target():
                self.get_logger().info("  Arm arrived at lift pose.")
                self._advance("DONE")
            elif elapsed > ARM_MOVE_TIMEOUT:
                self.get_logger().warn("  Lift timed out.")
                self._advance("DONE")

        # ── DONE ───────────────────────────────────────────────────────
        elif self.state == "DONE":
            self.gripper_value = GRIPPER_CLOSE
            if not self.arm_cmd_sent:
                fingers = self._get_finger_positions()
                self.get_logger().info(f"  Final gripper (fingers: {fingers}).")
                self._publish_status(REASON_SUCCESS)
                self.get_logger().info(f"Pick sequence complete – result: {REASON_SUCCESS}")
                self.arm_cmd_sent = True

    # =====================================================================
    #  STATE HELPERS
    # =====================================================================

    def _advance(self, new_state: str):
        self.state = new_state
        self.state_start_time = None
        self.arm_cmd_sent = False

    def _log_state(self):
        messages = {
            "OPEN_GRIPPER":   "Step 1 – Opening gripper …",
            "MOVE_GRASP":     "Step 2 – Moving to GRASP pose …",
            "PAUSE_AT_GRASP": f"Step 3 – At grasp pose. Holding open {PAUSE_AT_GRASP_SECS}s …",
            "CLOSE_GRIPPER":  "Step 4 – Closing gripper …",
            "MOVE_LIFT":      "Step 5 – LIFTING …",
            "DONE":           "Done!",
        }
        self.get_logger().info(messages.get(self.state, f"State: {self.state}"))

    def _publish_status(self, reason: str):
        status_msg = String()
        status_msg.data = reason
        self.status_pub.publish(status_msg)

    # =====================================================================
    #  ARM COMMAND
    # =====================================================================

    def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
        """Send arm to target pose via planner or hardcoded IK."""
        target_pose = copy.deepcopy(grasp_pose)
        target_pose.position.z += z_offset

        if self.use_planner:
            joint_targets = self._call_planner_service(target_pose)
            if joint_targets is None:
                self.get_logger().error(
                    f"IK failed for z={target_pose.position.z:.3f}. Aborting."
                )
                self._publish_status(REASON_IK_FAIL)
                self._advance("DONE")
                return
            # Planner with execute=True already sends the command.
            # We just store the target for arrival tracking.
            self.current_arm_target = list(joint_targets)
            self.get_logger().info(
                f"  Planner sent arm to joints: "
                f"{[f'{j:.2f}' for j in joint_targets]}  (z={target_pose.position.z:.3f})"
            )
        else:
            # Hardcoded IK fallback (sim only)
            joint_targets = self._hardcoded_ik(target_pose)
            self.current_arm_target = list(joint_targets)

            if self.sim_mode:
                cmd = ArmCommand()
                cmd.joint_positions = list(joint_targets)
                cmd.duration = self.move_duration
                self.arm_cmd_pub.publish(cmd)
            elif HAS_INTERBOTIX:
                cmd = JointGroupCommand()
                cmd.name = f'{self.arm_name}_arm'
                cmd.cmd = list(joint_targets)
                self.real_arm_cmd_pub.publish(cmd)

            self.get_logger().info(
                f"  ArmCommand: {[f'{j:.2f}' for j in joint_targets]}  "
                f"duration={self.move_duration}s  (z={target_pose.position.z:.3f})"
            )

    def _call_planner_service(self, target_pose: Pose):
        """Call /plan_to_target with execute=True. Returns joint positions or None."""
        state_label = {
            "MOVE_GRASP": "grasp", "MOVE_LIFT": "lift"
        }.get(self.state, self.state)

        request = PlanToTarget.Request()
        request.arm_name = self.arm_name
        request.target_pose = target_pose
        request.use_orientation = False     # position-only for now
        request.execute = True              # let planner send the arm command
        request.duration = self.move_duration
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
                self.get_logger().error("  Planner timed out!")
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

        self.get_logger().info(f"  IK success: {response.message}")
        if hasattr(response, 'position_error'):
            self.get_logger().info(
                f"    pos_err={response.position_error:.4f}m  "
                f"ori_err={np.degrees(response.orientation_error):.1f}deg"
            )

        return list(response.joint_positions)

    def _hardcoded_ik(self, target_pose: Pose):
        """Fallback hardcoded IK for sim testing without planner."""
        z = target_pose.position.z
        if z > 0.20:
            joint_targets = [0.0, -0.5, 0.3, 0.0, -0.8, 0.0]
        elif z > 0.10:
            joint_targets = [0.0, 0.2, 0.3, 0.0, -0.5, 0.0]
        else:
            joint_targets = [0.0, 0.5, 0.5, 0.0, -0.2, 0.0]
        self.get_logger().warn(f"  HARDCODED fallback IK for z={z:.3f}")
        return joint_targets

    # =====================================================================
    #  CALLBACKS
    # =====================================================================

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg
        # Also store by name for real hardware
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions_by_name[name] = msg.position[i]

    def _grasp_pose_cb(self, msg: PoseStamped):
        if self.state != "IDLE":
            self.get_logger().warn("Already executing – ignoring new pose.")
            return
        self.grasp_pose = msg.pose
        self.get_logger().info(
            f"Received grasp pose in '{msg.header.frame_id}':  "
            f"pos=({self.grasp_pose.position.x:.3f}, "
            f"{self.grasp_pose.position.y:.3f}, "
            f"{self.grasp_pose.position.z:.3f})"
        )
        self._advance("OPEN_GRIPPER")

    # =====================================================================
    #  HARDCODED SIM POSE
    # =====================================================================

    def _publish_hardcoded_pose(self):
        """Publish a test pose (fires once after 3s)."""
        self.startup_timer.cancel()

        qw, qx, qy, qz = ORIENT_FINGERS_DOWN

        hardcoded = PoseStamped()
        hardcoded.header.stamp = self.get_clock().now().to_msg()
        hardcoded.header.frame_id = "base_link"

        if self.sim_mode:
            # Sim test pose (uses hardcoded IK, z-based joint lookup)
            hardcoded.pose.position.x = 0.30
            hardcoded.pose.position.y = -0.15
            hardcoded.pose.position.z = 0.10
            hardcoded.pose.orientation.w = 1.0
            hardcoded.pose.orientation.x = 0.0
            hardcoded.pose.orientation.y = 0.0
            hardcoded.pose.orientation.z = 0.0
        else:
            # Real robot test pose (from test_planner_real.py)
            hardcoded.pose.position.x = -0.10
            hardcoded.pose.position.y = -0.35
            hardcoded.pose.position.z = 0.55
            hardcoded.pose.orientation.w = qw
            hardcoded.pose.orientation.x = qx
            hardcoded.pose.orientation.y = qy
            hardcoded.pose.orientation.z = qz

        self.get_logger().info(
            f"Publishing {'SIM' if self.sim_mode else 'REAL'} hardcoded grasp pose …"
        )
        self._grasp_pose_cb(hardcoded)


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
