#!/usr/bin/env python3
"""
manipulation_executor_node.py

ME326 Collaborative Robotics – Group 6
Authors: LY, OP (Vision + Manipulation subteam)

This ROS 2 node is the "brain calls this to grasp" entry point.
It receives a target grasp pose (PoseStamped) and executes a full
pregrasp → grasp → lift sequence using a STATE MACHINE driven by a
50 Hz timer (same pattern as test_arms_sim.py).

────────────────────────────────────────────────────────────────────────────────
HOW ROS 2 TOPICS / SUBSCRIPTIONS WORK (quick reference for the team):

  • A "topic" is a named channel, e.g. /planned_grasp
  • Any node can PUBLISH messages to a topic (many-to-many)
  • Any node can SUBSCRIBE to a topic and get a callback each time a new message
    arrives
  • Messages have a specific type, e.g. geometry_msgs/msg/PoseStamped
  • Publisher:   self.publisher  = self.create_publisher(MsgType, '/topic', qos)
  • Subscriber:  self.subscriber = self.create_subscription(
                      MsgType, '/topic', self.callback_fn, qos)
  • Services are request/response (like a function call across nodes)
  • Actions are for long-running tasks with feedback (not used here yet)
────────────────────────────────────────────────────────────────────────────────

TOPICS USED (matching existing TidyBot2 stack conventions):
  - /right_arm/cmd       (tidybot_msgs/ArmCommand)   – arm joint position commands
  - /right_gripper/cmd   (std_msgs/Float64MultiArray) – gripper commands
  - /joint_states         (sensor_msgs/JointState)    – arm state feedback
  - /planned_grasp        (geometry_msgs/PoseStamped) – grasp target from planner
  - /manipulation/task_status (std_msgs/String)       – result status

GRIPPER CONVENTION (same as test_arms_sim.py):
  -1.0 = fully OPEN
  1.0 = fully CLOSED

ARCHITECTURE:
  This node uses a STATE MACHINE inside a 50 Hz timer callback (same pattern
  as test_arms_sim.py).  There are NO blocking time.sleep() calls anywhere.
  The timer fires every 20 ms, publishes the gripper command every tick, and
  advances through states based on elapsed time.  This guarantees the gripper
  is continuously held at the desired value even while the arm is moving.

  States:
    IDLE            – waiting for a grasp pose
    OPEN_GRIPPER    – opening gripper before approach
    MOVE_PREGRASP   – arm moving to pre-grasp position above target
    MOVE_GRASP      – arm descending to grasp position
    PAUSE_AT_GRASP  – settling at grasp pose (gripper still open)
    CLOSE_GRIPPER   – closing gripper on object
    MOVE_LIFT       – lifting object
    DONE            – sequence complete, publish status
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import time
import copy

# ── Standard ROS 2 message types ────────────────────────────────────────────
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray

# ── TidyBot custom message (same as test_arms_sim.py) ───────────────────────
from tidybot_msgs.msg import ArmCommand


# =============================================================================
#  CONSTANTS
# =============================================================================

DEFAULT_ARM = "right"

# Gripper values (matching test_arms_sim.py)
GRIPPER_OPEN = -1.0    # 0.0 = fully open
GRIPPER_CLOSE = 1.0   # 1.0 = fully closed

# Vertical offsets
PREGRASP_Z_OFFSET = 0.08   # 8 cm above the grasp pose
LIFT_HEIGHT = 0.15          # 15 cm lift after grasping

# Timing (seconds)
GRIPPER_OPEN_TIME = 1.5     # how long to hold gripper open before moving
MOVE_DURATION = 2.0         # arm interpolation time (ArmCommand.duration)
SETTLE_TIME = 0.5           # pause after arm arrives before next step
PAUSE_AT_GRASP_TIME = 1.0   # pause at grasp pose before closing gripper
GRIPPER_CLOSE_TIME = 1.5    # how long to wait for gripper to close

# ── Joint state indices for the right arm ────────────────────────────────────
# From /joint_states:
#   [0] camera_pan        [1] camera_tilt
#   [2] right_waist       [3] right_shoulder    [4] right_elbow
#   [5] right_forearm_roll [6] right_wrist_angle [7] right_wrist_rotate
#   [8] right_left_finger [9] right_right_finger
#   [10-17] left arm + fingers
RIGHT_ARM_START_IDX = 2
RIGHT_ARM_END_IDX = 8

# Reason codes
REASON_SUCCESS = "SUCCESS"
REASON_IK_FAIL = "IK_FAIL"
REASON_TIMEOUT = "TIMEOUT"
REASON_EXEC_FAIL = "EXEC_FAIL"


# =============================================================================
#  NODE
# =============================================================================

class ManipulationExecutorNode(Node):
    """
    State-machine-based pick executor.

    A 50 Hz timer drives the entire sequence.  Every tick:
      1. Publish the current gripper value (keeps gripper locked in place)
      2. Check which state we're in
      3. If enough time has elapsed, advance to the next state

    This guarantees the gripper is ALWAYS being actively controlled,
    even while the arm is mid-motion.
    """

    def __init__(self):
        super().__init__("manipulation_executor_node")

        self.cb_group = ReentrantCallbackGroup()

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("arm_name", DEFAULT_ARM)
        self.declare_parameter("use_hardcoded_pose", True)
        self.declare_parameter("sim_mode", True)

        self.arm_name = self.get_parameter("arm_name").value
        self.use_hardcoded = self.get_parameter("use_hardcoded_pose").value
        self.sim_mode = self.get_parameter("sim_mode").value

        self.get_logger().info(
            f"ManipulationExecutor starting  |  arm={self.arm_name}  "
            f"sim={self.sim_mode}  hardcoded={self.use_hardcoded}"
        )

        # ─── State machine ──────────────────────────────────────────────
        self.state = "IDLE"
        self.state_start_time = None
        self.logged_state = None        # track which state we already logged
        self.gripper_value = GRIPPER_OPEN
        self.grasp_pose = None          # the target Pose for the current pick
        self.pick_result = None         # reason code after sequence completes

        # ─── Joint state cache ──────────────────────────────────────────
        self.current_joint_state = None

        # ================================================================
        #  SUBSCRIBERS
        # ================================================================

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        # ────────────────────────────────────────────────────────────────
        # Grasp-pose topic – the grasp-planning node will publish here.
        #
        # The grasp planner (grasp_planner_node.py – DOES NOT EXIST YET)
        # will eventually publish geometry_msgs/PoseStamped messages on
        # /planned_grasp.  When a message arrives, _grasp_pose_cb fires
        # and kicks off the state machine.
        #
        # When the real grasp planner is ready:
        #   1. Set parameter  use_hardcoded_pose := False
        #   2. The grasp planner publishes to /planned_grasp
        #   3. This callback receives it – no other changes needed.
        # ────────────────────────────────────────────────────────────────

        self.grasp_pose_sub = self.create_subscription(
            PoseStamped, "/planned_grasp", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )

        # ================================================================
        #  PUBLISHERS
        # ================================================================

        # Arm command (ArmCommand with joint_positions + duration)
        self.arm_cmd_pub = self.create_publisher(
            ArmCommand, f"/{self.arm_name}_arm/cmd", 10,
        )

        # Gripper command (Float64MultiArray, 0.0=open, 1.0=closed)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, f"/{self.arm_name}_gripper/cmd", 10,
        )

        # Task status
        self.status_pub = self.create_publisher(
            String, "/manipulation/task_status", 10,
        )

        # ================================================================
        #  SERVICE CLIENTS  (to the existing motion-planner node)
        # ================================================================
        # Uncomment once custom_srvs/PlanToPose is available:
        # ----------------------------------------------------------------
        # from custom_srvs.srv import PlanToPose
        # self.plan_to_pose_client = self.create_client(
        #     PlanToPose,
        #     f"/{self.arm_name}_arm/plan_to_pose",
        #     callback_group=self.cb_group,
        # )
        # ----------------------------------------------------------------

        # ================================================================
        #  50 Hz CONTROL LOOP  (drives everything)
        # ================================================================
        self.control_timer = self.create_timer(0.02, self._control_loop)

        # ================================================================
        #  STARTUP: publish hardcoded pose after 3 seconds
        # ================================================================
        if self.use_hardcoded:
            self.startup_timer = self.create_timer(
                3.0, self._publish_hardcoded_pose, callback_group=self.cb_group,
            )

        self.get_logger().info("ManipulationExecutor ready – waiting for grasp poses …")

    # =====================================================================
    #  50 Hz CONTROL LOOP  (the heart of the node)
    # =====================================================================

    def _control_loop(self):
        """
        Runs at 50 Hz.  Every tick:
          1. Publish gripper command (keeps gripper locked at desired value)
          2. Check state and elapsed time → advance when ready
        """
        now = time.time()

        # ── Always publish gripper ──────────────────────────────────────
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(self.gripper_value)]
        self.gripper_pub.publish(gripper_msg)

        # ── If idle, nothing else to do ─────────────────────────────────
        if self.state == "IDLE":
            return

        # ── Initialize timing on first tick of a new state ──────────────
        if self.state_start_time is None:
            self.state_start_time = now

        elapsed = now - self.state_start_time

        # ── Log state transitions (once per state) ──────────────────────
        if self.state != self.logged_state:
            self._log_state()
            self.logged_state = self.state

        # ── STATE: OPEN_GRIPPER ─────────────────────────────────────────
        if self.state == "OPEN_GRIPPER":
            self.gripper_value = GRIPPER_OPEN
            if elapsed > GRIPPER_OPEN_TIME:
                self._advance("MOVE_PREGRASP")

        # ── STATE: MOVE_PREGRASP ────────────────────────────────────────
        elif self.state == "MOVE_PREGRASP":
            self.gripper_value = GRIPPER_OPEN
            # Send arm command once (on first tick, logged_state handles this)
            if elapsed < 0.05:
                self._send_arm_to_pose(self.grasp_pose, z_offset=PREGRASP_Z_OFFSET)
            if elapsed > MOVE_DURATION + SETTLE_TIME:
                self._advance("MOVE_GRASP")

        # ── STATE: MOVE_GRASP ──────────────────────────────────────────
        elif self.state == "MOVE_GRASP":
            self.gripper_value = GRIPPER_OPEN
            if elapsed < 0.05:
                self._send_arm_to_pose(self.grasp_pose, z_offset=0.0)
            if elapsed > MOVE_DURATION + SETTLE_TIME:
                self._advance("PAUSE_AT_GRASP")

        # ── STATE: PAUSE_AT_GRASP ──────────────────────────────────────
        elif self.state == "PAUSE_AT_GRASP":
            self.gripper_value = GRIPPER_OPEN  # gripper STAYS OPEN during pause
            if elapsed > PAUSE_AT_GRASP_TIME:
                self._advance("CLOSE_GRIPPER")

        # ── STATE: CLOSE_GRIPPER ───────────────────────────────────────
        elif self.state == "CLOSE_GRIPPER":
            self.gripper_value = GRIPPER_CLOSE
            if elapsed > GRIPPER_CLOSE_TIME:
                self._advance("MOVE_LIFT")

        # ── STATE: MOVE_LIFT ───────────────────────────────────────────
        elif self.state == "MOVE_LIFT":
            self.gripper_value = GRIPPER_CLOSE  # keep gripper closed while lifting
            if elapsed < 0.05:
                self._send_arm_to_pose(self.grasp_pose, z_offset=LIFT_HEIGHT)
            if elapsed > MOVE_DURATION + SETTLE_TIME:
                self._advance("DONE")

        # ── STATE: DONE ────────────────────────────────────────────────
        elif self.state == "DONE":
            self.gripper_value = GRIPPER_CLOSE  # keep holding object
            if elapsed < 0.05:
                # Publish result once
                self.pick_result = REASON_SUCCESS
                status_msg = String()
                status_msg.data = self.pick_result
                self.status_pub.publish(status_msg)
            # Stay in DONE (gripper keeps publishing closed)
            # Reset to IDLE if you want to accept another grasp:
            # if elapsed > 2.0:
            #     self._advance("IDLE")

    # =====================================================================
    #  STATE HELPERS
    # =====================================================================

    def _advance(self, new_state: str):
        """Transition to a new state and reset the timer."""
        self.state = new_state
        self.state_start_time = None  # will be set on next tick

    def _log_state(self):
        """Log a message when entering a new state."""
        messages = {
            "OPEN_GRIPPER":   "Step 1/5 – Opening gripper …",
            "MOVE_PREGRASP":  f"Step 2/5 – Moving to PRE-GRASP (z+{PREGRASP_Z_OFFSET}m) …",
            "MOVE_GRASP":     "Step 3/5 – Descending to GRASP pose …",
            "PAUSE_AT_GRASP": "Step 4/5 – Pausing at grasp pose (gripper open) …",
            "CLOSE_GRIPPER":  "Step 4/5 – Closing gripper …",
            "MOVE_LIFT":      f"Step 5/5 – LIFTING (z+{LIFT_HEIGHT}m) …",
            "DONE":           f"Pick sequence complete – result: {REASON_SUCCESS}",
        }
        msg = messages.get(self.state, f"State: {self.state}")
        self.get_logger().info(msg)

    # =====================================================================
    #  ARM COMMAND HELPER
    # =====================================================================

    def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
        """
        Compute joint targets for a given pose + z_offset and send ArmCommand.

        ────────────────────────────────────────────────────────────────────
        INTEGRATION POINT:
        Replace the hardcoded joint lookup below with a real service call
        to motion_planner_node.py once the custom service type is available.
        ────────────────────────────────────────────────────────────────────
        """

        # ─────────────────────────────────────────────────────────────────
        #  OPTION A  (FUTURE) – Call the motion-planner service
        # ─────────────────────────────────────────────────────────────────
        # Uncomment once custom_srvs/PlanToPose is available:
        #
        # from custom_srvs.srv import PlanToPose
        # from std_msgs.msg import Header
        #
        # target = copy.deepcopy(grasp_pose)
        # target.position.z += z_offset
        #
        # request = PlanToPose.Request()
        # request.target_pose = PoseStamped(
        #     header=Header(frame_id="base_link"),
        #     pose=target,
        # )
        # request.execute = True
        # request.arm_name = self.arm_name
        #
        # future = self.plan_to_pose_client.call_async(request)
        # # Handle response in a callback or check on next timer tick
        # return
        # ─────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────
        #  OPTION B  (CURRENT PLACEHOLDER) – Hardcoded joint lookup
        # ─────────────────────────────────────────────────────────────────

        z = grasp_pose.position.z + z_offset

        ####################################################################
        #  HARDCODED JOINT TARGETS — PLACEHOLDER IK                        #
        #                                                                   #
        #  These are example joint angles for the WX250s that approximate  #
        #  a top-down reach at roughly (0.30, -0.15, z) in front of the    #
        #  right arm.  They are NOT real IK solutions — replace with the   #
        #  motion-planner service call (Option A) once available.           #
        #                                                                   #
        #  Joint order: [waist, shoulder, elbow, forearm_roll,              #
        #                wrist_angle, wrist_rotate]                         #
        ####################################################################
        if z > 0.20:
            # High / lifted pose
            joint_targets = [0.0, -0.5, 0.3, 0.0, -0.8, 0.0]
        elif z > 0.10:
            # Pre-grasp height
            joint_targets = [0.0, 0.2, 0.3, 0.0, -0.5, 0.0]
        else:
            # Grasp / table height
            joint_targets = [0.0, 0.5, 0.5, 0.0, -0.2, 0.0]
        ####################################################################
        #  END HARDCODED JOINT TARGETS                                     #
        ####################################################################

        cmd = ArmCommand()
        cmd.joint_positions = joint_targets
        cmd.duration = MOVE_DURATION
        self.arm_cmd_pub.publish(cmd)

        self.get_logger().info(
            f"  ArmCommand: {[f'{j:.2f}' for j in joint_targets]}  "
            f"duration={MOVE_DURATION}s  (target z={z:.3f})"
        )

    # =====================================================================
    #  CALLBACKS
    # =====================================================================

    def _joint_state_cb(self, msg: JointState):
        """Cache the latest joint state for monitoring."""
        self.current_joint_state = msg

    def _grasp_pose_cb(self, msg: PoseStamped):
        """
        ────────────────────────────────────────────────────────────────
        THIS IS THE MAIN ENTRY POINT FOR EXECUTION.

        When the grasp-planning node publishes a PoseStamped on
        /planned_grasp, this callback stores the pose and kicks off the
        state machine.

        When the real grasp planner is ready:
          1. Set parameter  use_hardcoded_pose := False
          2. The grasp planner publishes to /planned_grasp
          3. This callback receives it – no other changes needed.
        ────────────────────────────────────────────────────────────────
        """
        if self.state != "IDLE":
            self.get_logger().warn("Already executing a pick – ignoring new pose.")
            return

        self.grasp_pose = msg.pose
        frame_id = msg.header.frame_id

        self.get_logger().info(
            f"Received grasp pose in frame '{frame_id}':  "
            f"pos=({self.grasp_pose.position.x:.3f}, "
            f"{self.grasp_pose.position.y:.3f}, "
            f"{self.grasp_pose.position.z:.3f})"
        )

        # Kick off the state machine
        self._advance("OPEN_GRIPPER")

    # =====================================================================
    #  HARDCODED SIMULATION POSE
    # =====================================================================

    def _publish_hardcoded_pose(self):
        """
        ####################################################################
        #  HARDCODED EXAMPLE – FOR SIMULATION / TESTING ONLY               #
        #                                                                   #
        #  This pose is a made-up grasp target roughly in front of the     #
        #  right arm of a TidyBot WX250s, at table height.                 #
        #  Replace / remove once the real grasp_planner_node.py exists.    #
        ####################################################################
        """
        self.startup_timer.cancel()

        ####################################################################
        #  HARDCODED VALUES – a point ~30 cm in front of the right arm,    #
        #  10 cm above the table surface, with a top-down grasp            #
        #  orientation (gripper pointing straight down).                    #
        ####################################################################
        hardcoded = PoseStamped()
        hardcoded.header.stamp = self.get_clock().now().to_msg()
        hardcoded.header.frame_id = "base_link"

        hardcoded.pose.position.x = 0.30
        hardcoded.pose.position.y = -0.15
        hardcoded.pose.position.z = 0.10

        hardcoded.pose.orientation.w = 0.0
        hardcoded.pose.orientation.x = 1.0
        hardcoded.pose.orientation.y = 0.0
        hardcoded.pose.orientation.z = 0.0
        ####################################################################
        #  END HARDCODED VALUES                                            #
        ####################################################################

        self.get_logger().info(
            "Publishing HARDCODED grasp pose for simulation testing …"
        )
        self._grasp_pose_cb(hardcoded)


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)

    node = ManipulationExecutorNode()

    # MultiThreadedExecutor so the 50 Hz timer and joint_state callbacks
    # can run concurrently without blocking each other.
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
