#!/usr/bin/env python3
"""
manipulation_executor_node.py

ME326 Collaborative Robotics – Group 6
Authors: LY, OP (Vision + Manipulation subteam)

This ROS 2 node is the "brain calls this to grasp" entry point.
It receives a target grasp pose (PoseStamped), plans a pregrasp→grasp→lift→carry
sequence through the existing motion-planner service, and drives the arm + gripper
through the sequence while monitoring joint-state feedback.

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
  - /right_arm/cmd      (tidybot_msgs/ArmCommand)  – arm joint position commands
  - /right_gripper/cmd  (std_msgs/Float64MultiArray) – gripper commands
  - /joint_states        (sensor_msgs/JointState)    – arm state feedback
  - /planned_grasp       (geometry_msgs/PoseStamped) – grasp target from planner
  - /manipulation/task_status (std_msgs/String)      – result status

GRIPPER CONVENTION (same as test_arms_sim.py):
  0.0 = fully OPEN
  1.0 = fully CLOSED
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import time
import copy

# ── Standard ROS 2 message types ────────────────────────────────────────────
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray

# ── TidyBot custom message (same as test_arms_sim.py) ───────────────────────
from tidybot_msgs.msg import ArmCommand


# =============================================================================
#  CONSTANTS
# =============================================================================

# Arm to use (the TidyBot has left and right WX250s arms)
DEFAULT_ARM = "right"

# Gripper values (matching test_arms_sim.py convention)
GRIPPER_OPEN = 0.0    # 0.0 = fully open
GRIPPER_CLOSE = 1.0   # 1.0 = fully closed

# Vertical offset above the grasp pose where the end-effector moves first
PREGRASP_Z_OFFSET = 0.08   # 8 cm above the grasp pose
LIFT_HEIGHT = 0.15          # 15 cm lift after grasping

# Motion timing
MOVE_DURATION = 2.0         # seconds for arm interpolation (ArmCommand.duration)
SETTLE_TIME = 0.5           # extra wait after arm move completes
GRIPPER_WAIT = 1.5          # seconds to wait for gripper open/close

# ── Joint state indices for the right arm ────────────────────────────────────
# From /joint_states:
#   [0] camera_pan        [1] camera_tilt
#   [2] right_waist       [3] right_shoulder    [4] right_elbow
#   [5] right_forearm_roll [6] right_wrist_angle [7] right_wrist_rotate
#   [8] right_left_finger [9] right_right_finger
#   [10-17] left arm + fingers
RIGHT_ARM_START_IDX = 2
RIGHT_ARM_END_IDX = 8  # exclusive, so indices 2-7

# ── Reason codes returned by execute_grasp() ────────────────────────────────
REASON_SUCCESS = "SUCCESS"
REASON_IK_FAIL = "IK_FAIL"
REASON_TIMEOUT = "TIMEOUT"
REASON_EXEC_FAIL = "EXEC_FAIL"


# =============================================================================
#  NODE
# =============================================================================

class ManipulationExecutorNode(Node):
    """
    Subscribes to grasp poses and executes pick-up sequences.

    Lifecycle of a single grasp attempt:
        1. Open gripper
        2. Move to PREGRASP (grasp pose + z offset)
        3. Move to GRASP pose
        4. Pause at grasp pose, then close gripper
        5. LIFT (grasp pose + lift height)
        6. (Optional) Move to CARRY pose
        7. Publish status
    """

    def __init__(self):
        super().__init__("manipulation_executor_node")

        # Allow service calls inside subscription callbacks
        self.cb_group = ReentrantCallbackGroup()

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("arm_name", DEFAULT_ARM)
        self.declare_parameter("use_hardcoded_pose", True)   # flip to False once grasp planner exists
        self.declare_parameter("sim_mode", True)             # True = MuJoCo sim, False = real HW

        self.arm_name = self.get_parameter("arm_name").value
        self.use_hardcoded = self.get_parameter("use_hardcoded_pose").value
        self.sim_mode = self.get_parameter("sim_mode").value

        self.get_logger().info(
            f"ManipulationExecutor starting  |  arm={self.arm_name}  "
            f"sim={self.sim_mode}  hardcoded={self.use_hardcoded}"
        )

        # ─── State ──────────────────────────────────────────────────────
        self.current_joint_state: JointState | None = None
        self.is_executing = False           # prevent overlapping executions
        self._gripper_value = GRIPPER_OPEN  # current desired gripper state

        # ================================================================
        #  SUBSCRIBERS
        # ================================================================

        # 1) Joint states – used to monitor arm progress / confirm arrival
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10,
            callback_group=self.cb_group,
        )

        # ────────────────────────────────────────────────────────────────
        # 2) Grasp-pose topic – the grasp-planning node will publish here
        # ────────────────────────────────────────────────────────────────
        #
        # The grasp planner (grasp_planner_node.py – DOES NOT EXIST YET)
        # will eventually publish geometry_msgs/PoseStamped messages on
        # the topic /planned_grasp.  When a message arrives, our callback
        # _grasp_pose_cb fires and we start the pick sequence.
        #
        # When the real grasp planner is ready:
        #   1. Set the parameter  use_hardcoded_pose := False
        #   2. The grasp planner publishes to /planned_grasp
        #   3. This callback receives it – no other changes needed here.
        # ────────────────────────────────────────────────────────────────

        self.grasp_pose_sub = self.create_subscription(
            PoseStamped,
            "/planned_grasp",
            self._grasp_pose_cb,
            10,
            callback_group=self.cb_group,
        )

        # ================================================================
        #  PUBLISHERS
        # ================================================================

        # Arm command (ArmCommand with joint_positions + duration,
        # same as test_arms_sim.py uses)
        self.arm_cmd_pub = self.create_publisher(
            ArmCommand,
            f"/{self.arm_name}_arm/cmd",
            10,
        )

        # Gripper command (Float64MultiArray, 0.0=open, 1.0=closed)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            f"/{self.arm_name}_gripper/cmd",
            10,
        )

        # Task status – lets the high-level state machine know what happened
        self.status_pub = self.create_publisher(
            String,
            "/manipulation/task_status",
            10,
        )

        # ================================================================
        #  50 Hz CONTROL LOOP – continuously re-asserts gripper state
        # ================================================================
        # This is critical: without continuous gripper publishing, the
        # gripper drifts when arm commands are sent.  This matches the
        # pattern used in test_arms_sim.py.
        self.control_timer = self.create_timer(0.02, self._control_loop)

        # ================================================================
        #  SERVICE CLIENTS  (to the existing motion-planner node)
        # ================================================================
        #
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
        #  (OPTIONAL) TIMER – publish hardcoded pose once for sim testing
        # ================================================================
        if self.use_hardcoded:
            self.startup_timer = self.create_timer(
                3.0, self._publish_hardcoded_pose, callback_group=self.cb_group
            )

        self.get_logger().info("ManipulationExecutor ready – waiting for grasp poses …")

    # =====================================================================
    #  50 Hz CONTROL LOOP
    # =====================================================================

    def _control_loop(self):
        """
        Continuously publish the desired gripper state at 50 Hz.
        This prevents the gripper from drifting when arm commands are sent.
        Same pattern as test_arms_sim.py.
        """
        msg = Float64MultiArray()
        msg.data = [float(self._gripper_value)]
        self.gripper_pub.publish(msg)

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
        #  orientation (gripper pointing straight down, Z-axis down).       #
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
        /planned_grasp, this callback fires and we run the full
        pick sequence.

        When the real grasp planner is ready:
          1. Set the parameter  use_hardcoded_pose := False
          2. The grasp planner publishes to /planned_grasp
          3. This callback receives it – no other changes needed here.
        ────────────────────────────────────────────────────────────────
        """
        if self.is_executing:
            self.get_logger().warn("Already executing a grasp – ignoring new pose.")
            return

        self.is_executing = True
        grasp_pose = msg.pose
        frame_id = msg.header.frame_id

        self.get_logger().info(
            f"Received grasp pose in frame '{frame_id}':  "
            f"pos=({grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, "
            f"{grasp_pose.position.z:.3f})"
        )

        reason = self._execute_pick_sequence(grasp_pose)

        status_msg = String()
        status_msg.data = reason
        self.status_pub.publish(status_msg)

        self.get_logger().info(f"Pick sequence finished – result: {reason}")
        self.is_executing = False

    # =====================================================================
    #  PICK SEQUENCE
    # =====================================================================

    def _execute_pick_sequence(self, grasp_pose: Pose) -> str:
        """
        Full pick-up sequence:
            1. Open gripper
            2. Move to pre-grasp  (gripper stays open via 50Hz loop)
            3. Move to grasp pose (gripper stays open via 50Hz loop)
            4. Pause at grasp pose, then close gripper
            5. Lift with gripper closed
        Returns a reason code string.
        """

        # ── Step 1: Open gripper ────────────────────────────────────────
        self.get_logger().info("Step 1/5 – Opening gripper …")
        self._gripper_value = GRIPPER_OPEN  # 50Hz loop will publish this
        time.sleep(GRIPPER_WAIT)

        # ── Step 2: Move to pre-grasp ──────────────────────────────────
        pregrasp_pose = copy.deepcopy(grasp_pose)
        pregrasp_pose.position.z += PREGRASP_Z_OFFSET

        self.get_logger().info(
            f"Step 2/5 – Moving to PRE-GRASP  z={pregrasp_pose.position.z:.3f} …"
        )
        success = self._move_to_pose(pregrasp_pose)
        if not success:
            self.get_logger().error("Pre-grasp motion failed.")
            return REASON_IK_FAIL

        # ── Step 3: Descend to grasp pose ──────────────────────────────
        self.get_logger().info("Step 3/5 – Descending to GRASP pose …")
        success = self._move_to_pose(grasp_pose)
        if not success:
            self.get_logger().error("Grasp-pose motion failed.")
            return REASON_EXEC_FAIL

        # ── Step 4: Pause at grasp pose, then close gripper ────────────
        self.get_logger().info("Step 4/5 – Settled at grasp pose. Closing gripper …")
        self._gripper_value = GRIPPER_CLOSE  # 50Hz loop now publishes CLOSE
        time.sleep(GRIPPER_WAIT)
        self.get_logger().info("  Gripper closed.")

        # ── Step 5: Lift ───────────────────────────────────────────────
        lift_pose = copy.deepcopy(grasp_pose)
        lift_pose.position.z += LIFT_HEIGHT

        self.get_logger().info(
            f"Step 5/5 – LIFTING to z={lift_pose.position.z:.3f} …"
        )
        success = self._move_to_pose(lift_pose)
        if not success:
            self.get_logger().error("Lift motion failed.")
            return REASON_EXEC_FAIL

        self.get_logger().info("Object picked up successfully!")
        return REASON_SUCCESS

    # =====================================================================
    #  MOTION HELPERS
    # =====================================================================

    def _move_to_pose(self, target_pose: Pose) -> bool:
        """
        Send a Cartesian target to the arm and wait for execution.

        ────────────────────────────────────────────────────────────────────
        INTEGRATION POINT:
        Replace the body of this method with a real service call to the
        motion-planner node once the custom service type is available.
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
        # if not self.plan_to_pose_client.wait_for_service(timeout_sec=5.0):
        #     self.get_logger().error("Motion-planner service not available!")
        #     return False
        #
        # request = PlanToPose.Request()
        # request.target_pose = PoseStamped(
        #     header=Header(frame_id="base_link"),
        #     pose=target_pose,
        # )
        # request.execute = True
        # request.arm_name = self.arm_name
        #
        # future = self.plan_to_pose_client.call_async(request)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=MOVE_DURATION + 2.0)
        #
        # if future.result() is None:
        #     return False
        # return future.result().success
        # ─────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────
        #  OPTION B  (CURRENT PLACEHOLDER) – Send ArmCommand directly
        #  using hardcoded joint angles.  Replace with motion-planner
        #  service call above once available.
        # ─────────────────────────────────────────────────────────────────

        ####################################################################
        #  HARDCODED JOINT TARGETS — PLACEHOLDER IK                        #
        #                                                                   #
        #  These are example joint angles for the WX250s that approximate  #
        #  a top-down reach at roughly (0.30, -0.15, z) in front of the    #
        #  right arm.  They are NOT real IK solutions — replace with the   #
        #  motion-planner service call above once available.                #
        #                                                                   #
        #  Joint order: [waist, shoulder, elbow, forearm_roll,              #
        #                wrist_angle, wrist_rotate]                         #
        ####################################################################
        z = target_pose.position.z

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

        # Send arm command using ArmCommand message (same as test_arms_sim.py)
        cmd = ArmCommand()
        cmd.joint_positions = joint_targets
        cmd.duration = MOVE_DURATION
        self.arm_cmd_pub.publish(cmd)

        self.get_logger().info(
            f"  Published ArmCommand: {[f'{j:.2f}' for j in joint_targets]} "
            f"duration={MOVE_DURATION}s"
        )

        # Wait for arm to finish moving (duration + settle time)
        # The 50Hz control loop keeps publishing gripper state during this wait
        time.sleep(MOVE_DURATION + SETTLE_TIME)
        self.get_logger().info("  Arm move complete.")
        return True

    # =====================================================================
    #  GRIPPER HELPER (for logging / manual calls)
    # =====================================================================

    def _set_gripper(self, value: float):
        """
        Set the desired gripper state. The 50Hz control loop will
        continuously publish this value.
        0.0 = fully open, 1.0 = fully closed.
        """
        self._gripper_value = value
        self.get_logger().info(
            f"  Gripper target: {'OPEN' if value == GRIPPER_OPEN else 'CLOSED'} "
            f"({value:.1f})"
        )


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)

    node = ManipulationExecutorNode()

    # Use a multi-threaded executor so that the blocking pick sequence
    # doesn't prevent the 50Hz control loop and joint_state callbacks
    # from being delivered.
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
