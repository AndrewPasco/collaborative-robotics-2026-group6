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
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import time
import copy

# ── Standard ROS 2 message types ────────────────────────────────────────────
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ── TidyBot custom messages (uncomment once the package is built) ───────────
# from tidybot_msgs.msg import ArmCommand, GripperCommand


# =============================================================================
#  CONSTANTS
# =============================================================================

# Arm to use (the TidyBot has left and right WX250s arms)
DEFAULT_ARM = "right"

# Gripper open / close values  (normalized 0.0 = closed, 1.0 = fully open)
GRIPPER_OPEN  = 1.0
GRIPPER_CLOSE = 0.0

# Vertical offset above the grasp pose where the end-effector moves first
PREGRASP_Z_OFFSET = 0.08   # 8 cm above the grasp pose
LIFT_HEIGHT        = 0.15   # 15 cm lift after grasping

# Tolerances & timeouts
JOINT_GOAL_TOLERANCE = 0.05  # rad
MOVE_TIMEOUT         = 8.0   # seconds per motion segment
MAX_GRASP_RETRIES    = 3     # number of ranked grasps to try before giving up

# ── Reason codes returned by execute_grasp() ────────────────────────────────
REASON_SUCCESS   = "SUCCESS"
REASON_IK_FAIL   = "IK_FAIL"
REASON_TIMEOUT   = "TIMEOUT"
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
        4. Close gripper
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

        self.arm_name       = self.get_parameter("arm_name").value
        self.use_hardcoded  = self.get_parameter("use_hardcoded_pose").value
        self.sim_mode       = self.get_parameter("sim_mode").value

        self.get_logger().info(
            f"ManipulationExecutor starting  |  arm={self.arm_name}  "
            f"sim={self.sim_mode}  hardcoded={self.use_hardcoded}"
        )

        # ─── State ──────────────────────────────────────────────────────
        self.current_joint_state: JointState | None = None
        self.is_executing = False  # prevent overlapping executions

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
        # In ROS 2, subscribing is simple:
        #   self.create_subscription(MessageType, "/topic_name", callback, qos)
        #
        # The grasp planner (grasp_planner_node.py – DOES NOT EXIST YET)
        # will eventually publish geometry_msgs/PoseStamped messages on
        # the topic /planned_grasp.  When a message arrives, our callback
        # _grasp_pose_cb fires and we start the pick sequence.
        # ────────────────────────────────────────────────────────────────

        self.grasp_pose_sub = self.create_subscription(
            PoseStamped,
            "/planned_grasp",          # ← topic the grasp planner will publish to
            self._grasp_pose_cb,
            10,
            callback_group=self.cb_group,
        )

        # ================================================================
        #  PUBLISHERS
        # ================================================================

        # Arm joint-position command (Float64MultiArray – consumed by
        # arm_wrapper_node / arm_controller_node)
        self.arm_joint_pub = self.create_publisher(
            Float64MultiArray,
            f"/{self.arm_name}_arm/joint_position_cmd",
            10,
        )

        # Gripper command (normalized 0–1, consumed by gripper_wrapper_node)
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
        #  SERVICE CLIENTS  (to the existing motion-planner node)
        # ================================================================
        #
        # motion_planner_node.py (sim) or motion_planner_real_node.py (real)
        # exposes a service that takes a target Pose and returns the IK
        # joint solution + optionally executes it.
        #
        # The exact service type depends on what is defined in your stack.
        # Below is a *placeholder* using a generic PoseStamped → JointTrajectory
        # pattern.  Replace the service type with the real custom_srvs once
        # the message package is built.
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
            # Fire once after 3 seconds to give other nodes time to start
            self.startup_timer = self.create_timer(
                3.0, self._publish_hardcoded_pose, callback_group=self.cb_group
            )

        self.get_logger().info("ManipulationExecutor ready – waiting for grasp poses …")

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
        # Cancel the timer so this only fires once
        self.startup_timer.cancel()

        ####################################################################
        #  HARDCODED VALUES – a point ~30 cm in front of the right arm,    #
        #  10 cm above the table surface, with a top-down grasp            #
        #  orientation (gripper pointing straight down, Z-axis down).       #
        ####################################################################
        hardcoded = PoseStamped()
        hardcoded.header.stamp    = self.get_clock().now().to_msg()
        hardcoded.header.frame_id = "base_link"       # robot base frame

        # Position (meters, in base_link frame)
        hardcoded.pose.position.x = 0.30              # 30 cm forward
        hardcoded.pose.position.y = -0.15             # 15 cm to the right
        hardcoded.pose.position.z = 0.10              # 10 cm above table

        # Orientation – top-down grasp  (180° rotation about the X-axis so
        # the gripper fingers point downward)
        # Quaternion for Rx(π):  (w=0, x=1, y=0, z=0)
        hardcoded.pose.orientation.w = 0.0
        hardcoded.pose.orientation.x = 1.0
        hardcoded.pose.orientation.y = 0.0
        hardcoded.pose.orientation.z = 0.0
        ####################################################################
        #  END HARDCODED VALUES                                            #
        ####################################################################

        self.get_logger().info(
            "🔧 Publishing HARDCODED grasp pose for simulation testing …"
        )

        # Feed it into the same callback the real planner would trigger
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

        When the grasp-planning node (grasp_planner_node.py) publishes a
        PoseStamped on /planned_grasp, this callback fires and we run
        the full pick sequence.

        For now (simulation), the hardcoded timer above publishes one
        pose into this same callback.

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
        frame_id   = msg.header.frame_id

        self.get_logger().info(
            f"Received grasp pose in frame '{frame_id}':  "
            f"pos=({grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, "
            f"{grasp_pose.position.z:.3f})"
        )

        # Run the pick sequence (blocking within this callback – ok because
        # we use a ReentrantCallbackGroup + MultiThreadedExecutor)
        reason = self._execute_pick_sequence(grasp_pose)

        # Publish result
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
            2. Move to pre-grasp
            3. Move to grasp pose
            4. Close gripper
            5. Lift
        Returns a reason code string.
        """

        # ── Step 1: Open gripper ────────────────────────────────────────
        self.get_logger().info("Step 1/5 – Opening gripper …")
        self._command_gripper(GRIPPER_OPEN)
        time.sleep(1.0)  # wait for gripper to fully open

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

        # ── Step 4: Close gripper ──────────────────────────────────────
        self.get_logger().info("Step 4/5 – Closing gripper …")
        self._command_gripper(GRIPPER_CLOSE)
        time.sleep(1.0)  # let gripper close

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

        self.get_logger().info("✅  Object picked up successfully!")
        return REASON_SUCCESS

    # =====================================================================
    #  MOTION HELPERS
    # =====================================================================

    def _move_to_pose(self, target_pose: Pose) -> bool:
        """
        Send a Cartesian target to the motion planner and wait for execution.

        ────────────────────────────────────────────────────────────────────
        INTEGRATION POINT:
        Replace the body of this method with a real service call to the
        motion-planner node once the custom service type is available.
        ────────────────────────────────────────────────────────────────────

        For now we do a PLACEHOLDER that:
          a) Converts the pose to a simple joint command via a naive IK stub
          b) Publishes a Float64MultiArray to the arm joint-position topic
          c) Waits until joint_states are close enough (or timeout)
        """

        # ─────────────────────────────────────────────────────────────────
        #  OPTION A  (FUTURE) – Call the motion-planner service
        # ─────────────────────────────────────────────────────────────────
        # Uncomment once custom_srvs/PlanToPose is available:
        #
        # from custom_srvs.srv import PlanToPose
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
        # request.execute = True      # ask the planner to also execute
        # request.arm_name = self.arm_name
        #
        # future = self.plan_to_pose_client.call_async(request)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=MOVE_TIMEOUT)
        #
        # if future.result() is None:
        #     return False
        # response = future.result()
        # return response.success
        # ─────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────
        #  OPTION B  (CURRENT PLACEHOLDER) – Publish a joint command
        #  directly.  This uses a *very* rough stand-in; the real pipeline
        #  should go through the motion-planner IK solver.
        # ─────────────────────────────────────────────────────────────────

        ####################################################################
        #  HARDCODED JOINT TARGETS — PLACEHOLDER IK                        #
        #                                                                   #
        #  These are example joint angles for the WX250s that approximate  #
        #  a top-down reach at roughly (0.30, -0.15, z) in front of the    #
        #  right arm.  They are NOT real IK solutions — replace with the   #
        #  motion-planner service call above once available.                #
        ####################################################################
        z = target_pose.position.z

        # Very rough joint-angle lookup based on Z height
        if z > 0.20:
            # High / lifted pose
            joint_targets = [0.0, -0.5, 0.3, 0.0, -0.8, 0.0]   # [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
        elif z > 0.10:
            # Pre-grasp height
            joint_targets = [0.0, 0.2, 0.3, 0.0, -0.5, 0.0]
        else:
            # Grasp / table height
            joint_targets = [0.0, 0.5, 0.5, 0.0, -0.2, 0.0]
        ####################################################################
        #  END HARDCODED JOINT TARGETS                                     #
        ####################################################################

        # Publish joint command
        cmd = Float64MultiArray()
        cmd.data = joint_targets
        self.arm_joint_pub.publish(cmd)
        self.get_logger().info(f"  → Published joint cmd: {[f'{j:.2f}' for j in joint_targets]}")

        # Wait for the arm to reach the target (poll joint_states)
        return self._wait_for_joint_target(joint_targets, timeout=MOVE_TIMEOUT)

    def _wait_for_joint_target(
        self, target: list[float], timeout: float = MOVE_TIMEOUT
    ) -> bool:
        """
        Block until the arm's joint positions are within tolerance of
        `target`, or until `timeout` seconds elapse.
        """
        start = time.time()
        rate = self.create_rate(10)  # 10 Hz polling

        while (time.time() - start) < timeout:
            if self.current_joint_state is not None:
                # Extract the first 6 joint positions (the arm joints)
                # Joint ordering depends on the URDF; you may need to filter
                # by name to pick only the arm joints.
                positions = list(self.current_joint_state.position[:6])
                error = np.linalg.norm(np.array(positions) - np.array(target))
                if error < JOINT_GOAL_TOLERANCE:
                    self.get_logger().info(f"  ✓ Joint target reached (err={error:.4f})")
                    return True
            try:
                rate.sleep()
            except Exception:
                pass

        self.get_logger().warn(f"  ✗ Joint target NOT reached within {timeout}s")
        return False

    # =====================================================================
    #  GRIPPER HELPER
    # =====================================================================

    def _command_gripper(self, value: float):
        """
        Publish a normalized gripper command (0.0 = closed, 1.0 = open).
        The gripper_wrapper_node translates this to real PWM if on hardware.
        """
        msg = Float64MultiArray()
        msg.data = [float(value)]
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"  → Gripper cmd: {value:.2f}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)

    node = ManipulationExecutorNode()

    # Use a multi-threaded executor so that the blocking pick sequence
    # doesn't prevent joint_state callbacks from being delivered.
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
