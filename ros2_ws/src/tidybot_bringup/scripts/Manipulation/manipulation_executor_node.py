#!/usr/bin/env python3
"""
manipulation_executor_node.py

ME326 Collaborative Robotics – Group 6
Authors: LY, OP (Vision + Manipulation subteam)

This ROS 2 node is the "brain calls this to grasp" entry point.
It receives a target grasp pose (PoseStamped) and executes a full
pregrasp → grasp → lift sequence using a STATE MACHINE driven by a
50 Hz timer (same pattern as test_arms_sim.py).

ARCHITECTURE:
  50 Hz control loop publishes gripper every tick and checks state.
  State transitions are based on JOINT STATE FEEDBACK, not timers:
    - Arm moves advance when actual joint positions match the target
    - Gripper close advances when finger joints confirm closure
  This guarantees the arm has physically arrived before we proceed.

  States:
    IDLE            – waiting for a grasp pose
    OPEN_GRIPPER    – opening gripper before approach
    MOVE_PREGRASP   – arm moving to pre-grasp position above target
    MOVE_GRASP      – arm descending to grasp position
    PAUSE_AT_GRASP  – settling at grasp pose (gripper still open)
    CLOSE_GRIPPER   – closing gripper on object
    MOVE_LIFT       – lifting object
    DONE            – sequence complete, publish status

TOPICS USED:
  - /right_arm/cmd       (tidybot_msgs/ArmCommand)   – arm joint commands
  - /right_gripper/cmd   (std_msgs/Float64MultiArray) – gripper commands
  - /joint_states         (sensor_msgs/JointState)    – feedback
  - /planned_grasp        (geometry_msgs/PoseStamped) – grasp target
  - /manipulation/task_status (std_msgs/String)       – result

GRIPPER CONVENTION:
  -1.0 = fully OPEN
   1.0 = fully CLOSED
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

# ── Joint state feedback thresholds ──────────────────────────────────────────
ARM_ARRIVAL_TOLERANCE = 0.05     # rad – how close joints must be to target
GRIPPER_OPEN_THRESHOLD = 0.03    # finger position above this = open enough
GRIPPER_CLOSED_THRESHOLD = 0.018 # finger position below this = fully closed
PAUSE_AT_GRASP_SECS = 10      # seconds to hold still at grasp before closing

# Safety timeouts (fallback if joints never converge)
ARM_MOVE_TIMEOUT = 10000.0
GRIPPER_TIMEOUT = 10000.0

# ── Joint state indices ──────────────────────────────────────────────────────
# [0] camera_pan  [1] camera_tilt
# [2] right_waist [3] right_shoulder [4] right_elbow
# [5] right_forearm_roll [6] right_wrist_angle [7] right_wrist_rotate
# [8] right_left_finger  [9] right_right_finger
RIGHT_ARM_INDICES = [2, 3, 4, 5, 6, 7]
RIGHT_FINGER_LEFT = 8
RIGHT_FINGER_RIGHT = 9

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

        self.get_logger().info(
            f"ManipulationExecutor starting  |  arm={self.arm_name}  "
            f"sim={self.sim_mode}  hardcoded={self.use_hardcoded}"
            f"planner={self.use_planner}"
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

        # ================================================================
        #  SUBSCRIBERS
        # ================================================================

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        # ────────────────────────────────────────────────────────────────
        # Grasp-pose topic
        #
        # The grasp planner (DOES NOT EXIST YET) will publish
        # PoseStamped on /planned_grasp.
        #
        # When the real grasp planner is ready:
        #   1. Set parameter  use_hardcoded_pose := False
        #   2. Grasp planner publishes to /planned_grasp
        #   3. This callback receives it – no changes needed.
        # ────────────────────────────────────────────────────────────────

        self.grasp_pose_sub = self.create_subscription(
            PoseStamped, "/planned_grasp", self._grasp_pose_cb, 10,
            callback_group=self.cb_group,
        )

        # ================================================================
        #  PUBLISHERS
        # ================================================================

        self.arm_cmd_pub = self.create_publisher(
            ArmCommand, f"/{self.arm_name}_arm/cmd", 10,
        )

        self.gripper_pub = self.create_publisher(
            Float64MultiArray, f"/{self.arm_name}_gripper/cmd", 10,
        )

        self.status_pub = self.create_publisher(
            String, "/manipulation/task_status", 10,
        )

        # ================================================================
        #  SERVICE CLIENTS (future – motion planner)
        # ================================================================
        # Uncomment once custom_srvs/PlanToPose is available:
        # from custom_srvs.srv import PlanToPose
        # self.plan_to_pose_client = self.create_client(
        #     PlanToPose, f"/{self.arm_name}_arm/plan_to_pose",
        #     callback_group=self.cb_group,
        # )
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

        self.get_logger().info("ManipulationExecutor ready – waiting for grasp poses …")

    # =====================================================================
    #  JOINT STATE HELPERS
    # =====================================================================

    def _get_arm_positions(self):
        """Return current 6-DOF arm joint positions, or None."""
        if self.current_joint_state is None:
            return None
        return [self.current_joint_state.position[i] for i in RIGHT_ARM_INDICES]

    def _get_finger_positions(self):
        """Return (left_finger, right_finger) positions, or None."""
        if self.current_joint_state is None:
            return None
        return (
            self.current_joint_state.position[RIGHT_FINGER_LEFT],
            self.current_joint_state.position[RIGHT_FINGER_RIGHT],
        )

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
        # self.get_logger().info(f"  gripper is CLOSED cmd func: left: {left}, right: {right} …")
        return left < GRIPPER_CLOSED_THRESHOLD and right < GRIPPER_CLOSED_THRESHOLD

    def _gripper_is_open(self):
        """Check if finger joints confirm gripper is open."""
        fingers = self._get_finger_positions()
        if fingers is None:
            return False
        left, right = fingers
        # self.get_logger().info(f"  gripper is OPEN cmd func: left: {left}, right: {right} …")
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
        #  Condition to advance: gripper fingers confirm open, OR timeout
        # =================================================================
        if self.state == "OPEN_GRIPPER":
            self.gripper_value = GRIPPER_OPEN
            # if self._gripper_is_open() or elapsed > GRIPPER_TIMEOUT:
            if self._gripper_is_open():
                # if elapsed > GRIPPER_TIMEOUT:
                #     self.get_logger().warn("  Gripper open timed out – proceeding")
                # else:
                #     self.get_logger().info("  Gripper confirmed open.")
                self.get_logger().info("  Gripper confirmed open.")
                # self._advance("MOVE_PREGRASP")
                self._advance("MOVE_GRASP")

        # =================================================================
        #  STATE: MOVE_PREGRASP
        #  Send arm command once, then wait for joints to arrive
        # =================================================================
        elif self.state == "MOVE_PREGRASP":
            self.gripper_value = GRIPPER_OPEN
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=PREGRASP_Z_OFFSET)
                self.arm_cmd_sent = True
            if self._arm_at_target() or elapsed > ARM_MOVE_TIMEOUT:
                if elapsed > ARM_MOVE_TIMEOUT:
                    self.get_logger().warn("  Pre-grasp move timed out – proceeding")
                else:
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
            # if self._arm_at_target() or elapsed > ARM_MOVE_TIMEOUT:
            if self._arm_at_target():
            #     if elapsed > ARM_MOVE_TIMEOUT:
            #         self.get_logger().warn("  Grasp move timed out – proceeding")
            #     else:
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
        #  Condition to advance: finger joints confirm closed, OR timeout
        # =================================================================
        elif self.state == "CLOSE_GRIPPER":
            self.gripper_value = GRIPPER_CLOSE
            # self.get_logger().info(f"  Gripper closed :{self._gripper_is_closed()}'")
            # if self._gripper_is_closed() or elapsed > GRIPPER_TIMEOUT:
            if self._gripper_is_closed():
              # self.get_logger().info(f"  Gripper closed :{self._gripper_is_closed()}'")
              fingers = self._get_finger_positions()
              # if elapsed > GRIPPER_TIMEOUT:
              #     self.get_logger().warn(
              #         f"  Gripper close timed out (fingers: {fingers}) – proceeding"
              #     )
              # else:
              #     self.get_logger().info(
              #         f"  Gripper confirmed closed (fingers: {fingers})."
              #     )
              self.get_logger().info(
                      f"  Gripper confirmed closed (fingers: {fingers})."
                  )
              self._advance("MOVE_LIFT")

        # =================================================================
        #  STATE: MOVE_LIFT
        #  Send arm command once, then wait for joints to arrive
        # =================================================================
        elif self.state == "MOVE_LIFT":
            self.gripper_value = GRIPPER_CLOSE
            if not self.arm_cmd_sent:
                self._send_arm_to_pose(self.grasp_pose, z_offset=LIFT_HEIGHT)
                self.arm_cmd_sent = True
            if self._arm_at_target() or elapsed > ARM_MOVE_TIMEOUT:
                if elapsed > ARM_MOVE_TIMEOUT:
                    self.get_logger().warn("  Lift move timed out")
                else:
                    self.get_logger().info("  Arm arrived at lift pose.")
                self._advance("DONE")

        # =================================================================
        #  STATE: DONE
        # =================================================================
        elif self.state == "DONE":
            # #self.gripper_value = GRIPPER_CLOSE
            # fingers = self._get_finger_positions()
            # left, right = fingers
            # f"  gripper is CLOSED cmd func: left: {left}, right: {right} …"
            fingers = self._get_finger_positions()
            # self.get_logger().info(
            #         f"  Gripper confirmed closed (fingers: {fingers})."
            #     )
            if not self.arm_cmd_sent:
                # status_msg = String()
                # status_msg.data = REASON_SUCCESS
                # self.status_pub.publish(status_msg)
                # self.get_logger().info(f"Pick sequence complete – result: {REASON_SUCCESS}")
                # self.arm_cmd_sent = True  # reuse flag to only publish once
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

    # =====================================================================
    #  ARM COMMAND
    # =====================================================================

    # def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
    #   """
    #   Compute joint targets and send ArmCommand.
    #   Uses motion planner service if available, otherwise hardcoded IK.
    #   """
    #   # Adjust target pose with z_offset
    #   target_pose = copy.deepcopy(grasp_pose)
    #   target_pose.position.z += z_offset
    def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
      """
      Compute joint targets and send ArmCommand.
      Uses motion planner service if available, otherwise hardcoded IK.
      """
      # Adjust target pose with z_offset
      target_pose = copy.deepcopy(grasp_pose)
      target_pose.position.z += z_offset
      
      # ──────────────────────────────────────────────────────────────
      # USE MOTION PLANNER SERVICE
      # ──────────────────────────────────────────────────────────────
      if self.use_planner:
          joint_targets = self._call_planner_service(target_pose)
          
          if joint_targets is None:
              # IK failed - abort sequence
              self.get_logger().error(
                  f"IK failed for pose at z={target_pose.position.z:.3f}. "
                  "Aborting grasp sequence."
              )
              self._publish_status(REASON_IK_FAIL)
              self._advance("DONE")
              return
      else:
          # Fallback: Hardcoded IK
          self.get_logger().warn("Using hardcoded IK (planner unavailable)")
          joint_targets = self._hardcoded_ik(target_pose)
      
      # ──────────────────────────────────────────────────────────────
      # PUBLISH ARM COMMAND
      # ──────────────────────────────────────────────────────────────
      self.current_arm_target = joint_targets
      self.arm_cmd_sent_time = time.time()
      
      cmd = ArmCommand()
      cmd.joint_positions = joint_targets
      cmd.duration = MOVE_DURATION
      self.arm_cmd_pub.publish(cmd)
      
      self.get_logger().info(
          f"  ArmCommand: {[f'{j:.2f}' for j in joint_targets]}  "
          f"duration={MOVE_DURATION}s  (z={target_pose.position.z:.3f})"
      )


    def _call_planner_service(self, target_pose: Pose):
      """
      Call the /plan_to_target service to compute IK.
      
      Returns:
          List of 6 joint positions, or None if planning failed
      """
      # Determine which state we're in for logging
      state_label = {
          "MOVE_PREGRASP": "pre-grasp",
          "MOVE_GRASP": "grasp",
          "MOVE_LIFT": "lift"
      }.get(self.state, self.state)
      
      # Create service request
      request = PlanToTarget.Request()
      request.arm_name = self.arm_name
      request.target_pose = target_pose
      request.use_orientation = True      # Use full 6-DOF IK
      request.execute = False             # Plan only, don't execute
      request.duration = MOVE_DURATION
      request.max_condition_number = 100.0
      
      self.get_logger().info(
          f"  Calling planner for {state_label} pose: "
          f"({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, "
          f"{target_pose.position.z:.3f})"
      )
      
      # Call service (blocking with timeout)
      future = self.plan_client.call_async(request)
      
      # Wait for response with timeout
      timeout = 5.0
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
      self.get_logger().info(f"  ✓ IK success: {response.message}")
      if hasattr(response, 'position_error') and response.position_error is not None:
          self.get_logger().info(
              f"    Position error: {response.position_error:.4f}m, "
              f"Orientation error: {response.orientation_error:.4f}rad "
              f"({np.degrees(response.orientation_error):.1f}°)"
          )
      
      return response.joint_positions


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
    
    
    def _publish_status(self, reason: str):
        """Helper to publish task status."""
        status_msg = String()
        status_msg.data = reason
        self.status_pub.publish(status_msg)
        # # ──────────────────────────────────────────────────────────────
        # # USE MOTION PLANNER SERVICE
        # # ──────────────────────────────────────────────────────────────
        # if self.use_planner:
        #     joint_targets = self._call_planner_service(target_pose)
            
        #     if joint_targets is None:
        #         # IK failed - abort sequence
        #         self.get_logger().error(
        #             f"IK failed for pose at z={target_pose.position.z:.3f}. "
        #             "Aborting grasp sequence."
        #         )
        #         self._publish_status(REASON_IK_FAIL)
        #         self._advance("DONE")
        #         return
        # else:
        #     # Fallback: Hardcoded IK
        #     self.get_logger().warn("Using hardcoded IK (planner unavailable)")
        #     joint_targets = self._hardcoded_ik(target_pose)
        
        # # ──────────────────────────────────────────────────────────────
        # # PUBLISH ARM COMMAND
        # # ──────────────────────────────────────────────────────────────
        # self.current_arm_target = joint_targets
        # self.arm_cmd_sent_time = time.time()
        
        # cmd = ArmCommand()
        # cmd.joint_positions = joint_targets
        # cmd.duration = MOVE_DURATION
        # self.arm_cmd_pub.publish(cmd)
        
        # self.get_logger().info(
        #     f"  ArmCommand: {[f'{j:.2f}' for j in joint_targets]}  "
        #     f"duration={MOVE_DURATION}s  (z={target_pose.position.z:.3f})"
        # )
    
    # def _send_arm_to_pose(self, grasp_pose: Pose, z_offset: float = 0.0):
    #     """
    #     Compute joint targets and send ArmCommand.

    #     ────────────────────────────────────────────────────────────────
    #     INTEGRATION POINT:
    #     Replace the hardcoded joint lookup with a service call to
    #     motion_planner_node.py once custom_srvs/PlanToPose exists.
    #     ────────────────────────────────────────────────────────────────
    #     """
    #     z = grasp_pose.position.z + z_offset

    #     ####################################################################
    #     #  HARDCODED JOINT TARGETS — PLACEHOLDER IK                        #
    #     #                                                                   #
    #     #  NOT real IK solutions.  Replace with motion-planner service.    #
    #     #  Joint order: [waist, shoulder, elbow, forearm_roll,              #
    #     #                wrist_angle, wrist_rotate]                         #
    #     ####################################################################
    #     if z > 0.20:
    #         joint_targets = [0.0, -0.5, 0.3, 0.0, -0.8, 0.0]
    #     elif z > 0.10:
    #         joint_targets = [0.0, 0.2, 0.3, 0.0, -0.5, 0.0]
    #     else:
    #         joint_targets = [0.0, 0.5, 0.5, 0.0, -0.2, 0.0]
    #     ####################################################################
    #     #  END HARDCODED JOINT TARGETS                                     #
    #     ####################################################################

    #     # Store target so _arm_at_target() can check arrival
    #     self.current_arm_target = joint_targets

    #     cmd = ArmCommand()
    #     cmd.joint_positions = joint_targets
    #     cmd.duration = MOVE_DURATION
    #     self.arm_cmd_pub.publish(cmd)

    #     self.get_logger().info(
    #         f"  ArmCommand: {[f'{j:.2f}' for j in joint_targets]}  "
    #         f"duration={MOVE_DURATION}s  (z={z:.3f})"
    #     )

    # =====================================================================
    #  CALLBACKS
    # =====================================================================

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = msg

    def _grasp_pose_cb(self, msg: PoseStamped):
        """
        ────────────────────────────────────────────────────────────────
        MAIN ENTRY POINT.  Grasp planner publishes here (or hardcoded
        timer for testing).  Kicks off the state machine.
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
            f"{self.grasp_pose.position.z:.3f})"
        )
        self._advance("OPEN_GRIPPER")

    # =====================================================================
    #  HARDCODED SIM POSE
    # =====================================================================

    def _publish_hardcoded_pose(self):
        """
        ####################################################################
        #  HARDCODED EXAMPLE – FOR SIMULATION / TESTING ONLY               #
        #  Replace / remove once grasp_planner_node.py exists.             #
        ####################################################################
        """
        self.startup_timer.cancel()

        ####################################################################
        #  HARDCODED VALUES                                                #
        ####################################################################
        hardcoded = PoseStamped()
        hardcoded.header.stamp = self.get_clock().now().to_msg()
        hardcoded.header.frame_id = "base_link"

        hardcoded.pose.position.x = -0.10
        hardcoded.pose.position.y = -0.35
        hardcoded.pose.position.z = 0.50

        hardcoded.pose.orientation.w = 0.5
        hardcoded.pose.orientation.x = 0.5
        hardcoded.pose.orientation.y = 0.5
        hardcoded.pose.orientation.z = -0.5
        ####################################################################
        #  END HARDCODED VALUES                                            #
        ####################################################################

        self.get_logger().info("Publishing HARDCODED grasp pose for sim testing …")
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
