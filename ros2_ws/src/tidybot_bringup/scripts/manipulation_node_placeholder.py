#!/usr/bin/env python3
"""
TidyBot2 Manipulation Node (Placeholder)

This node is responsible for ALL arm and gripper control.  The brain_node
sends high-level goals here; this node translates them into ArmCommand /
GripperCommand messages.

Communication with brain_node
─────────────────────────────
  Subscribes:  /brain/manipulation_goal   (String)  ← brain tells us what to do
  Publishes:   /brain/manipulation_status (String)  → brain reads our status

Goal strings recognised (placeholder set — extend as needed):
  "grab"    — reach forward, close gripper, retract to safe hold
  "release" — open gripper, return arm to home
  "deposit" — lift arm slightly, open gripper, return arm to home
  "home"    — move arm to home pose (no gripper change)

Status strings published:
  "idle"      — waiting for a goal
  "executing" — currently moving arm / gripper
  "done"      — goal completed successfully
  "failed"    — could not complete goal

Topics published (low-level):
  /right_arm/cmd      (tidybot_msgs/ArmCommand)
  /right_gripper/cmd  (std_msgs/Float64MultiArray)

Usage:
    ros2 run tidybot_bringup manipulation_node.py

"""

import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float64MultiArray, Int32
from sensor_msgs.msg import RegionOfInterest
from tidybot_msgs.msg import ArmCommand


# ── Arm poses ────────────────────────────────────────────────────────
# [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
ARM_HOME      = [0.0,  0.0, 0.0, 0.0,  0.0, 0.0]
ARM_PRE_GRASP = [0.0,  0.4, 0.5, 0.0, -0.3, 0.0]   # reach forward/down
ARM_HOLD_SAFE = [0.0,  0.0, 0.3, 0.0,  0.0, 0.0]   # retracted, object held
ARM_DEPOSIT    = [0.0,  0.2, 0.4, 0.0,  0.0, 0.0]   # slightly higher for bin clearance

ARM_MOVE_DURATION = 2.0   # seconds per arm motion
GRIPPER_WAIT      = 1.5   # seconds to wait for gripper


class ManipulationNode(Node):
    """Handles all arm and gripper actions for TidyBot2."""

    def __init__(self):
        super().__init__('manipulation_node')

        # ── Communication with brain ────────────────────────────────
        self.status_pub = self.create_publisher(String, '/brain/manipulation_status', 10)
        self.create_subscription(String, '/brain/manipulation_goal', self._goal_cb, 10)
        
        # Metadata from brain (for real robot parity)
        self.create_subscription(Int32, '/arm_status', self._arm_status_cb, 10)
        self.create_subscription(Int32, '/task_status', self._task_status_cb, 10)

        # Vision bbox from segment_task3 (Task 3 only)
        self.latest_bbox = None
        self.bbox_received = False
        self.create_subscription(RegionOfInterest, '/vision/bbox', self._bbox_cb, 10)

        # ── Low-level publishers ────────────────────────────────────
        self.arm_pub     = self.create_publisher(ArmCommand,        '/right_arm/cmd',     10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)

        # ── Internal state ──────────────────────────────────────────
        self.current_goal = None
        self.sub_state = 'IDLE'  # IDLE, REACHING, CLOSING, RETRACTING, OPENING, HOMING, LIFTING
        self.state_start_time = time.time()
        
        self.active_arm = 'right'  # 0=left, 1=right
        self.current_task = 0      # 0=task1/2, 1=task3

        # ── Control loop (50 Hz) ────────────────────────────────────
        self.dt = 0.02
        self.create_timer(self.dt, self._control_loop)

        self._publish_status('idle')
        self.get_logger().info('Manipulation Node ready — waiting for goals on /brain/manipulation_goal')

    # ── Callbacks ───────────────────────────────────────────────────

    def _goal_cb(self, msg: String):
        """Receive a new goal from the brain node."""
        goal = msg.data.strip().lower()
        self.get_logger().info(f'Received manipulation goal: "{goal}"')

        self.current_goal = goal
        self.state_start_time = time.time()
        self._publish_status('executing')

        if goal == 'grab' and self.current_task == 1:
            # Task 3: wait for bbox from segment_task3, then grab
            self.sub_state = 'WAITING_BBOX'
            self.get_logger().info('  Task 3 grab: waiting for /vision/bbox ...')

        elif goal == 'grab':
            self.sub_state = 'REACHING'
            self._send_gripper(0.0)              # open gripper first
            self._send_arm(ARM_PRE_GRASP)        # reach to pre-grasp
            self.get_logger().info('  Opening gripper & reaching …')

        elif goal == 'release':
            self.sub_state = 'OPENING'
            self._send_gripper(0.0)              # open gripper
            self.get_logger().info('  Opening gripper …')

        elif goal == 'deposit':
            self.sub_state = 'LIFTING'
            self._send_arm(ARM_DEPOSIT)          # lift before release
            self.get_logger().info('  Lifting arm for deposit …')

        elif goal == 'home':
            self.sub_state = 'HOMING'
            self._send_arm(ARM_HOME)
            self.get_logger().info('  Moving arm to home …')

        else:
            self.get_logger().warn(f'Unknown goal: "{goal}"')
            self.sub_state = 'IDLE'
            self._publish_status('failed')

    def _arm_status_cb(self, msg: Int32):
        """Metadata from brain: 0=left, 1=right."""
        self.active_arm = 'left' if msg.data == 0 else 'right'
        
        # Clear bbox when switching arms/stages in Task 3
        self.bbox_received = False
        self.latest_bbox = None
        
        # Update publishers based on active arm
        self.arm_pub = self.create_publisher(ArmCommand, f'/{self.active_arm}_arm/cmd', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, f'/{self.active_arm}_gripper/cmd', 10)
        self.get_logger().info(f'Placeholder acknowledging active arm: {self.active_arm}')

    def _task_status_cb(self, msg: Int32):
        """Metadata from brain: 0=task1/2, 1=task3."""
        self.current_task = msg.data
        task_label = "Task 3" if msg.data == 1 else "Task 1 or 2"
        self.get_logger().info(f'Placeholder acknowledging task: {task_label}')

    def _bbox_cb(self, msg: RegionOfInterest):
        """Receive a bounding box from segment_task3."""
        self.latest_bbox = msg
        self.bbox_received = True
        self.get_logger().info(
            f'  [BBOX] Received bbox from segment_task3: '
            f'x={msg.x_offset} y={msg.y_offset} w={msg.width} h={msg.height}'
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _send_arm(self, positions, duration=ARM_MOVE_DURATION):
        cmd = ArmCommand()
        cmd.joint_positions = positions
        cmd.duration = duration
        self.arm_pub.publish(cmd)

    def _send_gripper(self, value: float):
        """value: 0.0 = open, 1.0 = closed."""
        msg = Float64MultiArray()
        msg.data = [value]
        self.gripper_pub.publish(msg)

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    # ── Control loop ────────────────────────────────────────────────

    def _control_loop(self):
        elapsed = time.time() - self.state_start_time

        # ── TASK 3: wait for bbox, then grab ────────────────────────
        if self.sub_state == 'WAITING_BBOX':
            if self.bbox_received:
                bbox = self.latest_bbox
                self.get_logger().info(
                    f'  [OK] Bbox received — x={bbox.x_offset} y={bbox.y_offset} '
                    f'w={bbox.width} h={bbox.height}. Executing grab on {self.active_arm} arm...'
                )
                self._send_gripper(0.0)      # open gripper
                self._send_arm(ARM_PRE_GRASP)
                self.sub_state = 'REACHING'
                self.state_start_time = time.time()
            elif elapsed > 12.0:
                # Log periodically but avoid spamming (the control loop runs at 50Hz)
                if not hasattr(self, '_last_wait_log_time') or (time.time() - self._last_wait_log_time) >= 10.0:
                    self.get_logger().info(f'  Still waiting for /vision/bbox... ({int(elapsed)}s)')
                    self._last_wait_log_time = time.time()
            return

        # ── GRAB sequence: reach → close → retract ──────────────────
        elif self.sub_state == 'REACHING':
            if elapsed > ARM_MOVE_DURATION + 0.5:
                self.get_logger().info('  Reached pre-grasp. Closing gripper …')
                self._send_gripper(1.0)
                self.sub_state = 'CLOSING'
                self.state_start_time = time.time()

        elif self.sub_state == 'CLOSING':
            if elapsed > GRIPPER_WAIT:
                self.get_logger().info('  Gripper closed. Retracting to safe pose …')
                self._send_arm(ARM_HOLD_SAFE)
                self.sub_state = 'RETRACTING'
                self.state_start_time = time.time()

        elif self.sub_state == 'RETRACTING':
            if elapsed > ARM_MOVE_DURATION + 0.5:
                self.get_logger().info('  Object secured.')
                self.sub_state = 'IDLE'
                self.current_goal = None
                self._publish_status('done')

        # ── RELEASE sequence: open gripper → home ───────────────────
        elif self.sub_state == 'OPENING':
            if elapsed > GRIPPER_WAIT:
                self.get_logger().info('  Gripper open. Returning to home …')
                self._send_arm(ARM_HOME)
                self.sub_state = 'HOMING'
                self.state_start_time = time.time()

        # ── HOME ────────────────────────────────────────────────────
        elif self.sub_state == 'HOMING':
            if elapsed > ARM_MOVE_DURATION + 0.5:
                self.get_logger().info('  Arm at home.')
                self.sub_state = 'IDLE'
                self.current_goal = None
                self._publish_status('done')

        # ── DEPOSIT sequence: lift → open → home ───────────────────
        elif self.sub_state == 'LIFTING':
            if elapsed > ARM_MOVE_DURATION + 0.5:
                self.get_logger().info('  Lifted. Opening gripper …')
                self._send_gripper(0.0)
                self.sub_state = 'OPENING'
                self.state_start_time = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
