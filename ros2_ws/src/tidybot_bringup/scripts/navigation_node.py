#!/usr/bin/env python3
"""
TidyBot2 Navigation Node (Placeholder)

This node is responsible for ALL base movement.  The brain_node sends
high-level goals here; this node translates them into cmd_vel / target_pose.

Communication with brain_node
─────────────────────────────
  Subscribes:  /brain/navigation_goal   (String)   ← brain tells us what to do
  Publishes:   /brain/navigation_status (String)   → brain reads our status

Goal strings recognised (placeholder set — extend as needed):
  "find <item>"     — placeholder: drive forward, turn left, drive forward
  "return_to_start" — use /base/target_pose to go to (0, 0, 0)
  "stop"            — immediately stop the base

Status strings published:
  "idle"        — waiting for a goal
  "navigating"  — currently executing a goal
  "arrived"     — goal completed successfully
  "failed"      — could not complete goal

Topics published (low-level):
  /cmd_vel          (geometry_msgs/Twist)
  /base/target_pose (geometry_msgs/Pose2D)

Topics subscribed (low-level):
  /base/goal_reached (std_msgs/Bool)
  /odom              (nav_msgs/Odometry)

Usage:
    ros2 run tidybot_bringup navigation_node.py

"""

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


# ── Placeholder parameters ──────────────────────────────────────────
NAV_FORWARD_SPEED  = 0.2   # m/s
NAV_TURN_SPEED     = 0.5   # rad/s
NAV_FORWARD_DIST_1 = 0.5   # metres before turning
NAV_TURN_ANGLE     = math.pi / 2  # 90° left turn
NAV_FORWARD_DIST_2 = 0.3   # metres after turning
RETURN_PROXIMITY   = 0.15  # metres — "close enough" to origin
RETURN_TIMEOUT     = 60.0  # seconds


class NavigationNode(Node):
    """Handles all base movement for TidyBot2."""

    def __init__(self):
        super().__init__('navigation_node')

        # -- Communication with brain --------------------------------
        self.status_pub = self.create_publisher(String, '/brain/navigation_status', 10)
        self.create_subscription(String, '/brain/navigation_goal', self._goal_cb, 10)
        # Speech eavesdropping removed - brain now centralizes control

        # ── Low-level publishers ────────────────────────────────────
        self.cmd_vel_pub     = self.create_publisher(Twist,  '/cmd_vel',          10)
        self.target_pose_pub = self.create_publisher(Pose2D, '/base/target_pose', 10)

        # ── Low-level subscribers ───────────────────────────────────
        self.goal_reached = False
        self.create_subscription(Bool, '/base/goal_reached', self._goal_reached_cb, 10)

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # ── Internal state ──────────────────────────────────────────
        self.current_goal = None        # e.g. "find banana", "return_to_start"
        self.sub_state = 'IDLE'         # IDLE, FORWARD_1, TURN, FORWARD_2, RETURNING
        self.distance_accum = 0.0
        self.angle_accum = 0.0
        self.state_start_time = time.time()

        # ── Control loop (50 Hz) ────────────────────────────────────
        self.dt = 0.02
        self.create_timer(self.dt, self._control_loop)

        self._publish_status('idle')
        self.get_logger().info('Navigation Node ready — waiting for goals OR speech results')

    # ── Callbacks ───────────────────────────────────────────────────


    def _goal_cb(self, msg: String):
        """Receive a new goal from the brain node."""
        goal = msg.data.strip().lower()
        self.get_logger().info(f'Received navigation goal: "{goal}"')

        if goal == 'stop':
            self._stop_base()
            self.current_goal = None
            self.sub_state = 'IDLE'
            self._publish_status('idle')
            return

        self.current_goal = goal
        self.goal_reached = False

        if goal == 'return_to_start':
            self.sub_state = 'RETURNING'
            self.state_start_time = time.time()
            self._publish_status('navigating')
            # Send go-to-goal command
            target = Pose2D()
            target.x = 0.0
            target.y = 0.0
            target.theta = 0.0
            self.target_pose_pub.publish(target)
            self.get_logger().info('  Sent target_pose (0, 0, 0)')

        elif goal.startswith('find '):
            self.sub_state = 'FINDING'
            self.state_start_time = time.time()
            self._publish_status('navigating')
            # Send coordinate for object (1.0, 1.0)
            target = Pose2D()
            target.x = 1.0
            target.y = 1.0
            target.theta = 1.57
            self.target_pose_pub.publish(target)
            self.get_logger().info('  Sent target_pose (1.0, 1.0) for item')

        else:
            self.get_logger().warn(f'Unknown goal: "{goal}"')
            self._publish_status('failed')

    def _goal_reached_cb(self, msg: Bool):
        if msg.data:
            self.goal_reached = True

    def _odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ── Helpers ─────────────────────────────────────────────────────

    def _stop_base(self):
        self.cmd_vel_pub.publish(Twist())

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    # ── Control loop ────────────────────────────────────────────────

    def _control_loop(self):
        elapsed = time.time() - self.state_start_time

        # --- Searching for object ---------------------------------
        if self.sub_state == 'FINDING':
            if self.goal_reached:
                self._stop_base()
                self.get_logger().info('  Arrived at search coordinate. Settling 1s...')
                self.sub_state = 'SETTLING'
                self.state_start_time = time.time()

        elif self.sub_state == 'SETTLING':
            if elapsed > 1.0:
                self.get_logger().info('  Settle complete -- ready for grab.')
                self.sub_state = 'IDLE'
                self.current_goal = None
                self._publish_status('arrived')

        # --- Return to start -----------------------------------------
        elif self.sub_state == 'RETURNING':
            # Use MuJoCo's goal_reached signal for strict synchronization
            if self.goal_reached:
                self._stop_base()
                self.get_logger().info('  Returned to start (Simulation confirmed).')
                self.sub_state = 'IDLE'
                self.current_goal = None
                self._publish_status('arrived')
            elif elapsed > RETURN_TIMEOUT:
                self._stop_base()
                self.get_logger().warn('  Return timeout — giving up.')
                self.sub_state = 'IDLE'
                self.current_goal = None
                self._publish_status('failed')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_base()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
