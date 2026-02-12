#!/usr/bin/env python3
"""
TidyBot2 Brain Node - Orchestrator

Pure orchestrator — only publishes goals, reads statuses.
Handles user interaction (blocking wait) via background thread to keep ROS spinning.

States:
  IDLE                → Waiting for user 'Enter'
  WAITING_FOR_COMMAND → Sending 'listen'
  LISTENING           → Waiting for speech result
  NAVIGATING          → Waiting for navigation
  GRABBING            → Waiting for manipulation
  RETURNING           → Waiting for navigation
  RELEASING           → Waiting for manipulation
  COMPLETED           → Task done, waiting for reset

"""

import time
import threading
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState


# ── States ──────────────────────────────────────────────────────────

class BrainState(Enum):
    IDLE                = auto()   # Waiting for user input to start
    WAITING_FOR_COMMAND = auto()   # A — tell speech_node to listen
    LISTENING           = auto()   # A — waiting for speech_node result
    NAVIGATING          = auto()   # B — waiting for navigation_node
    GRABBING            = auto()   # C — waiting for manipulation_node
    RETURNING           = auto()   # D — waiting for navigation_node
    RELEASING           = auto()   # E — waiting for manipulation_node
    COMPLETED           = auto()


# ── Node ────────────────────────────────────────────────────────────

class BrainNode(Node):

    def __init__(self):
        super().__init__('brain_node')

        # ── Pub/Sub ─────────────────────────────────────────────────
        self.speech_goal_pub = self.create_publisher(String, '/brain/speech_goal',       10)
        self.nav_goal_pub    = self.create_publisher(String, '/brain/navigation_goal',   10)
        self.manip_goal_pub  = self.create_publisher(String, '/brain/manipulation_goal', 10)

        # ── Status subscribers ──────────────────────────────────────
        self.speech_result = None
        self.nav_status    = 'idle'
        self.manip_status  = 'idle'

        self.create_subscription(String, '/brain/speech_result',        self._speech_cb, 10)
        self.create_subscription(String, '/brain/navigation_status',    self._nav_cb,    10)
        self.create_subscription(String, '/brain/manipulation_status',  self._manip_cb,  10)
        
        # Check if MuJoCo is running
        self.mujoco_ready = False
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # ── State machine ───────────────────────────────────────────
        self.state = BrainState.IDLE
        self.target_item: str | None = None
        self.goal_sent = False
        self.state_start_time = time.time()

        # -- Startup delay then control loop -------------------------
        self.get_logger().info('  Waiting 5s for other nodes to start ...')
        self.create_timer(5.0, self._start_control_loop, callback_group=None)
        self.control_timer = None

        self.get_logger().info('=' * 55)
        self.get_logger().info('  Brain Node ready')
        self.get_logger().info('=' * 55)

    def _start_control_loop(self):
        """Called once after startup delay."""
        if self.control_timer is not None:
            return
        self.get_logger().info('Starting control loop.')
        self.state_start_time = time.time()
        self.control_timer = self.create_timer(1.0, self._control_loop)




    # -- Callbacks ---------------------------------------------------

    def _speech_cb(self, msg: String):
        self.speech_result = msg.data

    def _nav_cb(self, msg: String):
        self.nav_status = msg.data

    def _manip_cb(self, msg: String):
        self.manip_status = msg.data

    def _joint_state_cb(self, msg: JointState):
        self.mujoco_ready = True

    # -- Helpers -----------------------------------------------------

    def _transition(self, new_state: BrainState):
        self.get_logger().info(f'State: {self.state.name}  ->  {new_state.name}')
        self.state = new_state
        self.goal_sent = False
        self.state_start_time = time.time()
        self.state_start_time = time.time()
        self.speech_result = None
        self.nav_status = 'idle'
        self.manip_status = 'idle'

    def _pub(self, publisher, data: str):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    # ── Control loop ────────────────────────────────────────────────

    def _control_loop(self):
        elapsed = time.time() - self.state_start_time

        # --- IDLE: Wait for MuJoCo then Auto-start -----------------
        if self.state == BrainState.IDLE:
            if not self.mujoco_ready:
                if elapsed > 1.0 and int(elapsed) % 5 == 0:
                     self.get_logger().info('Waiting for MuJoCo simulation to start...')
                return

            # Simulation confirmed running -> Auto start
            self.get_logger().info('[OK] MuJoCo running. Starting task sequence in 5 seconds...')
            time.sleep(5.0)
            self._transition(BrainState.WAITING_FOR_COMMAND)

        # --- A: tell speech_node to listen -------------------------
        elif self.state == BrainState.WAITING_FOR_COMMAND:
            if not self.goal_sent:
                self.get_logger().info('--- A: WAITING FOR VERBAL COMMAND ---')
                self._pub(self.speech_goal_pub, 'listen')
                self.get_logger().info('  -> speech_node: "listen"')
                self.goal_sent = True
                self._transition(BrainState.LISTENING)

        # ─── A (cont): wait for result ─────────────────────────────
        elif self.state == BrainState.LISTENING:
            if self.speech_result is not None:
                # Check for "ERROR" or "error" (case insensitive)
                res = self.speech_result.strip().upper()
                
                if res != 'ERROR':
                    self.target_item = self.speech_result
                    self.get_logger().info(f'[OK] Target item: "{self.target_item}"')
                    self._transition(BrainState.NAVIGATING)
                else:
                    # RETRY logic as requested
                    self.get_logger().warn('Speech failed/error -- Retrying automatically in 5s ...')
                    time.sleep(5.0)
                    self._transition(BrainState.WAITING_FOR_COMMAND)

            elif elapsed > 60.0:
                self.get_logger().warn('Speech timeout -- Retrying ...')
                self._transition(BrainState.WAITING_FOR_COMMAND)

        # --- B: Navigate -------------------------------------------
        elif self.state == BrainState.NAVIGATING:
            if not self.goal_sent:
                self.get_logger().info(f'--- B: NAVIGATE TO "{self.target_item}" ---')
                self._pub(self.nav_goal_pub, f'find {self.target_item}')
                self.goal_sent = True

            if self.nav_status == 'arrived':
                self.get_logger().info('  Navigation complete. Settling 1s...')
                time.sleep(1.0)
                self._transition(BrainState.GRABBING)
            elif self.nav_status == 'failed':
                self.get_logger().error('  Navigation failed!')
                self._transition(BrainState.COMPLETED)

        # --- C: Grab -----------------------------------------------
        elif self.state == BrainState.GRABBING:
            if not self.goal_sent:
                self.get_logger().info('--- C: GRAB OBJECT ---')
                self._pub(self.manip_goal_pub, 'grab')
                self.goal_sent = True

            if self.manip_status == 'done':
                self.get_logger().info('  Grab complete. Settling 1s...')
                time.sleep(1.0)
                self._transition(BrainState.RETURNING)
            elif self.manip_status == 'failed':
                self.get_logger().error('  Grab failed!')
                self._transition(BrainState.COMPLETED)

        # --- D: Return ---------------------------------------------
        elif self.state == BrainState.RETURNING:
            if not self.goal_sent:
                self.get_logger().info('--- D: RETURN TO START ---')
                self._pub(self.nav_goal_pub, 'return_to_start')
                self.goal_sent = True

            if self.nav_status == 'arrived':
                self.get_logger().info('  Back at start. Settling 1s...')
                time.sleep(1.0)
                self._transition(BrainState.RELEASING)
            elif self.nav_status == 'failed':
                self.get_logger().warn('  Return failed -- releasing anyway.')
                self._transition(BrainState.RELEASING)

        # --- E: Release --------------------------------------------
        elif self.state == BrainState.RELEASING:
            if not self.goal_sent:
                self.get_logger().info('--- E: RELEASE OBJECT ---')
                self._pub(self.manip_goal_pub, 'release')
                self.goal_sent = True

            if self.manip_status == 'done':
                self.get_logger().info('  Object released. Settling 1s...')
                time.sleep(1.0)
                self._transition(BrainState.COMPLETED)

# --- COMPLETED ---------------------------------------------
        elif self.state == BrainState.COMPLETED:
            if not self.goal_sent:
                self.get_logger().info('')
                self.get_logger().info('=' * 55)
                self.get_logger().info(f'  TASK COMPLETE -- "{self.target_item}" delivered')
                self.get_logger().info('=' * 55)
                self.goal_sent = True
                
                # Auto reset after delay
                self.get_logger().info('Resetting in 5 seconds ...')

            if elapsed > 5.0:
                self.target_item = None
                self._transition(BrainState.WAITING_FOR_COMMAND)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
