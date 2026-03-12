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
  OBSERVING           → Starting vision, waiting for grasp pose, stopping vision
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
from std_msgs.msg import String, Int32, Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from composition_interfaces.srv import LoadNode, UnloadNode


# ── States ──────────────────────────────────────────────────────────

class BrainState(Enum):
    IDLE                = auto()   # Waiting for user input to start
    WAITING_FOR_COMMAND = auto()   # A — tell speech_node to listen
    LISTENING           = auto()   # A — waiting for speech_node result
    NAVIGATING          = auto()   # B — waiting for navigation_node
    OBSERVING           = auto()   # B.5 — Vision processing
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
        self.arm_status_pub  = self.create_publisher(Int32,  '/arm_status',             10)
        self.task_status_pub = self.create_publisher(Int32,  '/task_status',            10)
        self.pt_cmd_pub      = self.create_publisher(Float64MultiArray, '/camera/pan_tilt_cmd', 10)

        # ── Status subscribers ──────────────────────────────────────
        self.speech_result = None
        self.nav_status    = 'idle'
        self.manip_status  = 'idle'
        self.latest_grasp_pose = None

        self.create_subscription(String, '/brain/speech_result',        self._speech_cb, 10)
        self.create_subscription(String, '/brain/navigation_status',    self._nav_cb,    10)
        self.create_subscription(String, '/brain/manipulation_status',  self._manip_cb,  10)
        self.create_subscription(PoseStamped, '/detected_grasps/pose', self._grasp_cb, 10)
        
        # Check if MuJoCo is running
        self.mujoco_ready = False
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # ── User Commands ───────────────────────────────────────────
        self.start_command = None
        self.create_subscription(String, '/brain/command', self._command_cb, 10)

        # ── State machine ───────────────────────────────────────────
        self.state = BrainState.IDLE
        self.target_item: str | None = None
        self.goal_sent = False
        self.state_start_time = time.time()

        # Service clients to dynamically load/unload vision processing
        self.load_client = self.create_client(LoadNode, '/vision_container/_container/load_node')
        self.unload_client = self.create_client(UnloadNode, '/vision_container/_container/unload_node')
        
        # This will store the ID of the point_cloud_xyz node once it's running
        self.pc_node_id = None

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
        
        # Initialize manipulation status for the real robot node
        arm_msg = Int32()
        arm_msg.data = 1  # 1 = right arm
        self.arm_status_pub.publish(arm_msg)
        
        task_msg = Int32()
        task_msg.data = 0  # 0 = task 1 or 2
        self.task_status_pub.publish(task_msg)

        # Set camera to look at 0 0
        self._set_camera_pan_tilt(0.0, 0.0)

        self.state_start_time = time.time()
        self.control_timer = self.create_timer(1.0, self._control_loop)

    # -- Callbacks ---------------------------------------------------

    def _command_cb(self, msg: String):
        self.start_command = msg.data.strip()
        self.get_logger().info(f'Received user command: {self.start_command}')

    def _speech_cb(self, msg: String):
        self.speech_result = msg.data

    def _nav_cb(self, msg: String):
        self.nav_status = msg.data

    def _manip_cb(self, msg: String):
        self.manip_status = msg.data

    def _joint_state_cb(self, msg: JointState):
        self.mujoco_ready = True

    def _grasp_cb(self, msg: PoseStamped):
        if self.state == BrainState.OBSERVING:
            self.latest_grasp_pose = msg

    # -- Helpers -----------------------------------------------------

    def _transition(self, new_state: BrainState):
        self.get_logger().info(f'State: {self.state.name}  ->  {new_state.name}')
        self.state = new_state
        self.goal_sent = False
        self.state_start_time = time.time()
        self.speech_result = None
        self.nav_status = 'idle'
        self.manip_status = 'idle'

    def _pub(self, publisher, data: str):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    def _set_camera_pan_tilt(self, pan: float, tilt: float):
        msg = Float64MultiArray()
        msg.data = [float(pan), float(tilt)]
        self.pt_cmd_pub.publish(msg)
        self.get_logger().info(f'Setting camera pan-tilt: {pan}, {tilt}')

    # Point cloud load/unload
    def start_point_cloud_processing(self):
        if self.pc_node_id is not None:
            self.get_logger().info("Point cloud processing is already running.")
            return

        if not self.load_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Vision container load service not available! Is the container running?")
            return

        req = LoadNode.Request()
        req.package_name = 'depth_image_proc'
        req.plugin_name = 'depth_image_proc::PointCloudXyzNode'
        req.node_name = 'point_cloud_xyz'
        req.node_namespace = ''
        
        req.remap_rules = [
            'image_rect:=/camera/depth/image_raw',
            'camera_info:=/camera/depth/camera_info',
            'points:=/camera/points'
        ]

        self.get_logger().info("Starting point cloud generation...")
        future = self.load_client.call_async(req)
        future.add_done_callback(self._load_done_callback)

    def _load_done_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.pc_node_id = response.unique_id
                self.get_logger().info(f"Loaded point_cloud_xyz (ID: {self.pc_node_id}). CPU will spike now.")
            else:
                self.get_logger().error(f"Failed to load vision node: {response.error_message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def stop_point_cloud_processing(self):
        if self.pc_node_id is None:
            self.get_logger().warn("Cannot stop: point cloud processing isn't running.")
            return

        if not self.unload_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Vision container unload service not available!")
            return

        req = UnloadNode.Request()
        req.unique_id = self.pc_node_id

        self.get_logger().info("Stopping point cloud generation...")
        future = self.unload_client.call_async(req)
        future.add_done_callback(self._unload_done_callback)

    def _unload_done_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Successfully killed point_cloud_xyz (ID: {self.pc_node_id}). CPU drops to 0%.")
                self.pc_node_id = None
            else:
                self.get_logger().error(f"Failed to unload vision node: {response.error_message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    # ── Control loop ────────────────────────────────────────────────

    def _control_loop(self):
        elapsed = time.time() - self.state_start_time

        # --- IDLE: Wait for MuJoCo then Auto-start -----------------
        if self.state == BrainState.IDLE:
            if not self.mujoco_ready:
                if elapsed > 1.0 and int(elapsed) % 5 == 0:
                     self.get_logger().info('Waiting for MuJoCo simulation to start...')
                return

            if self.start_command is None:
                if elapsed > 1.0 and int(elapsed) % 5 == 0:
                     self.get_logger().info('Waiting for start command on /brain/command...')
                return

            # Simulation confirmed running and start command received
            self.get_logger().info(f'[OK] MuJoCo running. Executing command "{self.start_command}"...')
            self._set_camera_pan_tilt(0.0, 0.0)
            self._transition(BrainState.WAITING_FOR_COMMAND)

        # --- A: tell speech_node to listen -------------------------
        elif self.state == BrainState.WAITING_FOR_COMMAND:
            if not self.goal_sent:
                self.get_logger().info('--- A: WAITING FOR VERBAL COMMAND ---')
                if self.start_command.startswith('test_audio'):
                    self._pub(self.speech_goal_pub, self.start_command)
                    self.get_logger().info(f'  -> speech_node: "{self.start_command}"')
                else:
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
                self._pub(self.nav_goal_pub, f'{self.target_item}')
                self.goal_sent = True

            if self.nav_status == 'arrived':
                self.get_logger().info('  Navigation complete. Settling 1s...')
                time.sleep(1.0)
                self._transition(BrainState.GRABBING)
            elif self.nav_status == 'failed':
                self.get_logger().error('  Navigation failed!')
                self._transition(BrainState.COMPLETED)

        # --- B.5: Observe (Point Cloud processing) -----------------
        elif self.state == BrainState.OBSERVING:
            if not self.goal_sent:
                self.get_logger().info('--- B.5: OBSERVING ITEM ---')
                self.latest_grasp_pose = None
                self.start_point_cloud_processing()
                self.goal_sent = True

            if self.latest_grasp_pose is not None:
                self.get_logger().info('  Grasp pose received! Stopping vision...')
                self.stop_point_cloud_processing()
                self._transition(BrainState.GRABBING)
            elif elapsed > 5.0:
                self.get_logger().warn('  Observation timeout (5s) -- no grasp found. Stopping vision.')
                self.stop_point_cloud_processing()
                self._transition(BrainState.COMPLETED) # Or trigger recovery logic

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
                self.start_command = None
                self._transition(BrainState.IDLE)


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