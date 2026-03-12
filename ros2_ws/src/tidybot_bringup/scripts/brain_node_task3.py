#!/usr/bin/env python3
"""
TidyBot2 Brain Node — Task 3: Liquid Pouring

Orchestrator for the liquid pouring task:
  "Locate the bottle and pour the liquid."

Differences from Task 1 and 2:
  - Navigation leg: Only 1 (navigate to the bottle).
  - Manipulation: Two grabs in sequence (left arm grabs body, right arm grabs lid, untwists, pours).
  - Uses '/arm_status' and '/task_status' to trigger segmentation via 'segment_task3.py'.

States:
  IDLE                → Waiting for MuJoCo + user command
  WAITING_FOR_COMMAND → Sending 'listen' to speech_node
  LISTENING           → Waiting for JSON speech result
  NAV_TO_BOTTLE       → Navigating to the bottle
  SEGMENT_BODY        → Signal vision to segment bottle body (red cube)
  GRAB_BODY           → Left arm grabs the body
  SEGMENT_LID         → Signal vision to segment bottle lid (yellow cube)
  GRAB_LID_UNTWIST_POUR → Right arm grabs lid, twists, pours
  COMPLETED           → Task done, auto-reset

"""

import json
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String, Int32, Float64MultiArray
from sensor_msgs.msg import JointState
from composition_interfaces.srv import LoadNode, UnloadNode


# ── States ──────────────────────────────────────────────────────────

class BrainState(Enum):
    IDLE                  = auto()   # Waiting for user input to start
    WAITING_FOR_COMMAND   = auto()   # A — tell speech_node to listen
    LISTENING             = auto()   # A — waiting for speech_node JSON result
    NAV_TO_BOTTLE         = auto()   # B — navigate to the bottle
    SEGMENT_BODY          = auto()   # C — trigger body segmentation
    GRAB_BODY             = auto()   # D — grab the body (left arm)
    SEGMENT_LID           = auto()   # E — trigger lid segmentation
    GRAB_LID_UNTWIST_POUR = auto()   # F — untwist and pour (right arm)
    COMPLETED             = auto()


# ── Node ────────────────────────────────────────────────────────────

class BrainNodeTask3(Node):

    def __init__(self):
        super().__init__('brain_node_task3')

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

        self.create_subscription(String, '/brain/speech_result',        self._speech_cb, 10)
        self.create_subscription(String, '/brain/navigation_status',    self._nav_cb,    10)
        self.create_subscription(String, '/brain/manipulation_status',  self._manip_cb,  10)

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
        self.nav_retries = 0
        self.max_nav_retries = 3

        # Service clients to dynamically load/unload vision processing
        self.load_client = self.create_client(LoadNode, '/vision_container/_container/load_node')
        self.unload_client = self.create_client(UnloadNode, '/vision_container/_container/unload_node')
        self.pc_node_id = None

        # Async param client to configure the grasp planner at runtime
        # self.grasp_param_client = rclpy.parameter.AsyncParametersClient(
        #     self, 'simple_grasp_planner'
        # )

        # -- Startup delay then control loop -------------------------
        self.get_logger().info('  Waiting 5s for other nodes to start ...')
        self.create_timer(5.0, self._start_control_loop, callback_group=None)
        self.control_timer = None

        self.get_logger().info('=' * 55)
        self.get_logger().info('  Brain Node Task 3 (Liquid Pouring) ready')
        self.get_logger().info('=' * 55)

    def _start_control_loop(self):
        """Called once after startup delay."""
        if self.control_timer is not None:
            return
        self.get_logger().info('Starting control loop.')
        
        # Initialize task status for task 3
        task_msg = Int32()
        task_msg.data = 1  # 1 = task 3
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

    def _parse_speech_json(self, raw: str):
        """
        Parse JSON from speech result.
        Expected: {"source": "bottle"}
        Returns: source item or None on failure.
        """
        try:
            data = json.loads(raw)
            # Use 'source', 'payload', or 'target' based on what speech node outputs
            # Assuming 'source' based on task_3_SYSTEM_FLOW.md
            item = data.get('source', data.get('payload', data.get('target', ''))).strip().lower()
            if item:
                return item
            self.get_logger().warn(f'Missing item field in JSON: {data}')
            return None
        except (json.JSONDecodeError, AttributeError) as e:
            # Fallback if the speech node just sends the string
            res = raw.strip().lower()
            if res and res != 'error':
                 return res
            self.get_logger().warn(f'Failed to parse speech result: {e} -- raw: "{raw}"')
            return None

    # Grasp type configuration
    def _set_grasp_type(self, grasp_type: str):
        """Asynchronously set grasp_type param on simple_grasp_planner ('top' or 'side')."""
        self.get_logger().info(f"Setting grasp_type to '{grasp_type}' on simple_grasp_planner...")
        # future = self.grasp_param_client.set_parameters([
        #     Parameter('grasp_type', Parameter.Type.STRING, grasp_type)
        # ])
        # future.add_done_callback(
        #     lambda f: self.get_logger().info(
        #         f"grasp_type set to '{grasp_type}' OK"
        #     ) if not f.exception() else self.get_logger().error(
        #         f"Failed to set grasp_type: {f.exception()}"
        #     )
        # )

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

        # --- IDLE: Wait for MuJoCo then user command ----------------
        if self.state == BrainState.IDLE:
            # if not self.mujoco_ready:
            #     if elapsed > 1.0 and int(elapsed) % 5 == 0:
            #          self.get_logger().info('Waiting for MuJoCo simulation to start...')
            #     return

            if self.start_command is None:
                if elapsed > 1.0 and int(elapsed) % 5 == 0:
                     self.get_logger().info('Waiting for start command on /brain/command...')
                return

            # Simulation confirmed running and start command received
            self.get_logger().info(f'[OK] MuJoCo running. Executing command "{self.start_command}"...')
            self._set_camera_pan_tilt(0.0, 0.0)
            self._transition(BrainState.WAITING_FOR_COMMAND)

        # --- A: tell speech_node to listen --------------------------
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
                res = self.speech_result.strip().upper()

                if res != 'ERROR':
                    target = self._parse_speech_json(self.speech_result)

                    if target:
                        self.target_item = target
                        self.get_logger().info(f'[OK] Target: "{self.target_item}"')
                        self.nav_retries = 0
                        self._transition(BrainState.NAV_TO_BOTTLE)
                    else:
                        # Parse failed — retry
                        self.get_logger().warn('Invalid speech result -- Retrying in 5s ...')
                        time.sleep(5.0)
                        self._transition(BrainState.WAITING_FOR_COMMAND)
                else:
                    # ERROR — retry
                    self.get_logger().warn('Speech failed/error -- Retrying automatically in 5s ...')
                    time.sleep(5.0)
                    self._transition(BrainState.WAITING_FOR_COMMAND)

            elif elapsed > 60.0:
                self.get_logger().warn('Speech timeout -- Retrying ...')
                self._transition(BrainState.WAITING_FOR_COMMAND)

        # --- B: Navigate to Bottle ----------------------------------
        elif self.state == BrainState.NAV_TO_BOTTLE:
            if not self.goal_sent:
                self.get_logger().info(f'--- B: NAVIGATE TO "{self.target_item}" ---')
                self._pub(self.nav_goal_pub, f'{self.target_item}')
                self.goal_sent = True

            if self.nav_status == 'arrived':
                self.get_logger().info('  Navigation complete. Settling 1s...')
                time.sleep(1.0)
                self._transition(BrainState.SEGMENT_BODY)
            elif self.nav_status == 'failed':
                self.nav_retries += 1
                if self.nav_retries < self.max_nav_retries:
                    self.get_logger().warn(f'  Nav to bottle failed (attempt {self.nav_retries}/{self.max_nav_retries}). Retrying...')
                    self._transition(BrainState.NAV_TO_BOTTLE)
                else:
                    self.get_logger().error('  Navigation to bottle failed after max retries!')
                    self._transition(BrainState.COMPLETED)

        # --- C: Segment Body (Left Arm) -----------------------------
        elif self.state == BrainState.SEGMENT_BODY:
            if not self.goal_sent:
                self.get_logger().info('--- C: TRIGGER SEGMENTATION FOR BODY (LEFT ARM) ---')
                self._set_grasp_type('side')
                self.start_point_cloud_processing()
                arm_msg = Int32()
                arm_msg.data = 0  # 0 = left arm
                self.arm_status_pub.publish(arm_msg)
                
                task_msg = Int32()
                task_msg.data = 1  # Task 3
                self.task_status_pub.publish(task_msg)
                
                self.goal_sent = True

            # Wait a little for vision node to process and publish bbox to manip node
            if elapsed > 10.0:
                self._transition(BrainState.GRAB_BODY)

        # --- D: Grab Body (Left Arm) --------------------------------
        elif self.state == BrainState.GRAB_BODY:
            if not self.goal_sent:
                self.get_logger().info('--- D: GRAB BODY (LEFT ARM) ---')
                self._pub(self.manip_goal_pub, 'grab')
                self.goal_sent = True

            if self.manip_status == 'done':
                self.get_logger().info('  Grab body complete. Stopping vision. Settling 1s...')
                self.stop_point_cloud_processing()
                time.sleep(1.0)
                self._transition(BrainState.SEGMENT_LID)
            elif self.manip_status == 'failed':
                self.get_logger().error('  Grab body failed!')
                self.stop_point_cloud_processing()
                self._transition(BrainState.COMPLETED)

        # --- E: Segment Lid (Right Arm) -----------------------------
        elif self.state == BrainState.SEGMENT_LID:
            if not self.goal_sent:
                self.get_logger().info('--- E: TRIGGER SEGMENTATION FOR LID (RIGHT ARM) ---')
                self._set_grasp_type('top')
                self.start_point_cloud_processing()
                arm_msg = Int32()
                arm_msg.data = 1  # 1 = right arm
                self.arm_status_pub.publish(arm_msg)
                
                task_msg = Int32()
                task_msg.data = 1  # Task 3
                self.task_status_pub.publish(task_msg)
                
                self.goal_sent = True

            # Wait a little for vision node to process and publish bbox to manip node
            if elapsed > 10.0:
                self._transition(BrainState.GRAB_LID_UNTWIST_POUR)

        # --- F: Grab Lid, Untwist, and Pour (Right Arm) -------------
        elif self.state == BrainState.GRAB_LID_UNTWIST_POUR:
            if not self.goal_sent:
                self.get_logger().info('--- F: GRAB LID AND POUR (RIGHT ARM) ---')
                self._pub(self.manip_goal_pub, 'grab')
                self.goal_sent = True

            if self.manip_status == 'done':
                self.get_logger().info('  Pouring sequence complete. Stopping vision. Settling 1s...')
                self.stop_point_cloud_processing()
                time.sleep(1.0)
                self._transition(BrainState.COMPLETED)
            elif self.manip_status == 'failed':
                self.get_logger().error('  Pouring sequence failed!')
                self.stop_point_cloud_processing()
                self._transition(BrainState.COMPLETED)

        # --- COMPLETED ---------------------------------------------
        elif self.state == BrainState.COMPLETED:
            if not self.goal_sent:
                self.get_logger().info('')
                self.get_logger().info('=' * 55)
                self.get_logger().info(f'  TASK 3 COMPLETE')
                self.get_logger().info('=' * 55)
                self.goal_sent = True

                # Auto reset after delay
                self.get_logger().info('Resetting in 5 seconds ...')

            if elapsed > 5.0:
                self.target_item = None
                self.start_command = None
                self.nav_retries = 0
                self._transition(BrainState.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNodeTask3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
