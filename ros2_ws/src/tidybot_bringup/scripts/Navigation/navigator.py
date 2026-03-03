#!/usr/bin/env python3
"""
navigator.py — ME 326 Navigator Node (Revised)
================================================
Built on top of HW2:
  - P7 structure: Node class, /cmd_vel publisher, /odom subscriber, Kp control
  - P6 pattern:   AprilTag PnP pose for return-to-start and distance checks
  - P3-P4 output: pixel coordinates from Vision API pipeline

Handles steps B, D, E, I from the task plan:
  B  Scan scene (spin in place, stop early if target found)
  D  Plan path (visual servo — proportional control from HW2 P7)
  E  Navigate until "sufficiently close" (bbox + AprilTag PnP depth)
  I  Return to start (AprilTag homing, odometry fallback)

Setup:
  1. cp navigator.py ros2_ws/src/tidybot_bringup/scripts/
  2. chmod +x ros2_ws/src/tidybot_bringup/scripts/navigator.py
  3. Add to CMakeLists.txt under install(PROGRAMS ...)
  4. cd ros2_ws && colcon build --packages-select tidybot_bringup
  5. source setup_env.bash
  6. ros2 run tidybot_bringup navigator.py
  
# 1. Start the simulation
ros2 launch tidybot_bringup sim.launch.py scene:=scene_pickup.xml use_rviz:=true show_mujoco_viewer:=true

# 2. Run the Vision node
ros2 run tidybot_bringup vision_yolo_gemini.py

# 3. Run the Navigator node
ros2 run tidybot_bringup navigator.py

# 4. Trigger the approach (this automatically targets the vision node and starts scanning)
ros2 topic pub --once /brain/navigation_goal std_msgs/msg/String "{data: 'banana'}"
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
import math

from geometry_msgs.msg import Twist, Point, Pose, Pose2D
from std_msgs.msg import String, Bool, Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


# --- Camera ---
IMAGE_WIDTH   = 640
IMAGE_HEIGHT  = 480
IMAGE_CENTER_X = IMAGE_WIDTH / 2.0

# --- Visual Servo Gains (step D) ---
# Same concept as HW2 P7 Kp, but in pixel-space:
#   HW2: angular_vel = Kp * (theta_ref - theta_current)       [rad error]
#   Here: angular_vel = KP_ANGULAR * (center_px - object_px)  [pixel error]
KP_ANGULAR = 0.000625          # rad/s per pixel of centering error IMAGE_WIDTH/2 * 0.000625 = 0.2 rad/s at edge of frame
APPROACH_LINEAR_VEL = 0.5   # m/s max forward speed
CENTERING_DEADZONE = 5       # pixels - Stop turning if we're within this many pixels of center (prevents jitter)
CENTERING_SLOWDOWN_ZONE = 40 # pixels - Doesnt move until within this many pixels of center (face-first alignment
KP_APPROACH_LINEAR = 0.02    # m/s per pixel of vertical error (sharper deceleration)
MIN_APPROACH_LINEAR_VEL = 0.04 # m/s floor to prevent stalling

# --- AprilTag Return-to-Start Gains (step I) ---
# Same Kp structure as HW2 P7, but error is in meters from PnP:
#   angular_vel = -KP_LATERAL * tx   (center on tag laterally)
#   linear_vel  =  KP_FORWARD * tz   (drive toward tag)
KP_LATERAL = 2.0             # rad/s per meter of lateral offset
KP_FORWARD = 0.8             # m/s per meter of depth

# --- "Sufficiently Close" Thresholds (step E) ---
CLOSE_BBOX_AREA_RATIO = 0.15   # bbox_area / image_area (coarse check)
CLOSE_DEPTH_PNP = 0.45         # meters via AprilTag PnP (precise check)
HOME_ARRIVAL_DEPTH = 0.30      # meters — "back at start" via AprilTag

# --- Scanning (step B) ---
SCAN_ANGULAR_VEL = 0.2       # rad/s while spinning
SCAN_FULL_ROTATION = 2 * math.pi / SCAN_ANGULAR_VEL  # seconds for 360°

# --- Safety Limits ---
MAX_LINEAR_VEL  = 0.25
MAX_ANGULAR_VEL = 1.0

# --- Odometry Return Gains (fallback if no AprilTag) ---
# Same as HW2 P7: v = Kp * distance_error, omega = Kp * angle_error
KP_ODOM_LINEAR  = 0.8
KP_ODOM_ANGULAR = 1.5

# --- Fine-centering (after coarse approach) ---
FINE_CENTERING_DEADZONE = 3   # pixels — tighter than coarse CENTERING_DEADZONE (5px)
FINE_CENTERING_HOLD     = 0.5 # seconds centred before advancing

# --- Final approach (odom-based, no YOLO) ---
FINAL_APPROACH_DIST = 0.6   # metres to drive after camera reset
FINAL_APPROACH_VEL  = 0.1   # m/s (slow — we are very close)

# --- States ---
STATE_IDLE           = "IDLE"
STATE_SCANNING       = "SCANNING"
STATE_APPROACHING    = "APPROACHING"
STATE_FINE_CENTERING = "FINE_CENTERING"
STATE_CAMERA_RESET   = "CAMERA_RESET"
STATE_FINAL_APPROACH = "FINAL_APPROACH"
STATE_ARRIVED        = "ARRIVED"
STATE_RETURNING      = "RETURNING"


class Navigator(Node):
    """
    Navigator node — drives the TidyBot2 base.

    Structure mirrors HW2 P7 TrajectoryTracker:
      - Publisher:  /cmd_vel (Twist)
      - Subscriber: /odom (Odometry)
      - Timer:      control loop at 10 Hz
      - Controller: proportional (Kp * error)

    New additions beyond P7:
      - /object_detection subscriber (from Vision node, HW2 P3-P4 output)
      - /apriltag_pose subscriber (from Vision node, HW2 P6 solvePnP)
      - State machine (IDLE → SCAN → APPROACH → ARRIVED → RETURN)
    """

    def __init__(self):
        super().__init__('navigator')
        self.get_logger().info('Navigator node starting...')

        # ── State machine ──
        self.state = STATE_IDLE
        self.scan_start_time = None

        # ── Object detection (from Vision node — HW2 P3/P4 output) ──
        self.latest_detection = None       # Point(x=px, y=py, z=area)
        self.detection_stamp = 0.0
        self.detection_timeout = 3.0       # seconds — longer to survive manual testing gaps

        # ── AprilTag pose (from Vision node — HW2 P6 solvePnP output) ──
        self.apriltag_pose = None          # Pose with position = tvec
        self.apriltag_stamp = 0.0
        self.apriltag_timeout = 1.0

        # ── Odometry (same as HW2 P7) ──
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0

        # ── Fine-centering state ──
        self._fine_center_start = None  # time when we first became centred

        # ── Camera-reset state ──
        self._cam_reset_phase     = 0    # 0=tilt-down, 1=wait, 2=tilt-up, 3=wait2
        self._cam_reset_phase_t   = 0.0

        # ── Final-approach (odom) state ──
        self._final_start_x = None
        self._final_start_y = None

        # ══════════ Publishers (same as HW2 P7) ══════════
        self.cmd_vel_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub     = self.create_publisher(String, '/brain/navigation_status', 10)
        self.target_pub     = self.create_publisher(String, '/vision/target', 10)
        self.target_pose_pub = self.create_publisher(Pose2D, '/base/target_pose', 10)
        self.pan_tilt_pub   = self.create_publisher(Float64MultiArray, '/camera/pan_tilt_cmd', 10)

        # ══════════ Subscribers ══════════

        # Commands from Brain/Coordinator
        self.create_subscription(
            String, '/brain/navigation_goal',
            self.command_cb, 10)
            
        # Target Pose reached confirmation from base controller (sim or real)
        self.goal_reached_flag = False
        self.create_subscription(
            Bool, '/base/goal_reached',
            self.goal_reached_cb, 10)

        # Odometry — from /odom on real robot (may not exist in sim)
        self.create_subscription(
            Odometry, '/odom',
            self.odom_cb, 10)

        # Joint states — extract base pose (joint_x, joint_y, joint_th)
        # This is the primary odometry source in MuJoCo simulation
        self.create_subscription(
            JointState, '/joint_states',
            self.joint_states_cb, 10)

        # Object detection from Vision node (HW2 P3-P4 pipeline output)
        # Convention: x=pixel_x, y=pixel_y, z=bbox_area
        self.create_subscription(
            Point, '/object_detection',
            self.detection_cb, 10)

        # AprilTag pose from Vision node (HW2 P6 solvePnP output)
        # position.z = depth to tag in meters
        self.create_subscription(
            Pose, '/apriltag_pose',
            self.apriltag_cb, 10)

        # ══════════ Control loop (10 Hz, same rate as HW2 P7) ══════════
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Navigator ready. Waiting for Brain navigation goals on /brain/navigation_goal')
        self.publish_status('idle')

    # ═══════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════

    def command_cb(self, msg: String):
        """Commands from Brain node."""
        cmd = msg.data.strip()
        self.get_logger().info(f'Command: {cmd}')

        cmd_upper = cmd.upper()

        if cmd_upper == "STOP":
            self.send_vel(0.0, 0.0)
            self.state = STATE_IDLE
            self.publish_status('idle')

        elif cmd_upper == "RETURN_TO_START":
            self.state = STATE_RETURNING
            self.goal_reached_flag = False
            self.publish_status('navigating')
            
            # Use the robust position controller from the base node 
            target = Pose2D()
            target.x = 0.0
            target.y = 0.0
            target.theta = 0.0
            self.target_pose_pub.publish(target)
            self.get_logger().info('Sent target_pose (0.0, 0.0, 0.0) for return')

        else:
            # We treat any other string as an item name to find
            self.get_logger().info(f'Received item target: {cmd}')
            
            # Publish to vision node
            target_msg = String()
            target_msg.data = cmd
            self.target_pub.publish(target_msg)
            
            # Auto-start scan
            self.state = STATE_SCANNING
            self.scan_start_time = time.time()
            self.scan_accumulated_theta = 0.0
            self.last_scan_theta = self.odom_theta
            
            # Save home pose for odom-based return fallback
            self.start_x = self.odom_x
            self.start_y = self.odom_y
            self.start_theta = self.odom_theta
            self.publish_status('navigating')

    def odom_cb(self, msg: Odometry):
        """
        Odometry callback — identical pattern to HW2 P7.

        Extracts (x, y, theta) from the Odometry message.
        On the real robot and in MuJoCo sim, /odom is published
        by the base driver.
        """
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        # Extract yaw from quaternion (same math you'd use in HW2 P7)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_theta = math.atan2(siny_cosp, cosy_cosp)

    def goal_reached_cb(self, msg: Bool):
        if msg.data:
            self.goal_reached_flag = True

    def joint_states_cb(self, msg: JointState):
        """
        Extract base pose from /joint_states (MuJoCo simulation).

        The MuJoCo bridge publishes joint_x, joint_y, joint_th in
        /joint_states but does NOT publish /odom. This callback
        provides odometry in simulation.
        """
        try:
            names = list(msg.name)
            positions = list(msg.position)
            if 'joint_x' in names and 'joint_y' in names and 'joint_th' in names:
                idx_x = names.index('joint_x')
                idx_y = names.index('joint_y')
                idx_th = names.index('joint_th')
                self.odom_x = positions[idx_x]
                self.odom_y = positions[idx_y]
            
                # The base_th published to /odom receives a +pi/2 offset inside mujoco_bridge_node.py.
                self.odom_theta = self._wrap(positions[idx_th] + math.pi/2)
        except (ValueError, IndexError):
            pass

    def detection_cb(self, msg: Point):
        """Object detection from Vision node (HW2 P3-P4 pipeline)."""
        self.latest_detection = msg
        self.detection_stamp = time.time()

    def apriltag_cb(self, msg: Pose):
        """AprilTag PnP pose from Vision node (HW2 P6 solvePnP)."""
        self.apriltag_pose = msg
        self.apriltag_stamp = time.time()

    # ═══════════════════════════════════════════════════════════
    #  CONTROL LOOP (10 Hz)
    # ═══════════════════════════════════════════════════════════

    def control_loop(self):
        if self.state == STATE_SCANNING:
            self.do_scan()
        elif self.state == STATE_APPROACHING:
            self.do_approach()
        elif self.state == STATE_FINE_CENTERING:
            self.do_fine_centering()
        elif self.state == STATE_CAMERA_RESET:
            self.do_camera_reset()
        elif self.state == STATE_FINAL_APPROACH:
            self.do_final_approach()
        elif self.state == STATE_RETURNING:
            self.do_return()
        # IDLE and ARRIVED: do nothing, wait for Brain

    # ═══════════════════════════════════════════════════════════
    #  STEP B: SCAN (spin, stop early if target found)
    # ═══════════════════════════════════════════════════════════

    def do_scan(self):
        # Track rotation via odometry diff (handles wrap-around)
        diff = self._wrap(self.odom_theta - self.last_scan_theta)
        self.scan_accumulated_theta += abs(diff)
        self.last_scan_theta = self.odom_theta

        # Log progress every ~1 second (every 10 ticks at 10Hz)
        if not hasattr(self, '_scan_log_count'):
            self._scan_log_count = 0
        self._scan_log_count += 1
        if self._scan_log_count % 10 == 0:
            elapsed = time.time() - self.scan_start_time
            self.get_logger().info(
                f'Scan: {self.scan_accumulated_theta:.2f}/{2*math.pi:.2f} rad, '
                f'odom_theta={self.odom_theta:.3f}, elapsed={elapsed:.1f}s, '
                f'has_detection={self._has_fresh_detection()}')

        # If Vision found the target mid-scan, auto-transition to APPROACH
        if self._has_fresh_detection():
            self.send_vel(0.0, 0.0)
            self.state = STATE_APPROACHING
            self._scan_log_count = 0
            self.get_logger().info('Scan: target detected — transitioning to APPROACH')
            self.publish_status(STATE_APPROACHING)
            return

        # Otherwise keep spinning until a full 360 (2*pi) is reached
        if self.scan_accumulated_theta < (2 * math.pi - 0.1):
            self.send_vel(0.0, SCAN_ANGULAR_VEL)
        else:
            self.send_vel(0.0, 0.0)
            self.state = STATE_IDLE
            self._scan_log_count = 0
            self.get_logger().info(f'Scan: full rotation complete but target not found.')
            self.publish_status('failed')

    # ═══════════════════════════════════════════════════════════
    #  STEPS D+E: VISUAL SERVO APPROACH
    #
    #  Same Kp structure as HW2 P7, but:
    #    reference = image center (320 px)
    #    measurement = object pixel_x from Vision node
    #    error = reference - measurement
    #    control = Kp * error → angular velocity
    # ═══════════════════════════════════════════════════════════

    def do_approach(self):
        # ── Check "sufficiently close" via AprilTag PnP (HW2 P6) ──
        if self._has_fresh_apriltag():
            depth = self.apriltag_pose.position.z
            if depth < CLOSE_DEPTH_PNP:
                self.send_vel(0.0, 0.0)
                self.state = STATE_ARRIVED
                self.get_logger().info(
                    f'ARRIVED — AprilTag depth = {depth:.2f}m < {CLOSE_DEPTH_PNP}m')
                self.publish_status('arrived')
                return

        # ── Check stale detection (Object Lost) ──
        if not self._has_fresh_detection():
            self.get_logger().warn('Detection lost! Reverting to SCANNING to find it again...')
            self.send_vel(0.0, 0.0)
            self.state = STATE_SCANNING
            self.scan_start_time = time.time()
            self.scan_accumulated_theta = 0.0
            self.last_scan_theta = self.odom_theta
            self.publish_status('navigating (resuming scan)')
            return

        # ── Visual Servoing with Lag Protection ──
        # If the frame is older than 0.5s, stop to prevent massive overshoot (shimmying)
        age = time.time() - self.detection_stamp
        if age > 0.5:
            self.send_vel(0.0, 0.0)
            return

        det = self.latest_detection
        pixel_x   = det.x
        pixel_y   = det.y
        bbox_area = det.z

        # ── Check if close enough — banana at bottom of frame ──
        # Transition to fine-centering rather than immediately declaring arrived.
        if pixel_y > IMAGE_HEIGHT - 50:
            self.send_vel(0.0, 0.0)
            self.state = STATE_FINE_CENTERING
            self._fine_center_start = None
            self.get_logger().info(
                f'Banana at bottom (y={pixel_y:.0f}) — transitioning to FINE_CENTERING')
            self.publish_status('fine_centering')
            return

        # ── Proportional control (HW2 P7 pattern) ──
        # error in pixels
        error_x = IMAGE_CENTER_X - pixel_x

        # Angular: Kp * error (same as HW2 P7, different units)
        omega = KP_ANGULAR * error_x
        if abs(error_x) < CENTERING_DEADZONE:
            omega = 0.0

        # ── FACE-FIRST ALIGNMENT (User request) ──
        # If centering error is large, don't move forward yet.
        # This prevents the "arc" motion and ensure a direct "b-line".
        # Once reasonably centered, drive while continuing to adjust.
        if abs(error_x) > CENTERING_SLOWDOWN_ZONE:
            v = 0.0
            
            # Throttle the logging to avoid spamming at 10Hz
            import time as _t
            _now = _t.time()
            if _now - getattr(self, '_last_align_log', 0) >= 0.5:
                self._last_align_log = _now
                self.get_logger().info(f'APPROACH: face-first aligning (offset {error_x:.1f}px)')
        else:
            # ── Proportional Slowdown (User request) ──
            # v = KP_APPROACH_LINEAR * (target_y - current_y)
            target_y = IMAGE_HEIGHT - 30
            error_y = target_y - pixel_y
            
            # v scales from APPROACH_LINEAR_VEL down to MIN_APPROACH_LINEAR_VEL
            v = KP_APPROACH_LINEAR * error_y
            v = max(MIN_APPROACH_LINEAR_VEL, min(v, APPROACH_LINEAR_VEL))

        self.send_vel(v, omega)

    # ═══════════════════════════════════════════════════════════
    #  FINE CENTERING — precise lateral alignment before camera reset
    # ═══════════════════════════════════════════════════════════

    def do_fine_centering(self):
        """Hold centred for FINE_CENTERING_HOLD seconds, then move to CAMERA_RESET."""
        if not self._has_fresh_detection():
            # Lost the target — go back to scanning
            self.send_vel(0.0, 0.0)
            self.state = STATE_SCANNING
            self.scan_start_time = time.time()
            self.scan_accumulated_theta = 0.0
            self.last_scan_theta = self.odom_theta
            self._fine_center_start = None
            self.get_logger().warn('Fine-center: detection lost, reverting to SCAN')
            return

        age = time.time() - self.detection_stamp
        if age > 0.5:
            self.send_vel(0.0, 0.0)
            return

        det = self.latest_detection
        error_x = IMAGE_CENTER_X - det.x

        omega = KP_ANGULAR * error_x
        if abs(error_x) < FINE_CENTERING_DEADZONE:
            omega = 0.0

        # Log diagnostic info every ~0.5s
        now = time.time()
        if now - getattr(self, '_last_fine_log', 0) >= 0.5:
            self._last_fine_log = now
            hold_time = (now - self._fine_center_start) if self._fine_center_start else 0.0
            self.get_logger().info(
                f'Fine-center: err_x={error_x:.1f}px, omega={omega:.3f}, '
                f'within_deadzone={abs(error_x) < FINE_CENTERING_DEADZONE}, '
                f'hold={hold_time:.2f}/{FINE_CENTERING_HOLD}s'
            )

        self.send_vel(0.0, omega)

        # Track how long we've stayed centred
        if abs(error_x) < FINE_CENTERING_DEADZONE:
            if self._fine_center_start is None:
                self._fine_center_start = time.time()
            elif time.time() - self._fine_center_start >= FINE_CENTERING_HOLD:
                self.send_vel(0.0, 0.0)
                self.state = STATE_CAMERA_RESET
                self._cam_reset_phase   = 0
                self._cam_reset_phase_t = time.time()
                self.get_logger().info('Fine-centering done — transitioning to CAMERA_RESET')
                self.publish_status('camera_reset')
        else:
            self._fine_center_start = None  # reset hold timer if we drift

    # ═══════════════════════════════════════════════════════════
    #  CAMERA RESET — tilt down then up so the banana is visible
    #  at close range (sim camera freaks out on direct downward tilt)
    # ═══════════════════════════════════════════════════════════

    def _pub_pan_tilt(self, pan: float, tilt: float):
        msg = Float64MultiArray()
        msg.data = [pan, tilt]
        self.pan_tilt_pub.publish(msg)

    def do_camera_reset(self):
        now = time.time()
        elapsed = now - self._cam_reset_phase_t

        if self._cam_reset_phase == 0:
            # Send tilt down
            self._pub_pan_tilt(0.0, -0.6)
            self.get_logger().info('Camera: tilting up (-0.6)')
            self._cam_reset_phase = 1
            self._cam_reset_phase_t = now

        elif self._cam_reset_phase == 1 and elapsed >= 1.2:
            # Send tilt up
            self._pub_pan_tilt(0.0, 0.6)
            self.get_logger().info('Camera: tilting down (0.6)')
            self._cam_reset_phase = 2
            self._cam_reset_phase_t = now

        elif self._cam_reset_phase == 2 and elapsed >= 1.2:
            # Camera settled — kick off final approach
            self._final_start_x = self.odom_x
            self._final_start_y = self.odom_y
            self.state = STATE_FINAL_APPROACH
            self.get_logger().info(
                f'Camera reset done — FINAL_APPROACH from '
                f'({self._final_start_x:.3f}, {self._final_start_y:.3f})')
            self.publish_status('final_approach')

    # ═══════════════════════════════════════════════════════════
    #  FINAL APPROACH — odom-based 0.7 m drive (no YOLO needed)
    # ═══════════════════════════════════════════════════════════

    def do_final_approach(self):
        dist = math.sqrt(
            (self.odom_x - self._final_start_x) ** 2 +
            (self.odom_y - self._final_start_y) ** 2
        )

        if dist < FINAL_APPROACH_DIST:
            self.send_vel(FINAL_APPROACH_VEL, 0.0)
            # Log every ~1 s
            now = time.time()
            if now - getattr(self, '_fa_log_t', 0) >= 1.0:
                self._fa_log_t = now
                self.get_logger().info(
                    f'Final approach: {dist:.3f}/{FINAL_APPROACH_DIST}m')
        else:
            self.send_vel(0.0, 0.0)
            self.state = STATE_ARRIVED
            self.get_logger().info(
                f'ARRIVED — final approach complete ({dist:.3f}m)')
            self.publish_status('arrived')

    # ═══════════════════════════════════════════════════════════
    #  STEP I: RETURN TO START
    #
    #  Strategy:
    #    1. If AprilTag at home is visible → visual servo to it
    #       using PnP pose (tx, tz) from HW2 P6
    #    2. If not visible → use /odom to drive roughly toward
    #       home (HW2 P7 style), then AprilTag for final approach
    # ═══════════════════════════════════════════════════════════

    def do_return(self):
        # We rely on the /base/target_pose command sent in command_cb.
        # mujoco_bridge_node.py (and phoenix6_base_node) handles the continuous control internally
        # and publishes True to /base/goal_reached when it finishes.
        
        # NOTE: Any calls to self.send_vel() here would CANCEL the base node's target_pose! 
        # So we just passively wait for the goal_reached flag.
        
        if self.goal_reached_flag:
            self.state = STATE_IDLE
            self.publish_status('arrived')
            self.get_logger().info('ARRIVED HOME (Target Pose Reached)')

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════

    def send_vel(self, linear: float, angular: float):
        """Publish Twist to /cmd_vel with safety clipping."""
        cmd = Twist()
        cmd.linear.x = float(np.clip(linear, -MAX_LINEAR_VEL, MAX_LINEAR_VEL))
        cmd.angular.z = float(np.clip(angular, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))
        self.cmd_vel_pub.publish(cmd)

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _has_fresh_detection(self) -> bool:
        if self.latest_detection is None:
            return False
        return (time.time() - self.detection_stamp) < self.detection_timeout

    def _has_fresh_apriltag(self) -> bool:
        if self.apriltag_pose is None:
            return False
        return (time.time() - self.apriltag_stamp) < self.apriltag_timeout

    @staticmethod
    def _wrap(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.send_vel(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
