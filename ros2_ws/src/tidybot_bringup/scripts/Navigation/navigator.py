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
  
Bringup:
ros2 launch tidybot_bringup sim.launch.py \
scene:=scene_pickup.xml \
use_rviz:=true \
show_mujoco_viewer:=true

ros2 run tidybot_bringup navigator.py

ros2 run tidybot_bringup vision_yolo_gemini.py

# 1. Set target to banana
ros2 topic pub --once /vision/target std_msgs/msg/String "{data: 'banana'}"

# 2. Trigger SCAN
ros2 topic pub --once /navigator/command std_msgs/msg/String "{data: 'SCAN'}"
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
import math

from geometry_msgs.msg import Twist, Point, Pose
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     TUNABLE PARAMETERS                         ║
# ║  Start with Kp ~ 1.0 (same ballpark as HW2 P7 gains).         ║
# ║  The pixel-space version is Kp_angular ≈ 1.0 / 320 ≈ 0.003    ║
# ╚══════════════════════════════════════════════════════════════════╝

# --- Camera ---
IMAGE_WIDTH   = 640
IMAGE_HEIGHT  = 480
IMAGE_CENTER_X = IMAGE_WIDTH / 2.0

# --- Visual Servo Gains (step D) ---
# Same concept as HW2 P7 Kp, but in pixel-space:
#   HW2: angular_vel = Kp * (theta_ref - theta_current)       [rad error]
#   Here: angular_vel = KP_ANGULAR * (center_px - object_px)  [pixel error]
KP_ANGULAR = 0.003           # rad/s per pixel of centering error
APPROACH_LINEAR_VEL = 0.5   # m/s max forward speed
CENTERING_DEADZONE = 5       # pixels — reduced to prevent deadlock with alignment threshold
KP_APPROACH_LINEAR = 0.01    # m/s per pixel of vertical error (sharper deceleration)
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
SCAN_ANGULAR_VEL = 0.5        # rad/s while spinning
SCAN_FULL_ROTATION = 2 * math.pi / SCAN_ANGULAR_VEL  # seconds for 360°

# --- Safety Limits ---
MAX_LINEAR_VEL  = 0.25
MAX_ANGULAR_VEL = 1.0

# --- Odometry Return Gains (fallback if no AprilTag) ---
# Same as HW2 P7: v = Kp * distance_error, omega = Kp * angle_error
KP_ODOM_LINEAR  = 0.8
KP_ODOM_ANGULAR = 1.5

# --- States ---
STATE_IDLE        = "IDLE"
STATE_SCANNING    = "SCANNING"
STATE_APPROACHING = "APPROACHING"
STATE_ARRIVED     = "ARRIVED"
STATE_RETURNING   = "RETURNING"


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

        # ══════════ Publishers (same as HW2 P7) ══════════
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub  = self.create_publisher(String, '/navigator/status', 10)

        # ══════════ Subscribers ══════════

        # Commands from Brain/Coordinator
        self.create_subscription(
            String, '/navigator/command',
            self.command_cb, 10)

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

        self.get_logger().info('Navigator ready. Send commands to /navigator/command')
        self.publish_status(STATE_IDLE)

    # ═══════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════

    def command_cb(self, msg: String):
        """Commands from Brain node."""
        cmd = msg.data.strip().upper()
        self.get_logger().info(f'Command: {cmd}')

        if cmd == "SCAN":
            self.state = STATE_SCANNING
            self.scan_start_time = time.time()
            self.scan_accumulated_theta = 0.0
            self.last_scan_theta = self.odom_theta
            
            # Save home pose for odom-based return fallback
            self.start_x = self.odom_x
            self.start_y = self.odom_y
            self.start_theta = self.odom_theta
            self.publish_status(STATE_SCANNING)

        elif cmd == "APPROACH":
            if not self._has_fresh_detection():
                self.get_logger().warn('No detection — run SCAN first')
                return
            self.state = STATE_APPROACHING
            self.publish_status(STATE_APPROACHING)

        elif cmd == "RETURN":
            self.state = STATE_RETURNING
            self.publish_status(STATE_RETURNING)

        elif cmd == "STOP":
            self.send_vel(0.0, 0.0)
            self.state = STATE_IDLE
            self.publish_status(STATE_IDLE)

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
                self.odom_theta = positions[idx_th]
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
            self.get_logger().info(f'Scan: full rotation complete (accumulated {self.scan_accumulated_theta:.2f} rad)')
            self.publish_status("SCAN_COMPLETE")

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
                self.publish_status(STATE_ARRIVED)
                return

        # ── Check stale detection ──
        if not self._has_fresh_detection():
            # Wait instead of spinning. Spinning causes huge oscillation when redetected.
            self.get_logger().warn('Detection lost — stopping and waiting for vision...')
            self.send_vel(0.0, 0.0)
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

        # ── Check if close enough for grasping (Bottom of screen) ──
        # Objects get lower in the image as we get closer.
        # Stop when the center of the object (pixel_y) is near the bottom edge.
        if pixel_y > IMAGE_HEIGHT - 30:
            self.send_vel(0.0, 0.0)
            self.state = STATE_ARRIVED
            self.get_logger().info(
                f'ARRIVED — grasp position reached (y={pixel_y:.0f})')
            self.publish_status(STATE_ARRIVED)
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
        if abs(error_x) > 20:
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
    #  STEP I: RETURN TO START
    #
    #  Strategy:
    #    1. If AprilTag at home is visible → visual servo to it
    #       using PnP pose (tx, tz) from HW2 P6
    #    2. If not visible → use /odom to drive roughly toward
    #       home (HW2 P7 style), then AprilTag for final approach
    # ═══════════════════════════════════════════════════════════

    def do_return(self):
        # ── Phase 1: If we see the home AprilTag, servo to it ──
        if self._has_fresh_apriltag():
            self._return_via_apriltag()
            return

        # ── Phase 2: Odometry-based navigation (HW2 P7 style) ──
        self._return_via_odom()

    def _return_via_apriltag(self):
        """
        Drive toward home AprilTag using PnP pose.
        Same proportional control as HW2 P7:
          angular = Kp * lateral_error
          linear  = Kp * depth
        """
        tx = self.apriltag_pose.position.x   # lateral offset (meters)
        tz = self.apriltag_pose.position.z   # depth (meters)

        if tz < HOME_ARRIVAL_DEPTH:
            self.send_vel(0.0, 0.0)
            self.state = STATE_IDLE
            self.get_logger().info(f'HOME — AprilTag depth = {tz:.2f}m')
            self.publish_status("RETURNED")
            return

        omega = np.clip(-KP_LATERAL * tx, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        v = np.clip(KP_FORWARD * tz, 0.0, MAX_LINEAR_VEL)
        self.send_vel(v, omega)

    def _return_via_odom(self):
        """
        Drive toward saved start pose using odometry.
        Identical to HW2 P7 proportional controller:
          error = reference_pose - current_pose
          v = Kp * distance_error
          omega = Kp * angle_error
        """
        dx = self.start_x - self.odom_x
        dy = self.start_y - self.odom_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.15:
            # Close enough via odom — spin to look for home AprilTag
            self.send_vel(0.0, SCAN_ANGULAR_VEL * 0.5)
            self.get_logger().info('Near home (odom) — searching for AprilTag...')
            return

        target_angle = math.atan2(dy, dx)
        angle_err = self._wrap(target_angle - self.odom_theta)

        if abs(angle_err) > 0.2:
            # Turn first (same logic as HW2 P7 when far from reference)
            omega = np.clip(KP_ODOM_ANGULAR * angle_err,
                            -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
            self.send_vel(0.0, omega)
        else:
            # Drive + correct
            v = np.clip(KP_ODOM_LINEAR * dist, 0.0, MAX_LINEAR_VEL)
            omega = np.clip(KP_ODOM_ANGULAR * angle_err,
                            -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
            self.send_vel(v, omega)

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
