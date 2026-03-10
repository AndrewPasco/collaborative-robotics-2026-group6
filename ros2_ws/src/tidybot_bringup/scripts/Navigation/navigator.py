#!/usr/bin/env python3
"""
navigator.py — ME 326 Navigator Node 

  
# 1. Start the simulation
ros2 launch tidybot_bringup sim.launch.py scene:=scene_task2.xml use_rviz:=true show_mujoco_viewer:=true

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


# --- Camera Parameters ---
IMAGE_WIDTH    = 640
IMAGE_HEIGHT   = 480
IMAGE_CENTER_X = IMAGE_WIDTH / 2.0

# --- Velocity Limits ---
MAX_LINEAR_VEL          = 0.25 # m/s
MAX_ANGULAR_VEL         = 1.0  # rad/s
MIN_APPROACH_LINEAR_VEL = 0.04 # m/s floor to prevent stalling

# --- Navigation Tuning ---
KP_ANGULAR         = 0.001 # rad/s per pixel of x-error (e.g. 0.32 rad/s at edge of frame)
KP_APPROACH_LINEAR = 0.02  # m/s per pixel of y-error (sharper deceleration)
    
# --- Scanning ---
SCAN_ANGULAR_VEL   = 0.2  # rad/s while spinning
SCAN_FULL_ROTATION = 2 * math.pi / SCAN_ANGULAR_VEL  # seconds for 360°

# --- Coarse Centering & Approach ---
APPROACH_LINEAR_VEL     = 0.2  # m/s max forward speed
CENTERING_SLOWDOWN_ZONE = 45   # pixels - Wait to align if error_x > this threshold
CENTERING_DEADZONE      = 25   # pixels - Stop turning if |error_x| < this (prevents jitter)
CENTERING_MIN_OMEGA     = 0.15 # rad/s - Physical limit for turning in place
CENTERING_HOLD_TIME     = 2.0  # seconds - Stable centering before advancing
TARGET_Y_OFFSET         = 50   # pixels - Distance from bottom of frame to stop

# Overshoot recovery — if object is lost during approach, briefly counter-rotate
# before giving up and re-scanning (prevents false full-scan on a minor overshoot)
OVERSHOOT_RECOVER_VEL   = -0.15 # rad/s (counter-rotate) - Physical limit
OVERSHOOT_RECOVER_DUR   = 1.5   # seconds

# --- Fine Centering ---
FINE_CENTERING_DEADZONE = 15  # pixels - Tighter deadzone for final alignment
FINE_CENTERING_HOLD     = 2.0 # seconds - Must stay within deadzone before advancing

# --- Final Approach ---
FINAL_APPROACH_DIST = 0.5 # metres to drive blind after camera reset
FINAL_APPROACH_VEL  = 0.1 # m/s (slow final approach)

# --- State Machine States ---
STATE_IDLE           = "IDLE"
STATE_SCANNING       = "SCANNING"
STATE_APPROACHING    = "APPROACHING"
STATE_FINE_CENTERING = "FINE_CENTERING"
STATE_CAMERA_RESET   = "CAMERA_RESET"
STATE_FINAL_APPROACH = "FINAL_APPROACH"
STATE_ARRIVED        = "ARRIVED"
STATE_RETURNING      = "RETURNING"
STATE_PAUSING        = "PAUSING"


class Navigator(Node):
    """
    Navigator node — drives the TidyBot2 base.

    Structure:
      - Publisher:  /cmd_vel (Twist)
      - Subscriber: /odom (Odometry)
      - Timer:      control loop at 10 Hz
      - Controller: proportional (Kp * error)

    Key additions:
      - /object_detection subscriber (from Vision node)
      - State machine (IDLE → SCAN → APPROACH → ARRIVED → RETURN)
    """

    def __init__(self):
        super().__init__('navigator')
        self.get_logger().info('Navigator node starting...')

        # ── State machine ──
        self.state = STATE_IDLE
        self.scan_start_time = None

        # ── Object detection (from Vision node) ──
        self.latest_detection = None       # Point(x=px, y=py, z=area)
        self.detection_stamp = 0.0
        self.detection_timeout = 5.0       # seconds — longer to survive manual testing gaps

        # ── Odometry ──
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0

        # ── Fine-centering state ──
        self._fine_center_start = None  # time when we first became centred
        self._align_center_start_time = None # time when we first became aligned in Phase 0

        # ── Pause state ──
        self._pause_end_time    = 0.0
        self._pause_next_state  = STATE_IDLE
        self._pause_last_countdown = 0

        # ── Approach phase (0=aligning, 1=advancing) ──
        self._approach_phase = 0
        # Last commanded turn direction (+1=CCW/left, -1=CW/right)
        # Used to counter-rotate briefly if the object is lost due to overshoot
        self._last_turn_sign = 0
        self._overshoot_recover_end = 0.0  # wall-clock time when recovery finishes

        # ── Camera-reset state ──
        self._cam_reset_phase     = 0    # 0=tilt-down, 1=wait, 2=tilt-up, 3=wait2
        self._cam_reset_phase_t   = 0.0

        # ── Final-approach (odom) state ──
        self._final_start_x = None
        self._final_start_y = None

        # ══════════ Publishers ══════════
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

        # Object detection from Vision node
        # Convention: x=pixel_x, y=pixel_y, z=bbox_area
        self.create_subscription(
            Point, '/object_detection',
            self.detection_cb, 10)

        # ══════════ Control loop (10 Hz) ══════════
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
            
            # Reset camera for return
            self._pub_pan_tilt(0.0, 0.0)
            
            # Use the robust position controller from the base node 
            target = Pose2D()
            target.x = self.start_x
            target.y = self.start_y
            target.theta = self.start_theta
            self.target_pose_pub.publish(target)
            self.get_logger().info('Sent target_pose (0.0, 0.0, 0.0) for return')

        else:
            # We treat any other string as an item name to find
            self.get_logger().info(f'Received item target: {cmd}')
            
            # Reset camera to look for the new item
            self._pub_pan_tilt(0.0, 0.0)
            
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
            self.get_logger().info(f'Saved start pose for odom return fallback: x={self.start_x:.2f}, y={self.start_y:.2f}, theta={math.degrees(self.start_theta):.1f}°')
            self.publish_status('navigating')

    def odom_cb(self, msg: Odometry):
        """
        Odometry callback.

        Extracts (x, y, theta) from the Odometry message.
        On the real robot and in MuJoCo sim, /odom is published
        by the base driver.
        """
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        # Extract yaw from quaternion
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
        """Object detection from Vision node."""
        self.latest_detection = msg
        self.detection_stamp = time.time()

    # ═══════════════════════════════════════════════════════════
    #  CONTROL LOOP (10 Hz)
    # ═══════════════════════════════════════════════════════════

    def control_loop(self):
        if self.state == STATE_SCANNING:
            self.do_scan()
        elif self.state == STATE_PAUSING:
            self.do_pause()
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
        # IDLE, ARRIVED: do nothing, wait for Brain

    # ═══════════════════════════════════════════════════════════
    #  SCAN (spin, stop early if target found)
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
    #  VISUAL SERVO APPROACH
    #
    #  Proportional controller structure:
    #    reference = image center (320 px)
    #    measurement = object pixel_x from Vision node
    #    error = reference - measurement
    #    control = Kp * error → angular velocity
    # ═══════════════════════════════════════════════════════════

    def do_approach(self):
        """
        Two-phase visual-servo approach.

        Phase 0 (ALIGNING):
          Rotate in place (no forward motion) using sign-preserving min-omega
          of 0.15 rad/s until |error_x| <= CENTERING_SLOWDOWN_ZONE.
          Then pause 5 s and enter Phase 1.

        Phase 1 (ADVANCING):
          Drive forward (no rotation) using proportional slowdown until the
          object reaches the bottom of the frame.  Then pause 5 s and enter
          FINE_CENTERING.
        """
        # ── Check stale detection (Object Lost) ──
        if not self._has_fresh_detection():
            # Brief counter-rotate before committing to a full 360° re-scan.
            # If we were just turning and the object vanished, we likely overshot it.
            now = time.time()
            if self._last_turn_sign != 0 and now < self._overshoot_recover_end:
                # Still within the recovery window — counter-rotate
                self.send_vel(0.0, OVERSHOOT_RECOVER_VEL * self._last_turn_sign)
                return
            # Recovery window exhausted or no prior turn — fall back to scan
            self.get_logger().warn('Detection lost during approach — reverting to SCAN')
            self.send_vel(0.0, 0.0)
            self.state = STATE_SCANNING
            self.scan_start_time = time.time()
            self.scan_accumulated_theta = 0.0
            self.last_scan_theta = self.odom_theta
            self._last_turn_sign = 0
            self._approach_phase = 0
            self.publish_status('navigating (resuming scan)')
            return

        # ── Lag protection — if frame is stale, hold still ──
        age = time.time() - self.detection_stamp
        if age > 0.5:
            self.send_vel(0.0, 0.0)
            return

        det     = self.latest_detection
        pixel_x = det.x
        pixel_y = det.y
        error_x = IMAGE_CENTER_X - pixel_x
        omega_raw = KP_ANGULAR * error_x

        # ─────────────────────────────────────────────
        #  PHASE 0: ALIGNING — rotate until centred
        # ─────────────────────────────────────────────
        if self._approach_phase == 0:
            if abs(error_x) <= CENTERING_SLOWDOWN_ZONE:
                # Started being centered?
                if self._align_center_start_time is None:
                    self._align_center_start_time = time.time()
                
                # Check if we've been centered long enough
                elapsed = time.time() - self._align_center_start_time
                if elapsed >= CENTERING_HOLD_TIME:
                    pause_duration = 5.0
                    self.get_logger().info(
                        f'APPROACH ph0: stable alignment for {elapsed:.1f}s — '
                        f'pausing {pause_duration} s before ADVANCE')
                    self._approach_phase = 1
                    self._last_turn_sign = 0
                    self._align_center_start_time = None
                    self.start_pause(STATE_APPROACHING, 5.0)
                    return
                else:
                    # Within zone but waiting for timer — stop turning
                    self.send_vel(0.0, 0.0)
                    return
            else:
                # Out of zone — reset timer
                self._align_center_start_time = None

            # Turn speed logic with physical limit (0.15 rad/s) for in-place turns
            omega = math.copysign(max(CENTERING_MIN_OMEGA, abs(omega_raw)), omega_raw)
            self._last_turn_sign = int(math.copysign(1, omega))
            self._overshoot_recover_end = time.time() + OVERSHOOT_RECOVER_DUR

            _now = time.time()
            if _now - getattr(self, '_last_align_log', 0) >= 0.5:
                self._last_align_log = _now
                self.get_logger().info(
                    f'APPROACH ph0: aligning err_x={error_x:.1f}px, omega={omega:.3f}')

            self.send_vel(0.0, omega)
            return

        # ─────────────────────────────────────────────
        #  PHASE 1: ADVANCING — forward, with lateral steering
        # ─────────────────────────────────────────────
        if self._approach_phase == 1:
            # Object reached bottom of frame
            if pixel_y > IMAGE_HEIGHT - TARGET_Y_OFFSET:
                pause_duration = 5.0
                self.get_logger().info(
                    f'APPROACH ph1: object at bottom (y={pixel_y:.0f}) — '
                    f'pausing {pause_duration} s before FINE_CENTERING')
                self.start_pause(STATE_FINE_CENTERING, pause_duration)
                return

            target_y = IMAGE_HEIGHT - TARGET_Y_OFFSET
            error_y  = target_y - pixel_y
            v = KP_APPROACH_LINEAR * error_y
            v = max(MIN_APPROACH_LINEAR_VEL, min(v, APPROACH_LINEAR_VEL))

            # Simultaneous lateral correction — if the object drifts sideways
            # during the forward drive, steer gently to keep it centered.
            # No minimum omega here as forward motion helps the turn.
            omega = omega_raw

            _now = time.time()
            if _now - getattr(self, '_last_adv_log', 0) >= 0.5:
                self._last_adv_log = _now
                self.get_logger().info(
                    f'APPROACH ph1: v={v:.3f} m/s, omega={omega:.3f}, pixel_y={pixel_y:.0f}, error_x={error_x:.1f}px')

            self.send_vel(v, omega)
            return

    # ═══════════════════════════════════════════════════════════
    #  FINE CENTERING — precise lateral alignment before camera reset
    # ═══════════════════════════════════════════════════════════

    def do_fine_centering(self):
        """Rotate in place with physical min-omega until centred for 
        FINE_CENTERING_HOLD seconds, then pause 5 s before CAMERA_RESET."""
        if not self._has_fresh_detection():
            # Lost the target — go back to scanning
            self.send_vel(0.0, 0.0)
            self.state = STATE_SCANNING
            self.scan_start_time = time.time()
            self.scan_accumulated_theta = 0.0
            self.last_scan_theta = self.odom_theta
            self._fine_center_start = None
            self._approach_phase = 0
            self.get_logger().warn('Fine-center: detection lost, reverting to SCAN')
            return

        # Lag protection — if frame is stale, hold still
        age = time.time() - self.detection_stamp
        if age > 0.5:
            self.send_vel(0.0, 0.0)
            return

        det     = self.latest_detection
        error_x = IMAGE_CENTER_X - det.x

        # Sign-preserving minimum omega (real robot cannot move slower than 0.15)
        omega_raw = KP_ANGULAR * error_x
        if abs(error_x) < FINE_CENTERING_DEADZONE:
            omega = 0.0
        else:
            omega = math.copysign(max(CENTERING_MIN_OMEGA, abs(omega_raw)), omega_raw)

        # Log diagnostic info every ~0.5 s
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
                self.get_logger().info('Fine-centering: within deadzone, starting hold timer...')
            elif time.time() - self._fine_center_start >=FINE_CENTERING_HOLD:
                pause_duration = 5.0
                self.get_logger().info(f'Fine-centering done — pausing {pause_duration} s before CAMERA_RESET')
                self._fine_center_start = None
                self.publish_status('camera_reset')
                self.start_pause(STATE_CAMERA_RESET, pause_duration)
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

    # ═══════════════════════════════════════════════════════════
    #  PAUSE — generic 5-second countdown between steps
    # ═══════════════════════════════════════════════════════════

    def start_pause(self, next_state: str, duration: float = 5.0):
        """Stop the robot and wait *duration* seconds before entering *next_state*."""
        self.send_vel(0.0, 0.0)
        self._pause_end_time       = time.time() + duration
        self._pause_next_state     = next_state
        self._pause_last_countdown = int(math.ceil(duration))
        self.state = STATE_PAUSING
        self.get_logger().info(
            f'PAUSE: {duration:.0f} s before {next_state}  '
            f'[{self._pause_last_countdown}]')

    def do_pause(self):
        """Count down to zero with 1 s granularity logs, then advance."""
        remaining = self._pause_end_time - time.time()
        secs_left = int(math.ceil(remaining))
        if secs_left != self._pause_last_countdown and secs_left >= 0:
            self._pause_last_countdown = secs_left
            if secs_left > 0:
                self.get_logger().info(f'PAUSE: {secs_left}...')
        if remaining <= 0.0:
            next_st = self._pause_next_state
            self.get_logger().info(f'PAUSE done — entering {next_st}')
            if next_st == STATE_CAMERA_RESET:
                self._cam_reset_phase   = 0
                self._cam_reset_phase_t = time.time()
            elif next_st == STATE_FINE_CENTERING:
                self._fine_center_start = None
            self.state = next_st

    # ═══════════════════════════════════════════════════════════
    #  CAMERA RESET — tilt down then up so object visible close
    # ═══════════════════════════════════════════════════════════

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
    #  FINAL APPROACH — odom-based 0.5 m drive (no YOLO needed)
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
    #  RETURN TO START
    #
    #  Strategy:
    #    Use robust position controller from the base node by
    #    waiting for the /base/goal_reached flag.
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
        self.get_logger().debug(f'Checking detection freshness: latest_detection={self.latest_detection}, '
                               f'detection_stamp={self.detection_stamp:.2f}, now={time.time():.2f}, age={time.time() - self.detection_stamp:.2f}s')
        if self.latest_detection is None:
            return False
        return (time.time() - self.detection_stamp) < self.detection_timeout

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
