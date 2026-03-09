#!/usr/bin/env python3
"""
vision_nav.py — Vision Node for Navigator (Sim Testing)
========================================================
Combines HW2 code into a ROS 2 node:
  - P5: HSV color detection → object pixel coords + bbox area
  - P6: AprilTag detection + solvePnP → 6DOF pose

This is a MINIMAL version for testing the Navigator in simulation.
Your Vision teammates will build the real one with YOLO/Gemini/etc.

Publishes:
  /object_detection  (geometry_msgs/Point)  — x=pixel_x, y=pixel_y, z=bbox_area
  /apriltag_pose     (geometry_msgs/Pose)   — PnP pose (position.z = depth)

Subscribes:
  /camera/color/image_raw  (sensor_msgs/Image)  — RGB from RealSense sim

Setup:
  1. pip install opencv-contrib-python  (for ArUco detection)
     Or in the Docker container it should already be available.
  2. cp vision_nav.py ros2_ws/src/tidybot_bringup/scripts/
  3. chmod +x ros2_ws/src/tidybot_bringup/scripts/vision_nav.py
  4. Add to CMakeLists.txt
  5. colcon build --packages-select tidybot_bringup
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import String


# ╔══════════════════════════════════════════════════════════════╗
# ║  HSV COLOR RANGES — same tuning approach as HW2 P5         ║
# ║  Adjust these for whatever objects are in your sim scene    ║
# ╚══════════════════════════════════════════════════════════════╝

# Each entry: (lower_hsv, upper_hsv)
# OpenCV HSV: H=[0,179], S=[0,255], V=[0,255]
COLOR_RANGES = {
    'red':    [( np.array([0,   120, 70]),  np.array([10,  255, 255]) ),
               ( np.array([170, 120, 70]),  np.array([180, 255, 255]) )],  # red wraps around
    'green':  [( np.array([35,  120, 70]),  np.array([85,  255, 255]) )],
    'blue':   [( np.array([100, 120, 70]),  np.array([130, 255, 255]) )],
    'yellow': [( np.array([20,  120, 70]),  np.array([35,  255, 255]) )],
}

# Which color to track — Brain would set this, for now just hardcode
DEFAULT_TARGET_COLOR = 'red'

# Minimum contour area to count as a real detection (filters noise)
MIN_CONTOUR_AREA = 500  # pixels^2


# ╔══════════════════════════════════════════════════════════════╗
# ║  CAMERA INTRINSICS — from HW2 P6 or RealSense defaults     ║
# ║  Replace with actual K matrix from your camera/sim          ║
# ╚══════════════════════════════════════════════════════════════╝

# RealSense D435 typical intrinsics at 640x480
# You may need to adjust these for the sim camera
CAMERA_MATRIX = np.array([
    [615.0,   0.0, 320.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)  # assume no distortion in sim

# AprilTag physical size (meters) — half the side length
# Standard 6" tag ≈ 0.15m, so half = 0.075m
# Adjust to match whatever tags are in your scene
TAG_HALF_SIZE = 0.075


class VisionNav(Node):
    """
    Minimal Vision node for sim testing.

    Runs at camera frame rate. For each frame:
      1. HSV color detection (HW2 P5) → /object_detection
      2. ArUco/AprilTag detection + solvePnP (HW2 P6) → /apriltag_pose
    """

    def __init__(self):
        super().__init__('vision_nav')
        self.get_logger().info('Vision (nav) node starting...')

        # What color to look for
        self.target_color = DEFAULT_TARGET_COLOR

        # ── Publishers ──
        self.detection_pub = self.create_publisher(
            Point, '/object_detection', 10)
        self.apriltag_pub = self.create_publisher(
            Pose, '/apriltag_pose', 10)

        # ── Subscribers ──
        # Camera image — check `ros2 topic list` for the actual topic name
        # Common options: /camera/color/image_raw, /camera/image_raw,
        #                 /camera/rgb/image_raw
        self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_cb, 10)

        # Target color from Brain (optional — lets Brain say "find the red one")
        self.create_subscription(
            String, '/vision/target_color',
            self.color_cb, 10)

        # ── ArUco detector setup ──
        # Using ArUco as a stand-in for AprilTags — same concept, OpenCV native
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 3D points of tag corners in tag-local frame (HW2 P6 pattern)
        # Order: bottom-left, bottom-right, top-right, top-left
        # (matches AprilTag corner indexing from HW2: 0, 1, 2, 3)
        s = TAG_HALF_SIZE
        self.tag_object_points = np.array([
            [-s, -s, 0.0],
            [ s, -s, 0.0],
            [ s,  s, 0.0],
            [-s,  s, 0.0],
        ], dtype=np.float64)

        self.get_logger().info(
            f'Vision ready. Tracking color="{self.target_color}". '
            f'Listening on /camera/color/image_raw')

    def color_cb(self, msg: String):
        """Update which color to track."""
        color = msg.data.strip().lower()
        if color in COLOR_RANGES:
            self.target_color = color
            self.get_logger().info(f'Now tracking: {color}')
        else:
            self.get_logger().warn(f'Unknown color: {color}')

    def image_cb(self, msg: Image):
        """
        Main callback — runs on every camera frame.

        Converts ROS Image → OpenCV, then runs:
          1. Color detection (HW2 P5)
          2. ArUco detection + PnP (HW2 P6)
        """
        # ── Convert ROS Image to OpenCV BGR ──
        # sensor_msgs/Image with encoding "rgb8" or "bgr8"
        frame = self._ros_image_to_cv2(msg)
        if frame is None:
            return

        # ── 1. Color detection (HW2 P5) ──
        self.detect_color(frame)

        # ── 2. AprilTag detection + PnP (HW2 P6) ──
        self.detect_apriltag(frame)

    # ═══════════════════════════════════════════════════════════
    #  COLOR DETECTION (HW2 P5)
    #
    #  Same approach: BGR → HSV → mask → contours → centroid + area
    # ═══════════════════════════════════════════════════════════

    def detect_color(self, frame: np.ndarray):
        """
        Detect target color using HSV masking.
        Identical to your HW2 P5 color detection code.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build mask from all ranges for this color
        ranges = COLOR_RANGES.get(self.target_color, [])
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        # Clean up noise (optional but helps)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        # Take the largest contour (most likely the target object)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_CONTOUR_AREA:
            return

        # Centroid via moments
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Bounding box area (what Navigator uses for "close enough" check)
        x, y, w, h = cv2.boundingRect(largest)
        bbox_area = w * h

        # Publish
        det = Point()
        det.x = float(cx)
        det.y = float(cy)
        det.z = float(bbox_area)
        self.detection_pub.publish(det)

    # ═══════════════════════════════════════════════════════════
    #  APRILTAG DETECTION + PnP (HW2 P6)
    #
    #  Detect ArUco markers → get corner pixels → solvePnP
    #  Same as your HW2 P6 code but wrapped in a ROS callback
    # ═══════════════════════════════════════════════════════════

    def detect_apriltag(self, frame: np.ndarray):
        """
        Detect AprilTag/ArUco and compute pose via solvePnP.
        Same algorithm as HW2 P6.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return

        # Use the first detected tag
        # corners[0] shape: (1, 4, 2) — four corner pixel coordinates
        image_points = corners[0].reshape(4, 2).astype(np.float64)

        # solvePnP — this is your HW2 P6 code
        success, rvec, tvec = cv2.solvePnP(
            self.tag_object_points,
            image_points,
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE  # good for planar targets
        )

        if not success:
            return

        # Convert rotation vector to rotation matrix (for quaternion)
        R, _ = cv2.Rodrigues(rvec)

        # Convert rotation matrix to quaternion
        qw, qx, qy, qz = self._rotation_matrix_to_quaternion(R)

        # Publish pose
        pose = Pose()
        pose.position.x = float(tvec[0][0])  # lateral offset (meters)
        pose.position.y = float(tvec[1][0])  # vertical offset (meters)
        pose.position.z = float(tvec[2][0])  # DEPTH (meters)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self.apriltag_pub.publish(pose)

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════

    def _ros_image_to_cv2(self, msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image to OpenCV BGR numpy array.

        Done manually to avoid cv_bridge dependency issues in some
        Docker/ROS setups. If cv_bridge works for you, use that instead:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        """
        try:
            # Determine shape
            h = msg.height
            w = msg.width

            if msg.encoding == 'rgb8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            elif msg.encoding == 'mono8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f'Unknown encoding: {msg.encoding}', once=True)
                return None

            return frame
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}', once=True)
            return None

    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray):
        """
        Convert 3x3 rotation matrix to quaternion (w, x, y, z).
        Standard algorithm — same math you'd find in any robotics textbook.
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return w, x, y, z


def main(args=None):
    rclpy.init(args=args)
    node = VisionNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
