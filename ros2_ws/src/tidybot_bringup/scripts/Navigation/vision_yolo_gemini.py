#!/usr/bin/env python3
"""
vision_yolo_gemini.py — Enhanced Vision Node with YOLO + Gemini
================================================================
Combines fast local detection with flexible open-vocabulary queries:
  - YOLO: Fast object detection for known classes (runs every frame)
  - Gemini: Open-vocabulary detection for arbitrary queries (on demand)
  - AprilTag: Pose estimation via solvePnP (same as vision_nav.py)

Publishes:
  /object_detection  (geometry_msgs/Point)  — x=pixel_x, y=pixel_y, z=bbox_area
  /apriltag_pose     (geometry_msgs/Pose)   — PnP pose (position.z = depth)
  /vision/detections (std_msgs/String)      — JSON list of all detected objects

Subscribes:
  /camera/color/image_raw  (sensor_msgs/Image)  — RGB from camera
  /vision/target           (std_msgs/String)    — What to find (e.g., "red cup", "toy")

Setup:
  1. pip install ultralytics google-generativeai opencv-contrib-python
  2. Set GEMINI_API_KEY environment variable
  3. chmod +x vision_yolo_gemini.py
  4. Add to CMakeLists.txt
  5. colcon build --packages-select tidybot_bringup
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import os
import json
import base64
from io import BytesIO

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

# YOLO model — yolov8n is fastest, yolov8s/m for better accuracy
YOLO_MODEL = "yolov8n.pt"

# Gemini model
GEMINI_MODEL = "gemini-1.5-flash"

# Minimum confidence for YOLO detections
YOLO_CONFIDENCE = 0.5

# Camera intrinsics (adjust for your camera)
CAMERA_MATRIX = np.array([
    [615.0,   0.0, 320.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# AprilTag size (meters, half side length)
TAG_HALF_SIZE = 0.075

# How often to run Gemini (not every frame - too slow/expensive)
GEMINI_QUERY_INTERVAL = 2.0  # seconds


class VisionYoloGemini(Node):
    """
    Enhanced Vision node with YOLO + Gemini.

    YOLO runs on every frame for fast detection.
    Gemini is called on-demand for open-vocabulary queries.
    """

    def __init__(self):
        super().__init__('vision_yolo_gemini')
        self.get_logger().info('Vision (YOLO+Gemini) node starting...')

        # ── Target to find ──
        self.target_query = None  # e.g., "red cup", "toy dinosaur"
        self.target_class = None  # YOLO class name if applicable

        # ── Detection state ──
        self.latest_detections = []  # All YOLO detections
        self.gemini_result = None    # Last Gemini response
        self.last_gemini_time = 0.0

        # ── Load YOLO ──
        self.yolo_model = None
        self.yolo_available = False
        self._load_yolo()

        # ── Load Gemini ──
        self.gemini_client = None
        self.gemini_available = False
        self.gemini_use_vertex = False
        self._load_gemini()

        # ── Publishers ──
        self.detection_pub = self.create_publisher(
            Point, '/object_detection', 10)
        self.apriltag_pub = self.create_publisher(
            Pose, '/apriltag_pose', 10)
        self.detections_json_pub = self.create_publisher(
            String, '/vision/detections', 10)

        # ── Subscribers ──
        # Use sensor_data QoS profile for camera (BEST_EFFORT, VOLATILE)
        self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_cb, qos_profile_sensor_data)
        self.create_subscription(
            String, '/vision/target',
            self.target_cb, 10)

        # ── ArUco detector ──
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        s = TAG_HALF_SIZE
        self.tag_object_points = np.array([
            [-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]
        ], dtype=np.float64)

        self.get_logger().info(
            f'Vision ready. YOLO={self.yolo_available}, Gemini={self.gemini_available}')

    def _load_yolo(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL)
            self.yolo_available = True
            self.get_logger().info(f'YOLO loaded: {YOLO_MODEL}')
        except ImportError:
            self.get_logger().warn(
                'ultralytics not installed. Run: pip install ultralytics')
        except Exception as e:
            self.get_logger().error(f'YOLO load failed: {e}')

    def _load_gemini(self):
        """Load Gemini client (supports API key or service account)."""
        # Method 1: Check for simple API key
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel(GEMINI_MODEL)
                self.gemini_available = True
                self.get_logger().info(f'Gemini loaded via API key: {GEMINI_MODEL}')
                return
            except Exception as e:
                self.get_logger().warn(f'Gemini API key failed: {e}')

        # Method 2: Check for service account credentials file
        creds_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_file:
            # Check default location
            default_creds = os.path.expanduser('~/Downloads/astral-scout-444409-e9-e3859a36bb8d.json')
            if os.path.exists(default_creds):
                creds_file = default_creds
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file

        if creds_file and os.path.exists(creds_file):
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                # Extract project ID from credentials file
                import json
                with open(creds_file) as f:
                    creds = json.load(f)
                project_id = creds.get('project_id', 'astral-scout-444409-e9')

                vertexai.init(project=project_id, location='us-central1')
                self.gemini_client = GenerativeModel('gemini-1.5-flash')
                self.gemini_available = True
                self.gemini_use_vertex = True
                self.get_logger().info(f'Gemini loaded via Vertex AI (project: {project_id})')
                return
            except ImportError:
                self.get_logger().warn(
                    'vertexai not installed. Run: pip install google-cloud-aiplatform')
            except Exception as e:
                self.get_logger().warn(f'Vertex AI init failed: {e}')

        self.get_logger().warn(
            'Gemini not configured. Set GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS')

    def target_cb(self, msg: String):
        """Set what object to find."""
        query = msg.data.strip().lower()
        self.target_query = query
        self.get_logger().info(f'Target set: "{query}"')

        # Check if it matches a YOLO class
        if self.yolo_available:
            yolo_classes = self.yolo_model.names.values()
            for cls in yolo_classes:
                if cls.lower() in query or query in cls.lower():
                    self.target_class = cls
                    self.get_logger().info(f'Matched YOLO class: {cls}')
                    return

        self.target_class = None
        self.get_logger().info('No YOLO class match, will use Gemini')

    def image_cb(self, msg: Image):
        """Process each camera frame."""
        frame = self._ros_image_to_cv2(msg)
        if frame is None:
            return

        # 1. Run YOLO (every frame)
        if self.yolo_available:
            self._run_yolo(frame)

        # 2. Run Gemini (periodically, if target set and no YOLO match)
        if (self.gemini_available and
            self.target_query and
            self.target_class is None):
            self._run_gemini_if_needed(frame)

        # 3. Publish best detection for Navigator
        self._publish_detection()

        # 4. Detect AprilTags
        self._detect_apriltag(frame)

        # 5. Publish all detections as JSON
        self._publish_detections_json()

    # ═══════════════════════════════════════════════════════════
    #  YOLO DETECTION
    # ═══════════════════════════════════════════════════════════

    def _run_yolo(self, frame: np.ndarray):
        """Run YOLO object detection."""
        results = self.yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)

        self.latest_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                self.latest_detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'cx': cx,
                    'cy': cy,
                    'area': area,
                    'bbox': [x1, y1, x2, y2],
                    'source': 'yolo'
                })

    # ═══════════════════════════════════════════════════════════
    #  GEMINI DETECTION
    # ═══════════════════════════════════════════════════════════

    def _run_gemini_if_needed(self, frame: np.ndarray):
        """Run Gemini for open-vocabulary detection (rate-limited)."""
        import time
        now = time.time()
        if now - self.last_gemini_time < GEMINI_QUERY_INTERVAL:
            return

        self.last_gemini_time = now

        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_bytes = buffer.tobytes()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # Query Gemini
            prompt = f"""Look at this image and find: "{self.target_query}"

If you see the object, respond with ONLY a JSON object like:
{{"found": true, "x": <center_x_pixel>, "y": <center_y_pixel>, "width": <bbox_width>, "height": <bbox_height>, "description": "<brief description>"}}

If you don't see it, respond with:
{{"found": false}}

Image dimensions are 640x480. Give pixel coordinates."""

            # Call Gemini (different API for Vertex vs genai)
            if self.gemini_use_vertex:
                from vertexai.generative_models import Part, Image as VertexImage
                image_part = Part.from_data(img_bytes, mime_type="image/jpeg")
                response = self.gemini_client.generate_content([prompt, image_part])
            else:
                response = self.gemini_client.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_base64}
                ])

            # Parse response
            text = response.text.strip()
            # Extract JSON from response
            if '{' in text and '}' in text:
                json_str = text[text.find('{'):text.rfind('}')+1]
                result = json.loads(json_str)

                if result.get('found'):
                    cx = result.get('x', 320)
                    cy = result.get('y', 240)
                    w = result.get('width', 100)
                    h = result.get('height', 100)

                    self.gemini_result = {
                        'class': self.target_query,
                        'confidence': 0.8,  # Gemini doesn't give confidence
                        'cx': cx,
                        'cy': cy,
                        'area': w * h,
                        'bbox': [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        'source': 'gemini',
                        'description': result.get('description', '')
                    }
                    self.get_logger().info(
                        f'Gemini found "{self.target_query}" at ({cx}, {cy})')
                else:
                    self.gemini_result = None

        except Exception as e:
            self.get_logger().warn(f'Gemini query failed: {e}')

    # ═══════════════════════════════════════════════════════════
    #  PUBLISH DETECTION FOR NAVIGATOR
    # ═══════════════════════════════════════════════════════════

    def _publish_detection(self):
        """Publish the best detection matching the target."""
        best = None

        # If we have a target, find matching detection
        if self.target_query:
            # First check YOLO for matching class
            if self.target_class:
                for det in self.latest_detections:
                    if det['class'].lower() == self.target_class.lower():
                        if best is None or det['area'] > best['area']:
                            best = det

            # If no YOLO match, use Gemini result
            if best is None and self.gemini_result:
                best = self.gemini_result

        # If no target set, just use largest detection
        if best is None and self.latest_detections:
            best = max(self.latest_detections, key=lambda d: d['area'])

        # Publish
        if best:
            det = Point()
            det.x = float(best['cx'])
            det.y = float(best['cy'])
            det.z = float(best['area'])
            self.detection_pub.publish(det)

    def _publish_detections_json(self):
        """Publish all detections as JSON for debugging/Brain node."""
        all_dets = self.latest_detections.copy()
        if self.gemini_result:
            all_dets.append(self.gemini_result)

        msg = String()
        msg.data = json.dumps(all_dets)
        self.detections_json_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════
    #  APRILTAG DETECTION (same as vision_nav.py)
    # ═══════════════════════════════════════════════════════════

    def _detect_apriltag(self, frame: np.ndarray):
        """Detect AprilTag and compute pose via solvePnP."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        image_points = corners[0].reshape(4, 2).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.tag_object_points, image_points,
            CAMERA_MATRIX, DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if not success:
            return

        R, _ = cv2.Rodrigues(rvec)
        qw, qx, qy, qz = self._rotation_matrix_to_quaternion(R)

        pose = Pose()
        pose.position.x = float(tvec[0][0])
        pose.position.y = float(tvec[1][0])
        pose.position.z = float(tvec[2][0])
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self.apriltag_pub.publish(pose)

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════

    def _ros_image_to_cv2(self, msg: Image) -> np.ndarray:
        """Convert ROS Image to OpenCV BGR."""
        try:
            h, w = msg.height, msg.width
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
        """Convert rotation matrix to quaternion."""
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
    node = VisionYoloGemini()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
