#!/usr/bin/env python3
"""
vision_yolo_gemini.py — Enhanced Vision Node with YOLO + Gemini
================================================================
Combines fast local detection with flexible open-vocabulary queries:
  - YOLO: Fast object detection for known classes (runs every frame)
  - Gemini: Open-vocabulary detection for arbitrary queries (on demand)
  - NOTE: Gemini functionality is currently UNTESTED.

Publishes:
  /object_detection  (geometry_msgs/Point)  — x=pixel_x, y=pixel_y, z=bbox_area
  /vision/detections (std_msgs/String)      — JSON list of all detected objects
  /vision/target_bbox (sensor_msgs/RegionOfInterest) — Best detection bbox

Subscribes:
  /camera/color/image_raw  (sensor_msgs/Image)  — RGB from camera
  /vision/target           (std_msgs/String)    — What to find (e.g., "red cup", "toy")


ros2 topic pub --once /vision/target std_msgs/msg/String "{data: 'banana'}"

"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import json
import base64
import time
from dotenv import load_dotenv
load_dotenv()

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import RegionOfInterest

# --- Navigator Constants (imported for visualization) ---
from navigator import (
    IMAGE_CENTER_X, 
    CENTERING_SLOWDOWN_ZONE, 
    CENTERING_DEADZONE, 
    FINE_CENTERING_DEADZONE,
    TARGET_Y_OFFSET
)






# YOLO model 
YOLO_MODEL = "yolo26x.pt"

# Gemini model
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# Minimum confidence for YOLO detections
YOLO_CONFIDENCE = 0.5



# How often to run Gemini (not every frame - too slow/expensive)
GEMINI_QUERY_INTERVAL = 2.0  # seconds

# Maximum rate to publish the annotated debug image (Hz)
_ANNOTATED_MAX_HZ = 20.0
_ANNOTATED_MIN_PERIOD = 1.0 / _ANNOTATED_MAX_HZ


class VisionYoloGemini(Node):
    """
    Enhanced Vision node with YOLO + Gemini.

    YOLO runs on every frame for fast detection.
    Gemini is called on-demand for open-vocabulary queries.
    """

    def __init__(self):
        super().__init__('vision_yolo_gemini')
        self.get_logger().info('Vision (YOLO+Gemini) node starting...')
        self.get_logger().info(f'IMAGE_CENTER_X: {IMAGE_CENTER_X}')
        self.get_logger().info(f'CENTERING_SLOWDOWN_ZONE: {CENTERING_SLOWDOWN_ZONE}')
        self.get_logger().info(f'CENTERING_DEADZONE: {CENTERING_DEADZONE}')
        self.get_logger().info(f'FINE_CENTERING_DEADZONE: {FINE_CENTERING_DEADZONE}')
        self.get_logger().info(f'TARGET_Y_OFFSET: {TARGET_Y_OFFSET}')
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
        # DISABLED: self._load_gemini()

        # ── Publishers ──
        self.detection_pub = self.create_publisher(
            Point, '/object_detection', 10)
        self.detections_json_pub = self.create_publisher(
            String, '/vision/detections', 10)
        self.annotated_pub = self.create_publisher(
            Image, '/vision/annotated_image', 10)
        self.target_bbox_pub = self.create_publisher(
            RegionOfInterest,
            "/vision/target_bbox",
            10
        )

        self.cv_bridge = CvBridge()

        # ── Subscribers ──
        self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_cb, 5)
        self.create_subscription(
            String, '/vision/target',
            self.target_cb, 1)



        self.get_logger().info(
            f'Vision ready. YOLO={self.yolo_available}, Gemini={self.gemini_available}')

    def _load_yolo(self):
        """Load YOLO model from a stable directory (no cwd changes)."""
        try:
            from ultralytics import YOLO
            model_dir = os.path.expanduser('~/.yolo_models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, YOLO_MODEL)

            if not os.path.exists(model_path):
                self.get_logger().info(f'Downloading {YOLO_MODEL} to {model_path}...')
                import urllib.request
                url = f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{YOLO_MODEL}'
                urllib.request.urlretrieve(url, model_path)
                self.get_logger().info(f'Download complete: {model_path}')

            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.get_logger().info(f'Using device: {device}')
            
            self.yolo_model = YOLO(model_path).to(device)
            self.yolo_available = True
            
            # Pre-bake device / precision into the model's overrides so they
            # are NOT re-validated on every single predict() call.
            self.yolo_model.overrides.update({
                'device':  device,
                'conf':    YOLO_CONFIDENCE,
                'verbose': False,
            })

            self.get_logger().info('Warming up YOLO...')
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.yolo_model.predict(dummy_frame)

            yolo_classes = list(self.yolo_model.names.values())
            self.get_logger().info(f'YOLO classes (best.pt): {yolo_classes}')
            actual_device = next(self.yolo_model.model.parameters()).device
            self.get_logger().info(f'YOLO loaded on {actual_device}: {model_path}')
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
            yolo_classes = list(self.yolo_model.names.values())
            self.get_logger().info(f'Available YOLO classes: {yolo_classes}')
            for cls in yolo_classes:
                if cls.lower() in query or query in cls.lower():
                    self.target_class = cls
                    self.get_logger().info(f'Matched YOLO class: {cls}')
                    return

        self.target_class = None
        self.get_logger().info(f'No YOLO class match for "{query}", will use Gemini')

    def image_cb(self, msg: Image):
        """Process each camera frame."""
        if not hasattr(self, '_frame_count'): self._frame_count = 0
        if not hasattr(self, '_last_log_time'): self._last_log_time = 0.0
        self._frame_count += 1
        # self.get_logger().info(f'Frame {self._frame_count} received')

        now = self.get_clock().now().nanoseconds * 1e-9
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = now - msg_time if msg_time > 0 else 0.0

        if now - self._last_log_time >= 10.0:
            self._last_log_time = now
            yolo_classes = [d['class'] for d in self.latest_detections] if self.latest_detections else []
            self.get_logger().info(
                f'Frame {self._frame_count} | age: {age:.2f}s | target: {self.target_query} | '
                f'target_class: {self.target_class} | '
                f'YOLO sees: {yolo_classes}')

        frame = self._ros_image_to_cv2(msg)
        if frame is None:
            return
        
        # save raw frame for debugging
        # cv2.imwrite(f'/home/locobot/collaborative-robotics-2026-group6/ros2_ws/src/tidybot_bringup/scripts/Navigation/debuggging_saves/img_{self._frame_count}.jpg', frame)

        # 1. Run YOLO (every frame)
        if self.yolo_available:
            self._run_yolo(frame)

        # 2. Run Gemini if YOLO did not find the target
        if (self.gemini_available and self.target_query and self.target_class is None):
            self._run_gemini_if_needed(frame)        # 3. Publish best detection for Navigator
        self._publish_detection()


        # 5. Publish all detections as JSON
        self._publish_detections_json()

        # Annotated image: gated by subscriber count and rate-limited to 5 Hz
        _now = time.time()
        if (self.annotated_pub.get_subscription_count() > 0 and
                _now - getattr(self, '_last_annotated_time', 0.0) >= _ANNOTATED_MIN_PERIOD):
            self._last_annotated_time = _now
            self._publish_annotated_image(frame, msg.header)

    # ═══════════════════════════════════════════════════════════
    #  YOLO DETECTION
    # ═══════════════════════════════════════════════════════════

    def _run_yolo(self, frame: np.ndarray):
        """Run YOLO object detection."""
        # Diagnostic: Log resolution once on the first frame
        if self._frame_count == 1:
            h, w = frame.shape[:2]
            self.get_logger().info(f'Input frame resolution: {w}x{h}')
        
        results = self.yolo_model.predict(frame)

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


        # Publish only when a target has been requested and matched
        if best:
            det = Point()
            det.x = float(best['cx'])
            det.y = float(best['cy'])
            det.z = float(best['area'])
            self.detection_pub.publish(det)


            # ── For Pasco: Publish bbox + class id ──
            x1, y1, x2, y2 = best['bbox']

            cx = int(best['cx'])
            cy = int(best['cy'])
            w = int(x2 - x1)
            h = int(y2 - y1)

            roi_msg = RegionOfInterest()
            roi_msg.x_offset = int(cx - w / 2)
            roi_msg.y_offset = int(cy - h / 2)
            roi_msg.width = int(w)
            roi_msg.height = int(h)
            roi_msg.do_rectify = False

            self.target_bbox_pub.publish(roi_msg)


            # Log when we publish a detection (once per second max)
            import time as _t
            _now = _t.time()
            if _now - getattr(self, '_last_det_log', 0) >= 15.0:
                self._last_det_log = _now
                self.get_logger().info(
                    f'Published detection: {best.get("class","?")} '
                    f'at ({best["cx"]:.0f},{best["cy"]:.0f}) area={best["area"]:.0f}')

    def _publish_detections_json(self):
        """Publish all detections as JSON for debugging/Brain node."""
        all_dets = self.latest_detections.copy()
        if self.gemini_result:
            all_dets.append(self.gemini_result)

        msg = String()
        msg.data = json.dumps(all_dets)
        self.detections_json_pub.publish(msg)

    def _publish_annotated_image(self, frame, header):
        """Draw bounding boxes and publish the annotated image."""
        annotated_frame = frame.copy()
        
        # Draw YOLO detections
        for det in self.latest_detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det.get('class', '')} {det.get('confidence', 0.0):.2f}"
            
            # Highlight target class in green, others in blue
            if self.target_class and det.get('class', '').lower() == self.target_class.lower():
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (255, 0, 0)
                thickness = 1
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, label, (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            # Draw center dot for this detection
            cx, cy = int(det.get('cx', 0)), int(det.get('cy', 0))
            cv2.circle(annotated_frame, (cx, cy), 4, (255, 255, 255), -1)

        # Draw Gemini detection if any
        if self.gemini_result:
            x1, y1, x2, y2 = [int(v) for v in self.gemini_result['bbox']]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Gemini: {self.target_query}", (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw center dot for Gemini detection
            cx, cy = int(self.gemini_result.get('cx', 0)), int(self.gemini_result.get('cy', 0))
            cv2.circle(annotated_frame, (cx, cy), 4, (255, 255, 255), -1)

        # --- Draw Navigation Markers ---
        h, w = annotated_frame.shape[:2]
        
        # 1. Centering Zones (Vertical lines)
        # Slowdown Zone (Yellow)
        cv2.line(annotated_frame, (int(IMAGE_CENTER_X - CENTERING_SLOWDOWN_ZONE), 0), 
                 (int(IMAGE_CENTER_X - CENTERING_SLOWDOWN_ZONE), h), (0, 255, 255), 2)
        cv2.line(annotated_frame, (int(IMAGE_CENTER_X + CENTERING_SLOWDOWN_ZONE), 0), 
                 (int(IMAGE_CENTER_X + CENTERING_SLOWDOWN_ZONE), h), (0, 255, 255), 2)
        
        # Deadzone (Red)
        cv2.line(annotated_frame, (int(IMAGE_CENTER_X - CENTERING_DEADZONE), 0), 
                 (int(IMAGE_CENTER_X - CENTERING_DEADZONE), h), (0, 0, 255), 2)
        cv2.line(annotated_frame, (int(IMAGE_CENTER_X + CENTERING_DEADZONE), 0), 
                 (int(IMAGE_CENTER_X + CENTERING_DEADZONE), h), (0, 0, 255), 2)

        # Fine Centering Deadzone (Orange)
        cv2.line(annotated_frame, (int(IMAGE_CENTER_X - FINE_CENTERING_DEADZONE), 0), 
                 (int(IMAGE_CENTER_X - FINE_CENTERING_DEADZONE), h), (0, 165, 255), 2)
        cv2.line(annotated_frame, (int(IMAGE_CENTER_X + FINE_CENTERING_DEADZONE), 0), 
                 (int(IMAGE_CENTER_X + FINE_CENTERING_DEADZONE), h), (0, 165, 255), 2)

        # 2. Target Y Offset (Horizontal line - Blue)
        # The robot stops when pixel_y > IMAGE_HEIGHT - TARGET_Y_OFFSET
        y_line = int(h - TARGET_Y_OFFSET)
        cv2.line(annotated_frame, (0, y_line), (w, y_line), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Stop Line", (10, y_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Publish
        try:
            annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            annotated_msg.header = header
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}", once=True)

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