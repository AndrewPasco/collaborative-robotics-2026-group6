#!/usr/bin/env python3
"""
TidyBot2 Task 3 — Cube Segmentation Node

Listens to /arm_status and uses the Gemini Vision API to detect colored
cubes attached to the bottle:
  - arm_status = 0 (left arm active)  → detect RED / DARK-PINK cube (bottle side)
  - arm_status = 1 (right arm active) → detect YELLOW cube (bottle cap)

Publishes the detected bounding box once to /vision/bbox each time
arm_status changes to one of the above values.

Test Instructions (One-Command):
-------------------------------
1. In Terminal 1, run the node:
ros2 run tidybot_bringup segment_task3.py

2.
a. Bring up sim
ros2 launch tidybot_bringup sim.launch.py scene:=scene_task3.xml use_rviz:=true show_mujoco_viewer:=true camera_rate:=3.0
ros2 topic pub --once /arm_status std_msgs/msg/Int32 "{data: 0}"
ros2 topic pub --once /arm_status std_msgs/msg/Int32 "{data: 1}"

OR

b. Static
uv run python src/tidybot_bringup/scripts/vision_segmentation/test_segmentation.py ~/collaborative-robotics-2026-group6/examples/bottle.png 0


Topics:
-------
  Subscribes: /camera/color/image_raw, /arm_status
  Publishes:  /vision/bbox, /vision/bbox_debug
"""

import io, json, os, threading, time
import numpy as np
import google.generativeai as genai
from PIL import Image as PILImage
from cv_bridge import CvBridge
import cv2
from dotenv import load_dotenv, find_dotenv

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Int32, String

load_dotenv(find_dotenv())

VISION_MODEL = os.environ.get('GEMINI_VISION_MODEL', 'gemini-3-flash-preview')

CUBE_DESCRIPTIONS = {
    0: {'label': 'red_cubes', 'description': 'a series of severl red or dark-pink colored cubes stacked on top of each other'},
    1: {'label': 'yellow_cube', 'description': 'a small yellow colored cube'},
}


# ── Node ────────────────────────────────────────────────────────────────────

class SegmentTask3Node(Node):
    def __init__(self):
        super().__init__('segment_task3')
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('Missing GEMINI_API_KEY in environment or .env file')

        genai.configure(api_key=api_key)
        self.get_logger().info(f'Gemini configured with model: {VISION_MODEL}')

        self._latest_image = None
        self._image_lock = threading.Lock()
        self._arm_status = None
        self._detecting = False
        self._detection_lock = threading.Lock()
        self._cv_bridge = CvBridge()

        self._bbox_pub = self.create_publisher(RegionOfInterest, '/vision/bbox', 10)
        self._debug_pub = self.create_publisher(String, '/vision/bbox_debug', 10)
        self._image_debug_pub = self.create_publisher(Image, '/vision/segment_task3_debug', 10)

        self.create_subscription(Image, '/camera/color/image_raw', self._image_cb, 10)
        self.create_subscription(Int32, '/arm_status', self._arm_status_cb, 10)
        self.get_logger().info('Segment Task3 Node ready. Waiting for /arm_status...')

    def _image_cb(self, msg):
        with self._image_lock:
            self._latest_image = msg

    def _arm_status_cb(self, msg):
        new_status = msg.data
        if new_status not in CUBE_DESCRIPTIONS:
            return

        self.get_logger().info(f'arm_status changed: {self._arm_status} -> {new_status}')
        self._arm_status = new_status

        with self._image_lock:
            snapshot = self._latest_image

        if snapshot is None:
            self.get_logger().warn('No image received yet.')
            return

        with self._detection_lock:
            if self._detecting: return
            self._detecting = True

        threading.Thread(target=self._run_detection, args=(snapshot, new_status), daemon=True).start()

    def _run_detection(self, ros_img, arm_status):
        info = CUBE_DESCRIPTIONS[arm_status]
        try:
            while rclpy.ok():
                pil_img = self._ros_image_to_pil(ros_img)
                if pil_img is None: return

                w, h = pil_img.size
                bbox_norm = self._query_gemini(pil_img, info['description'])

                if bbox_norm:
                    ymin, xmin, ymax, xmax = bbox_norm
                    
                    # Scale normalized (0-1000) to pixel coordinates
                    padding = 3
                    x1 = max(0, int(xmin * w / 1000) - padding)
                    y1 = max(0, int(ymin * h / 1000) - padding)
                    x2 = min(w, int(xmax * w / 1000) + padding)
                    y2 = min(h, int(ymax * h / 1000) + padding)
                    
                    roi = RegionOfInterest(x_offset=x1, y_offset=y1,
                                         width=int(x2-x1), height=int(y2-y1), do_rectify=False)
                    self._bbox_pub.publish(roi)
                    self._publish_debug({'target': info['label'], 'arm_status': arm_status, 'bbox_px': [x1, y1, x2, y2], 'bbox_norm': list(bbox_norm)})
                    self.get_logger().info(f'Detected {info["label"]} at px: ({x1},{y1}) to ({x2},{y2})')
                    
                    # Publish debug image
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_img, f"{info['label']} ({arm_status})", (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    try:
                        debug_msg = self._cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                        debug_msg.header = ros_img.header
                        self._image_debug_pub.publish(debug_msg)
                    except Exception as e:
                        self.get_logger().error(f'Failed to publish debug image: {e}')
                    
                    break
                else:
                    self.get_logger().warn(f'Could not locate {info["label"]}. Retrying in 5s...')
                    # Publish the raw image anyway for context if detection failed
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.putText(cv_img, f"NOT FOUND: {info['label']}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    try:
                        debug_msg = self._cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                        debug_msg.header = ros_img.header
                        self._image_debug_pub.publish(debug_msg)
                    except: pass
                    
                    time.sleep(5)
                    with self._image_lock:
                        if self._latest_image is not None:
                            ros_img = self._latest_image
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
        finally:
            with self._detection_lock:
                self._detecting = False

    def _query_gemini(self, pil_img, desc):
        prompt = (f'Locate {desc}. '
                  'Return ONLY a JSON list of normalized coordinates [ymin, xmin, ymax, xmax] in the range 0 to 1000. '
                  'Example Output: [100, 200, 300, 400] or {"error": "not_found"}. No other text.')
        try:
            model = genai.GenerativeModel(VISION_MODEL)
            response = model.generate_content([prompt, pil_img])
            self.get_logger().info(f'Gemini response: {response.text}')
            text = response.text.strip('`').strip()
            if text.startswith('json'): text = text[4:].strip()
            
            # Try to parse as JSON first
            try:
                data = json.loads(text)
            except:
                # If it's not a list, it might be a dict with error
                if '"error"' in text: return None
                return None

            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], list):
                    data = data[0]
                if len(data) == 4:
                    return [int(v) for v in data]
            if isinstance(data, dict) and 'error' in data:
                return None
            return None
        except Exception as e:
            self.get_logger().warn(f'Gemini query failed: {e}')
            return None

    def _ros_image_to_pil(self, msg):
        try:
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            enc = msg.encoding.lower()
            if enc in ('rgb8', 'rgb'):
                return PILImage.fromarray(data.reshape((msg.height, msg.width, 3)), 'RGB')
            if enc in ('bgr8', 'bgr'):
                return PILImage.fromarray(data.reshape((msg.height, msg.width, 3))[:, :, ::-1], 'RGB')
            return None
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return None

    def _publish_debug(self, data):
        self._debug_pub.publish(String(data=json.dumps(data)))

def main(args=None):
    rclpy.init(args=args)
    node = SegmentTask3Node()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
