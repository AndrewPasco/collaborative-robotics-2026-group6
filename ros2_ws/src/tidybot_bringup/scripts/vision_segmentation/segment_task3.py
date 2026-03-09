#!/usr/bin/env python3
"""
TidyBot2 Task 3 — Cube Segmentation Node

Listens to /arm_status and uses the Gemini Vision API to detect colored
cubes attached to the bottle:
  - arm_status = 0 (left arm active)  → detect RED / DARK-PINK cube (bottle side)
  - arm_status = 1 (right arm active) → detect YELLOW cube (bottle cap)

Publishes the detected bounding box once to /vision/bbox each time
arm_status changes to one of the above values.

Topics
------
  Subscribes:
    /camera/color/image_raw   (sensor_msgs/Image)
    /arm_status               (std_msgs/Int32)

  Publishes:
    /vision/bbox              (sensor_msgs/RegionOfInterest)
        x_offset = left edge  (pixels)
        y_offset = top edge   (pixels)
        width    = box width  (pixels)
        height   = box height (pixels)

    /vision/bbox_debug        (std_msgs/String)  — JSON with full details

Environment
-----------
  Create  vision_segmentation/.env.segmentation.gemini  and set:
    GEMINI_API_KEY=your_key_here
    GEMINI_VISION_MODEL=gemini-2.0-flash   # optional override

Run
---
  ros2 run tidybot_bringup segment_task3.py
"""

import io
import json
import os
import threading
import time

import numpy as np

# ── dotenv: load API key from our local .env file ──────────────────────────
from dotenv import load_dotenv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_SCRIPT_DIR, '.env.segmentation.gemini')

if os.path.exists(_ENV_FILE):
    load_dotenv(_ENV_FILE)
else:
    # Fallback: check parent directories for a generic .env
    from dotenv import find_dotenv
    load_dotenv(find_dotenv())

# ── Gemini ──────────────────────────────────────────────────────────────────
import google.generativeai as genai
from PIL import Image as PILImage

# ── ROS2 ────────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Int32, String


# ── Constants ───────────────────────────────────────────────────────────────

# Model preference order — first available model is used
VISION_MODEL_CHAIN = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
]

# Cube descriptions sent to Gemini
CUBE_DESCRIPTIONS = {
    0: {
        'label': 'red_cube',
        'description': (
            'a small red or dark-pink colored cube / block '
            'attached to the side of a bottle'
        ),
    },
    1: {
        'label': 'yellow_cube',
        'description': (
            'a small yellow colored cube / block '
            'attached to the cap of a bottle'
        ),
    },
}


# ── Node ────────────────────────────────────────────────────────────────────

class SegmentTask3Node(Node):
    """Detects colored cubes on a bottle using the Gemini Vision API."""

    def __init__(self):
        super().__init__('segment_task3')

        # ── Gemini setup ─────────────────────────────────────────────
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self.get_logger().error(
                'GEMINI_API_KEY not found. '
                f'Add it to {_ENV_FILE}'
            )
            raise RuntimeError('Missing GEMINI_API_KEY')

        genai.configure(api_key=api_key)

        # Override model from env if provided
        self._model_override = os.environ.get('GEMINI_VISION_MODEL')
        self.get_logger().info(
            f'Gemini configured. Model override: {self._model_override or "none (auto)"}'
        )

        # ── State ────────────────────────────────────────────────────
        self._latest_image: Image | None = None
        self._image_lock = threading.Lock()

        self._arm_status: int | None = None   # None = not yet received
        self._detection_lock = threading.Lock()
        self._detecting = False               # prevent overlapping calls

        # ── Publishers ───────────────────────────────────────────────
        self._bbox_pub = self.create_publisher(
            RegionOfInterest, '/vision/bbox', 10
        )
        self._debug_pub = self.create_publisher(
            String, '/vision/bbox_debug', 10
        )

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            Image, '/camera/color/image_raw', self._image_cb, 10
        )
        self.create_subscription(
            Int32, '/arm_status', self._arm_status_cb, 10
        )

        self.get_logger().info('=' * 55)
        self.get_logger().info('  Segment Task3 Node ready')
        self.get_logger().info(f'  Env file: {_ENV_FILE}')
        self.get_logger().info('  Waiting for /arm_status ...')
        self.get_logger().info('=' * 55)

    # ── Callbacks ────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        """Buffer the latest RGB image (thread-safe)."""
        with self._image_lock:
            self._latest_image = msg

    def _arm_status_cb(self, msg: Int32):
        """Trigger detection whenever arm_status changes to 0 or 1."""
        new_status = msg.data

        if new_status not in CUBE_DESCRIPTIONS:
            return  # Ignore unknown values

        if new_status == self._arm_status:
            return  # No change — don't re-trigger

        self.get_logger().info(
            f'arm_status changed: {self._arm_status} → {new_status}'
        )
        self._arm_status = new_status

        # Grab the freshest image right now
        with self._image_lock:
            snapshot = self._latest_image

        if snapshot is None:
            self.get_logger().warn(
                'No image received yet on /camera/color/image_raw — skipping detection.'
            )
            return

        # Run Gemini detection in a background thread (non-blocking)
        with self._detection_lock:
            if self._detecting:
                self.get_logger().warn('Detection already in progress — skipping.')
                return
            self._detecting = True

        t = threading.Thread(
            target=self._run_detection,
            args=(snapshot, new_status),
            daemon=True,
        )
        t.start()

    # ── Detection pipeline ───────────────────────────────────────────

    def _run_detection(self, ros_img: Image, arm_status: int):
        """Worker thread: call Gemini, publish result once."""
        cube_info = CUBE_DESCRIPTIONS[arm_status]
        label = cube_info['label']
        description = cube_info['description']

        self.get_logger().info(
            f'[Detection] arm_status={arm_status} → looking for: {label}'
        )

        try:
            pil_img = self._ros_image_to_pil(ros_img)
            if pil_img is None:
                self.get_logger().error('Failed to convert ROS image — aborting detection.')
                return

            img_w, img_h = pil_img.size

            bbox = self._query_gemini(pil_img, description, img_w, img_h)

            if bbox is None:
                self.get_logger().warn(
                    f'[Detection] Gemini could not locate {label}.'
                )
                self._publish_debug({
                    'target': label,
                    'error': 'not_found',
                    'arm_status': arm_status,
                })
                return

            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            self.get_logger().info(
                f'[Detection] {label} bbox: '
                f'({x_min},{y_min}) → ({x_max},{y_max})  center=({cx},{cy})'
            )

            # Publish RegionOfInterest
            roi = RegionOfInterest()
            roi.x_offset = int(x_min)
            roi.y_offset = int(y_min)
            roi.width    = int(x_max - x_min)
            roi.height   = int(y_max - y_min)
            roi.do_rectify = False
            self._bbox_pub.publish(roi)

            # Publish debug JSON
            self._publish_debug({
                'target': label,
                'arm_status': arm_status,
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max),
                'center_x': int(cx),
                'center_y': int(cy),
                'image_width': img_w,
                'image_height': img_h,
            })

        except Exception as e:
            self.get_logger().error(f'[Detection] Exception: {e}')
        finally:
            with self._detection_lock:
                self._detecting = False

    def _query_gemini(
        self,
        pil_img: PILImage.Image,
        description: str,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int] | None:
        """
        Ask Gemini to locate the object and return pixel bounding box
        (x_min, y_min, x_max, y_max).  Returns None if not found.
        """
        prompt = (
            f'This is an image ({img_w}x{img_h} pixels) from a robot camera.\n'
            f'Locate {description}.\n'
            'Return ONLY a JSON object with the bounding box in pixel coordinates:\n'
            '{"x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>}\n'
            'If the object is not visible, return: {"error": "not_found"}\n'
            'Do NOT include any other text, markdown, or explanation.'
        )

        # Build model preference list
        model_chain = (
            [self._model_override] if self._model_override else VISION_MODEL_CHAIN
        )

        response_text = None

        for model_name in model_chain:
            model = genai.GenerativeModel(model_name)
            for attempt in range(3):
                try:
                    self.get_logger().info(
                        f'  Trying {model_name} (attempt {attempt + 1}/3) ...'
                    )
                    response = model.generate_content([prompt, pil_img])
                    response_text = response.candidates[0].content.parts[0].text.strip()
                    break
                except Exception as e:
                    err = str(e)
                    if '429' in err or 'RESOURCE_EXHAUSTED' in err or 'Resource exhausted' in err:
                        wait = 2 ** attempt
                        self.get_logger().warn(
                            f'  Rate limit on {model_name}, retrying in {wait}s...'
                        )
                        time.sleep(wait)
                    else:
                        self.get_logger().warn(f'  {model_name} error: {e}')
                        break  # try next model

            if response_text is not None:
                break

        if response_text is None:
            self.get_logger().error('All Gemini models failed or were rate-limited.')
            return None

        # Strip markdown fences if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            self.get_logger().warn(
                f'Gemini returned non-JSON: "{response_text}"'
            )
            return None

        if 'error' in data:
            return None

        try:
            x_min = int(data['x_min'])
            y_min = int(data['y_min'])
            x_max = int(data['x_max'])
            y_max = int(data['y_max'])
        except (KeyError, ValueError, TypeError) as e:
            self.get_logger().warn(f'Unexpected JSON keys: {data} — {e}')
            return None

        # Clamp to image bounds
        x_min = max(0, min(x_min, img_w - 1))
        y_min = max(0, min(y_min, img_h - 1))
        x_max = max(0, min(x_max, img_w - 1))
        y_max = max(0, min(y_max, img_h - 1))

        if x_max <= x_min or y_max <= y_min:
            self.get_logger().warn(f'Degenerate bbox from Gemini: {data}')
            return None

        return x_min, y_min, x_max, y_max

    # ── Image conversion ─────────────────────────────────────────────

    @staticmethod
    def _ros_image_to_pil(msg: Image) -> PILImage.Image | None:
        """Convert a sensor_msgs/Image to a PIL Image (RGB)."""
        try:
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            enc = msg.encoding.lower()

            if enc in ('rgb8', 'rgb'):
                arr = data.reshape((msg.height, msg.width, 3))
                return PILImage.fromarray(arr, 'RGB')

            elif enc in ('bgr8', 'bgr'):
                arr = data.reshape((msg.height, msg.width, 3))
                arr = arr[:, :, ::-1]  # BGR → RGB
                return PILImage.fromarray(arr, 'RGB')

            elif enc in ('rgba8', 'rgba'):
                arr = data.reshape((msg.height, msg.width, 4))
                return PILImage.fromarray(arr[:, :, :3], 'RGB')

            elif enc in ('bgra8', 'bgra'):
                arr = data.reshape((msg.height, msg.width, 4))
                arr = arr[:, :, 2::-1]  # BGRA → RGB
                return PILImage.fromarray(arr, 'RGB')

            elif enc in ('mono8', '8uc1'):
                arr = data.reshape((msg.height, msg.width))
                return PILImage.fromarray(arr, 'L').convert('RGB')

            else:
                print(f'[SegmentTask3] Unsupported encoding: {msg.encoding}')
                return None

        except Exception as e:
            print(f'[SegmentTask3] Image conversion error: {e}')
            return None

    # ── Helpers ──────────────────────────────────────────────────────

    def _publish_debug(self, data: dict):
        msg = String()
        msg.data = json.dumps(data)
        self._debug_pub.publish(msg)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SegmentTask3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
