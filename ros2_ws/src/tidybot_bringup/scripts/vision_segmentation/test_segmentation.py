#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Int32, String
import cv2
import os
import sys
import json
import threading
import time

class SegmentationTester(Node):
    def __init__(self, image_path, task_number):
        super().__init__('segmentation_tester')
        self.image_path = os.path.abspath(os.path.expanduser(image_path))
        self.task_number = task_number
        
        # Publishers
        self.img_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.arm_pub = self.create_publisher(Int32, '/arm_status', 10)
        
        # Subscribers
        self.bbox_sub = self.create_subscription(RegionOfInterest, '/vision/bbox', self.bbox_cb, 10)
        self.debug_sub = self.create_subscription(String, '/vision/bbox_debug', self.debug_cb, 10)
        
        self.received_bbox = None
        self.get_logger().info(f'Starting test with image: {self.image_path} and task number: {self.task_number}')

    def bbox_cb(self, msg):
        self.get_logger().info(f'RESULT: Received BBox: x={msg.x_offset}, y={msg.y_offset}, w={msg.width}, h={msg.height}')
        self.received_bbox = (msg.x_offset, msg.y_offset, msg.width, msg.height)

    def debug_cb(self, msg):
        data = json.loads(msg.data)
        self.get_logger().info(f'DEBUG: {json.dumps(data, indent=2)}')

    def run_test(self):
        time.sleep(1.0)
        
        if not os.path.exists(self.image_path):
            self.get_logger().error(f'Path does not exist: {self.image_path}')
            return

        img = cv2.imread(self.image_path)
        if img is None:
            self.get_logger().error(f'Failed to load image: {self.image_path}')
            return
            
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        
        self.get_logger().info('Publishing image...')
        self.img_pub.publish(msg)
        time.sleep(1.0)
        
        self.get_logger().info(f'Triggering detection (arm_status={self.task_number})...')
        self.arm_pub.publish(Int32(data=self.task_number))
        
        start_time = time.time()
        while self.received_bbox is None and (time.time() - start_time) < 15.0:
            time.sleep(0.1)
            
        if self.received_bbox is None:
            self.get_logger().error('TIMEOUT: No bounding box received.')
        else:
            # Draw and save result
            x, y, w, h = self.received_bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Detected Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            result_path = os.path.join(os.path.dirname(self.image_path), 'test_result.jpg')
            cv2.imwrite(result_path, img)
            self.get_logger().info(f'SUCCESS: Bounding box drawn and saved to {result_path}')

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 test_segmentation.py <path_to_image> <task_number>")
        return
        
    rclpy.init()
    tester = SegmentationTester(sys.argv[1], int(sys.argv[2]))
    
    # Run ROS2 spinning in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(tester,), daemon=True)
    thread.start()
    
    try:
        tester.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
