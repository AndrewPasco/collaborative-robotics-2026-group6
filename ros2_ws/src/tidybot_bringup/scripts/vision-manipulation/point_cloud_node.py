#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np

class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_node')

        # Subscribe to the color camera info since the depth is aligned to the color frame
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.info_cb,
            10
        )
        
        # Subscribe to the aligned depth image
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_cb,
            10
        )

        self.nav_msg_sub = self.create_subscription(
            String,
            '/brain/navigation_status',
            self.start_cb,
            10
        )

        self.manip_done_sub = self.create_subscription(
            String,
            'brain/manipulation_status',
            self.end_cb,
            10
        )
        
        # Publisher for the PointCloud2
        self.pc_pub = self.create_publisher(
            PointCloud2,
            '/camera/points',
            10
        )

        self.intrinsics = None
        self.sending = False
        
        # We will cache the meshgrid to save computation time during callbacks
        self.u = None
        self.v = None

    def info_cb(self, msg):
        if self.intrinsics is None:
            # K is a 1D array of length 9: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.intrinsics = {
                'fx': msg.k[0],
                'cx': msg.k[2],
                'fy': msg.k[4],
                'cy': msg.k[5]
            }
            self.get_logger().info('Camera intrinsics received.')
            # We only need the intrinsics once, so we can destroy the subscription to save bandwidth
            self.destroy_subscription(self.info_sub)

    def start_cb(self, msg):
        if msg.data == "final_approach":
            self.sending = True

    def end_cb(self, msg):
        if msg.data == "done":
            self.sending = False
        

    def depth_cb(self, msg):
        # if not self.sending:
        #     return
        
        if self.intrinsics is None:
            return

        # RealSense depth images are typically 16-bit unsigned integers denoting millimeters
        if msg.encoding != '16UC1':
            self.get_logger().error(f'Expected 16UC1 encoding, got {msg.encoding}')
            return

        # Convert raw ROS Image buffer to a 2D numpy array
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

        # Initialize the pixel meshgrid only once (or if dimensions change)
        if self.u is None or self.v is None or self.u.shape != (msg.height * msg.width,):
            v, u = np.indices((msg.height, msg.width))
            self.u = u.flatten()
            self.v = v.flatten()

        z = depth_image.flatten()

        # Filter out 0-depth values (invalid pixels) to dramatically reduce point cloud size
        valid_mask = (z > 0) & (z < 1000)
        z_valid = z[valid_mask]
        u_valid = self.u[valid_mask]
        v_valid = self.v[valid_mask]

        # Convert depth from mm to meters
        z_meters = z_valid / 1000.0

        # Back-project pixels to 3D space (Z is depth, X is right, Y is down)
        x = (u_valid - self.intrinsics['cx']) * z_meters / self.intrinsics['fx']
        y = (v_valid - self.intrinsics['cy']) * z_meters / self.intrinsics['fy']

        # Stack into an N x 3 array
        points = np.vstack((x, y, z_meters)).T

        # Create PointCloud2 message
        # Passing `msg.header` maintains the exact timestamp and frame_id (e.g., camera_color_optical_frame)
        pc_msg = point_cloud2.create_cloud_xyz32(msg.header, points)
        self.pc_pub.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()