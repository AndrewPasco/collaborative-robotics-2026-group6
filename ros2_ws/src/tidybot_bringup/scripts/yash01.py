#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np
import cv2

from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge

import tf2_ros
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped


class DepthToPointCloud(Node):

    def __init__(self):
        super().__init__('depth_to_pointcloud')

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.bridge = CvBridge()

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            qos)

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            qos)

        # Publisher
        self.pc_pub = self.create_publisher(
            PointCloud2,
            '/camera/pointcloud_base',
            10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_rgb = None
        self.latest_depth = None

        self.rgb_K = (
            625.2213755265124,
            625.2213755265124,
            320.0,
            240.0
        )
        self.depth_K = (
            625.2213755265124,
            625.2213755265124,
            320.0,
            240.0
        )

        self.cam2cam_transform = np.eye(4)

        self.get_logger().info("Depth → PointCloud node started")

    # --------------------------------------------------

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.get_logger().info("Getting camera data")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process()

    # --------------------------------------------------

    def align_depth(self, depth, rgb):

        old_fx, old_fy, old_cx, old_cy = self.depth_K
        new_fx, new_fy, new_cx, new_cy = self.rgb_K

        K_old = np.array([[old_fx, 0, old_cx],
                          [0, old_fy, old_cy],
                          [0, 0, 1]])

        K_new = np.array([[new_fx, 0, new_cx],
                          [0, new_fy, new_cy],
                          [0, 0, 1]])

        width, height = rgb.shape[1], rgb.shape[0]

        K_new_inv = np.linalg.inv(K_new)

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        homog = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=-1).T

        old_coords = K_old @ K_new_inv @ homog
        old_coords /= old_coords[2, :]

        map_x = old_coords[0].reshape(height, width).astype(np.float32)
        map_y = old_coords[1].reshape(height, width).astype(np.float32)

        depth_aligned = cv2.remap(depth, map_x, map_y, interpolation=cv2.INTER_NEAREST)

        return depth_aligned

    # --------------------------------------------------

    def process(self):
        self.get_logger().info("Depth → Pointsadsrted")

        print("sdsadasdsa")

        if self.latest_rgb is None or self.latest_depth is None:
            return

        rgb = self.latest_rgb
        depth = self.latest_depth.astype(np.float32)

        depth = self.align_depth(depth, rgb)

        fx, fy, cx, cy = self.rgb_K

        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        z = depth / 1000.0  # convert mm to meters if needed

        valid = z > 0

        u = u[valid]
        v = v[valid]
        z = z[valid]

        # --------------------------------------------------
        # Back-project to camera_link
        # camera_link: x forward, y left, z up
        # --------------------------------------------------

        x_cam = z
        y_cam = -(u - cx) * z / fx
        z_cam = -(v - cy) * z / fy

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # --------------------------------------------------
        # Transform to base_link
        # --------------------------------------------------

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_link',
                rclpy.time.Time())

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        trans = transform.transform.translation
        rot = transform.transform.rotation

        rotation = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        T = np.eye(4)
        T[:3, :3] = rotation.as_matrix()
        T[0, 3] = trans.x
        T[1, 3] = trans.y
        T[2, 3] = trans.z

        points_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_base = (T @ points_h.T).T[:, :3]

        # --------------------------------------------------
        # Publish PointCloud2
        # --------------------------------------------------

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        cloud_msg = point_cloud2.create_cloud_xyz32(header, points_base)

        self.pc_pub.publish(cloud_msg)

        self.get_logger().info(f"Published {points_base.shape[0]} points")


def main(args=None):
    print("Started")
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()