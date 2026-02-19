#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from gpd_ros2_interfaces.srv import DetectGrasps
from gpd_ros2_interfaces.msg import CloudIndexed, CloudSources
from geometry_msgs.msg import Point


class GraspTriggerNode(Node):
    def __init__(self):
        super().__init__("grasp_trigger_node")

        # Setup QoS for high-bandwidth point cloud data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        # 1. Subscribe to the continuous point cloud stream from depth_image_proc, started in gpd_bridge.launch.py (Input Data)
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/camera/points", self.cloud_callback, qos_profile
        )
        self.latest_cloud = None
        self.get_logger().info("Waiting for point cloud on /camera/points...")

        # 2. Subscribe to the Trigger Topic (The "Go" Signal)
        self.trigger_sub = self.create_subscription(
            Bool, "/grasp_pose_request", self.trigger_callback, 1
        )
        self.get_logger().info("Listening for requests on /grasp_pose_request...")

        # 3. Setup the GPD Service Client
        self.gpd_client = self.create_client(
            DetectGrasps, "detect_grasps"
        )
        while not self.gpd_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for GPD service...")
        self.get_logger().info("GPD Service Connected!")

    def cloud_callback(self, msg):
        """Always keep the latest cloud in memory."""
        self.latest_cloud = msg

    def trigger_callback(self, msg):
        """Called when someone publishes to /grasp_pose_request"""
        if not msg.data:
            return  # Ignore False messages if you want

        if self.latest_cloud is None:
            self.get_logger().warn(
                "Received request, but NO point cloud data available yet!"
            )
            return

        self.get_logger().info("Received request! Sending cloud to GPD...")
        self.call_gpd_service()

    def call_gpd_service(self):
        import struct
        import math

        # Prepare Request
        req = DetectGrasps.Request()
        
        # 1. Validate Point Cloud
        num_points = self.latest_cloud.width * self.latest_cloud.height
        if num_points == 0:
            self.get_logger().error("Point cloud is empty!")
            return

        self.get_logger().info(f"Preparing GPD request with {num_points} points...")

        # 2. Extract valid indices (points that are not NaN and within a small workspace)
        # Workspace crop for debugging: x[-0.1, 0.1], y[-0.1, 0.1], z[0.2, 0.8]
        fmt = "f" # float
        valid_indices = []
        point_step = self.latest_cloud.point_step
        
        for i in range(0, num_points):
            offset = i * point_step
            # Read x, y, z coordinates
            x_bytes = self.latest_cloud.data[offset + 0 : offset + 4]
            y_bytes = self.latest_cloud.data[offset + 4 : offset + 8]
            z_bytes = self.latest_cloud.data[offset + 8 : offset + 12]
            
            x = struct.unpack(fmt, x_bytes)[0]
            y = struct.unpack(fmt, y_bytes)[0]
            z = struct.unpack(fmt, z_bytes)[0]
            
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            # Crop box
            if -0.1 < x < 0.1 and -0.1 < y < 0.1 and 0.3 < z < 0.7:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            self.get_logger().warn("No valid points found in the crop box (x[-0.1, 0.1], y[-0.1, 0.1], z[0.3, 0.7])!")
            return

        # Limit to max 500 samples
        if len(valid_indices) > 500:
            stride = len(valid_indices) // 500
            valid_indices = valid_indices[::stride][:500]

        self.get_logger().info(f"Found {len(valid_indices)} points in crop box. Sending to GPD...")

        # Fill CloudSources
        sources = CloudSources()
        sources.cloud = self.latest_cloud
        sources.view_points = [Point(x=0.0, y=0.0, z=0.0)]
        sources.camera_source = [0] * num_points

        # Fill CloudIndexed
        indexed = CloudIndexed()
        indexed.cloud_sources = sources
        indexed.indices = valid_indices
        
        req.cloud_indexed = indexed

        # Async Service Call
        future = self.gpd_client.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()
            grasps = response.grasp_configs.grasps
            self.get_logger().info(f"SUCCESS: Received {len(grasps)} grasps")

            # Optional: Publish the best grasp to another topic for your robot to use
            # self.grasp_pub.publish(best_grasp)

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GraspTriggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
