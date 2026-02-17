import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Int64
from gpd_ros2_msgs.srv import DetectConstrainedGrasps
from geometry_msgs.msg import Point


class GraspTriggerNode(Node):
    def __init__(self):
        super().__init__("grasp_trigger_node")

        # 1. Subscribe to the continuous point cloud stream from depth_image_proc, started in gpd_bridge.launch.py (Input Data)
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/camera/points", self.cloud_callback, 1
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
            DetectConstrainedGrasps, "detect_constrained_grasps"
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
        # Prepare Request
        req = DetectConstrainedGrasps.Request()
        req.cloud_indexed.cloud_sources.cloud = self.latest_cloud

        # Viewpoint (Camera at 0,0,0 relative to cloud)
        req.cloud_indexed.cloud_sources.view_points = [Point(x=0.0, y=0.0, z=0.0)]

        # Sample Indices (Every 10th point to speed up detection)
        # Note: GPD might require indices to be unique and sorted
        num_points = self.latest_cloud.width * self.latest_cloud.height
        indices = [Int64(data=i) for i in range(0, num_points, 10)]
        req.cloud_indexed.indices = indices

        # Use Config File Params
        req.params_policy = DetectConstrainedGrasps.Request.USE_CFG_FILE

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
