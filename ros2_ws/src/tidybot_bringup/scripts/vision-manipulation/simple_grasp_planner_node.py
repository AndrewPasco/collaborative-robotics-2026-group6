#!/usr/bin/env python3
"""
Simple Centroid-Based Grasp Planner Node

This node provides a heuristic-based grasp planner for the TidyBot2. 
It calculates an object's centroid from a cropped point cloud and generates 
either a top-down or side-approach grasp pose.

The node can either publish the detected pose to a topic or directly call the 
/plan_to_target service to initiate motion planning.

Subscribed Topics:
    - /camera/points (sensor_msgs/PointCloud2): Raw or cropped point cloud data.
    - /grasp_pose_request_roi (sensor_msgs/RegionOfInterest): Trigger for grasp generation.

Published Topics:
    - /detected_grasps/pose (geometry_msgs/PoseStamped): Generated grasp transformed to base_link.
    - /detected_grasps/foreground_cloud (sensor_msgs/PointCloud2): Points identified as the object (for debugging).

Services Called:
    - /plan_to_target (tidybot_msgs/srv/PlanToTarget): Service to request motion planning.
      (Active when send_plan_request is True)

Parameters:
    - base_frame (string): The target frame for the grasp (default: 'base_link').
    - grasp_type (string): Heuristic type: 'top' (top-down) or 'side' (side-approach).
    - table_height_buffer (double): Height offset above the table to filter object points.
    - depth_adjust (double): Y-offset to adjust gripper position relative to centroid.
    - height_adjust (double): Z-offset to adjust gripper height above centroid.
    - send_plan_request (bool): If True, calls /plan_to_target service directly. Default: False.

Example Usage:
    # Launch with side-approach heuristic:
    ros2 run tidybot_bringup simple_grasp_planner_node.py --ros-args -p grasp_type:=side

    # Launch and automatically trigger motion planning:
    ros2 run tidybot_bringup simple_grasp_planner_node.py --ros-args -p send_plan_request:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField, RegionOfInterest
from geometry_msgs.msg import PoseStamped
from tidybot_msgs.srv import PlanToTarget
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R

class SimpleGraspPlannerNode(Node):
    def __init__(self):
        super().__init__("simple_grasp_planner")

        # --- Parameters ---
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("table_height_buffer", 0.01)  # 1cm above table is object
        self.declare_parameter("grasp_type", "top")          # "top" or "side"
        self.declare_parameter("depth_adjust", 0.01)        # how much to set gripper position back
        self.declare_parameter("height_adjust", 0.065)       # how much to raise gripper pose above object
        self.declare_parameter("send_plan_request", False)   # Option to call planner directly
        
        self.base_frame = self.get_parameter("base_frame").value
        
        # --- TF Buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Publishers/Subscribers ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/camera/points", self.cloud_callback, qos
        )
        self.trigger_sub = self.create_subscription(
            RegionOfInterest, "/vision/target_bbox", self.trigger_callback, 10
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/detected_grasps/pose", 10)
        self.rviz_pub = self.create_publisher(PoseStamped, "/detected_grasps/rviz_pose", 10)
        self.foreground_pub = self.create_publisher(PointCloud2, "/detected_grasps/foreground_cloud", 10)

        # --- Service Clients ---
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        self.latest_cloud = None
        self.processing = False
        self.get_logger().info("Simple Grasp Planner Ready (Manual TF). Waiting for ROI on /vision/target_bbox")

    def cloud_callback(self, msg):
        self.latest_cloud = msg
        # self.get_logger().info(f"Latest cloud received with {len(pc2.read_points_numpy(msg, field_names=('x', 'y', 'z')))} points")
        # self.get_logger().info(f"Latest cloud has points with z values: {pc2.read_points_numpy(msg, field_names=('z',))[:,0]}")

    def transform_points(self, cloud_msg, target_frame, roi=None):
        """Manually transform point cloud points to target frame, with optional ROI cropping."""
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                cloud_msg.header.frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return None

        # Create transformation matrix
        t = np.eye(4)
        q = tf.transform.rotation
        x = tf.transform.translation
        t[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t[:3, 3] = [x.x, x.y, x.z]

        # Read points directly into numpy array (faster and avoids casting issues)
        points_raw = pc2.read_points_numpy(cloud_msg, field_names=("x", "y", "z"))
        
        # ROI Cropping (only if cloud is organized)
        if roi is not None and cloud_msg.height > 1:
            h, w = cloud_msg.height, cloud_msg.width
            points_grid = points_raw.reshape((h, w, 3))
            
            y_start = max(0, int(roi.y_offset))
            y_end = min(h, int(roi.y_offset + roi.height))
            x_start = max(0, int(roi.x_offset))
            x_end = min(w, int(roi.x_offset + roi.width))
            
            points_raw = points_grid[y_start:y_end, x_start:x_end, :].reshape(-1, 3)

        # Filter NaNs
        points_raw = points_raw[~np.isnan(points_raw).any(axis=1)]
        
        if len(points_raw) == 0:
            return None
            
        # Homogeneous coordinates
        points_homo = np.ones((len(points_raw), 4))
        points_homo[:, :3] = points_raw
        
        # Apply transform
        points_trans = np.dot(t, points_homo.T).T
        return points_trans[:, :3]

    def trigger_callback(self, roi):
        if self.processing or self.latest_cloud is None:
            return
        
        self.processing = True
        self.get_logger().info(f"Generating simple {self.get_parameter('grasp_type').value} grasp with ROI: "
                               f"x={roi.x_offset}, y={roi.y_offset}, w={roi.width}, h={roi.height}...")

        try:
            # 1. Transform points to base_link manually (with ROI cropping)
            points = self.transform_points(self.latest_cloud, self.base_frame, roi)
            
            if points is None or len(points) < 10:
                self.get_logger().warn("Too few points in cloud.")
                return

            # 2. Foreground Segmentation (Z-Filter relative to table)
            z_heights = points[:, 2]
            # self.get_logger().info(f"z heights: {z_heights}")
            # table_level = np.percentile(z_heights, 10)
            # self.get_logger().info(f"table level: {table_level}")
            buffer = self.get_parameter("table_height_buffer").value
            
            # foreground = points[z_heights > (table_level + buffer)]
            foreground = points[z_heights > (buffer)]
            
            if len(foreground) < 5:
                self.get_logger().warn("No foreground points detected above table level.")
                return

            # 3. Calculate Centroid
            centroid = np.mean(foreground, axis=0)
            self.get_logger().info(f"Object Centroid: {centroid}")

            # --- Debug: Publish Foreground Cloud ---
            foreground_msg = pc2.create_cloud_xyz32(
                self.latest_cloud.header, 
                foreground.tolist()
            )
            foreground_msg.header.frame_id = self.base_frame
            self.foreground_pub.publish(foreground_msg)

            # 4. Apply Heuristics
            grasp_msg = PoseStamped()
            grasp_msg.header.frame_id = self.base_frame
            grasp_msg.header.stamp = self.get_clock().now().to_msg()
            
            grasp_type = self.get_parameter("grasp_type").value

            if grasp_type == "top":
                # Position: Directly above centroid
                grasp_msg.pose.position.x = centroid[0] 
                grasp_msg.pose.position.y = centroid[1] - self.get_parameter("depth_adjust").value
                grasp_msg.pose.position.z = centroid[2] + self.get_parameter("height_adjust").value
                
                q = R.from_euler('xyz', [0, 90, 0], degrees=True).as_quat()
            else: # Side approach
                # Position: Approach from front (X)
                grasp_msg.pose.position.x = centroid[0] - 0.03
                grasp_msg.pose.position.y = centroid[1] - self.get_parameter("depth_adjust").value
                grasp_msg.pose.position.z = centroid[2]
                
                q = R.from_euler('xyz', [0, 0, 180], degrees=True).as_quat()

            grasp_msg.pose.orientation.x = q[0]
            grasp_msg.pose.orientation.y = q[1]
            grasp_msg.pose.orientation.z = q[2]
            grasp_msg.pose.orientation.w = q[3]

            # 5. Output
            self.pose_pub.publish(grasp_msg)
            self.rviz_pub.publish(grasp_msg)
            
            if self.get_parameter("send_plan_request").value:
                self.call_planner(grasp_msg)
            else:
                self.get_logger().info(f"Published Simple Grasp Pose at {centroid}")

        except Exception as e:
            self.get_logger().error(f"Simple planning failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False

    def call_planner(self, grasp_msg):
        """Call the motion planner service."""
        if not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Planner service /plan_to_target not available.")
            return

        req = PlanToTarget.Request()
        req.target_pose = grasp_msg.pose
        req.use_orientation = True
        req.execute = True # Just plan for verification
        req.arm_name = "right" # Default arm
        
        self.get_logger().info("Calling motion planner service...")
        self.plan_client.call_async(req)

def main():
    rclpy.init()
    node = SimpleGraspPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
