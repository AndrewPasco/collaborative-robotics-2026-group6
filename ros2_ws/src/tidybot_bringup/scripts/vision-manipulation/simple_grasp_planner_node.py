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
from geometry_msgs.msg import PoseStamped, Point
from tidybot_msgs.srv import PlanToTarget
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker

class SimpleGraspPlannerNode(Node):
    def __init__(self):
        super().__init__("simple_grasp_planner")

        # --- Parameters ---
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("table_height_buffer", 0.01)  # 1cm above table is object
        self.declare_parameter("grasp_type", "top")          # "top" or "side"
        self.declare_parameter("depth_adjust", 0.00)        # how much to set gripper position back
        self.declare_parameter("height_adjust", 0.04)       # how much to raise gripper pose above object
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
        self.rviz_pub = self.create_publisher(PoseStamped, "/detected_grasps/rviz_pose_new", 10)
        self.foreground_pub = self.create_publisher(PointCloud2, "/detected_grasps/foreground_cloud", 10)
        self.axis_marker_pub = self.create_publisher(Marker, "/detected_grasps/pca_axis", 10)

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
            self.get_logger().info("Not executing")
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
            
            # ==========================================
            # DEBUG: Generate and save Z-value heatmap
            # ==========================================
            # Extract X, Y, and Z columns
            x = points[:, 0]
            y = points[:, 1]
            z_heights = points[:, 2]

            plt.figure(figsize=(8, 6))
            # Scatter plot acting as a top-down heatmap. 'c' maps the colors to Z values.
            scatter = plt.scatter(x, y, c=z_heights, cmap='viridis', s=5, alpha=0.8)
            plt.colorbar(scatter, label='Z Height (meters)')
            plt.xlabel(f'X in {self.base_frame} (m)')
            plt.ylabel(f'Y in {self.base_frame} (m)')
            plt.title('Top-Down Heatmap of Z-Values')
            plt.axis('equal') # Ensures spatial proportions aren't distorted
            
            # Save the plot to the /tmp directory
            heatmap_path = '/tmp/grasp_z_heatmap.png'
            plt.savefig(heatmap_path)
            plt.close() # Free up memory
            self.get_logger().info(f"Saved Z-heatmap to {heatmap_path}")
            # ==========================================

            # 2. Foreground Segmentation (Z-Filter relative to table)
            z_heights = points[:, 2]
            # self.get_logger().info(f"z heights: {z_heights}")
            table_level = np.percentile(z_heights, 10)
            self.get_logger().info(f"table level: {table_level}")
            buffer = self.get_parameter("table_height_buffer").value
            
            foreground = points[z_heights > (table_level + buffer)]
            # foreground = points[z_heights > (buffer)]
            
            if len(foreground) < 5:
                self.get_logger().warn("No foreground points detected above table level.")
                return

            # 3. Calculate Centroid
            centroid = np.mean(foreground, axis=0)
            self.get_logger().info(f"Object Centroid: {centroid}")

            # Calculate Yaw via 2D PCA
            # Extract just the X and Y columns
            xy_points = foreground[:, :2]
            
            # Mean center the points
            centered_xy = xy_points - centroid[:2]
            
            # Calculate the covariance matrix
            cov_matrix = np.cov(centered_xy, rowvar=False)
            
            # Calculate eigenvectors
            _, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # np.linalg.eigh sorts eigenvalues in ascending order.
            # The last column is the eigenvector for the largest eigenvalue (Major Axis)
            major_axis = eigenvectors[:, 1]
            
            # Calculate the yaw angle in radians
            yaw_rad = np.arctan2(major_axis[1], major_axis[0])
            self.get_logger().info(f"Calculated PCA Yaw: {np.degrees(yaw_rad):.2f} degrees")

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
                
                base_rot = R.from_euler('xyz', [0, 90, 0], degrees=True)

                yaw = R.from_euler('x', yaw_rad, degrees=False)

                # Publish RViz Arrow Marker for Major Axis
                marker = Marker()
                marker.header.frame_id = self.base_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "pca_axis"
                marker.id = 0
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                # Start point (Centroid)
                p1 = Point()
                p1.x = float(centroid[0])
                p1.y = float(centroid[1])
                p1.z = float(centroid[2])

                # End point (Projected 10cm along the 2D major axis)
                arrow_length = 0.10 # 10 cm
                p2 = Point()
                p2.x = float(centroid[0] + major_axis[0] * arrow_length)
                p2.y = float(centroid[1] + major_axis[1] * arrow_length)
                p2.z = float(centroid[2]) # Keep it perfectly flat relative to the table

                marker.points = [p1, p2]

                # For ARROW markers using 'points':
                # scale.x is shaft diameter, scale.y is head diameter, scale.z is head length
                marker.scale.x = 0.005 
                marker.scale.y = 0.015 
                marker.scale.z = 0.02 

                # Make it bright red and fully opaque
                marker.color.r = 1.0 
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0 

                self.axis_marker_pub.publish(marker)
                # ==========================================
                
                # Combine (apply yaw to the base rotation)
                final_rot = base_rot * yaw
                q = final_rot.as_quat()

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
