#!/usr/bin/env python3
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, RegionOfInterest
from geometry_msgs.msg import PoseStamped
from tidybot_msgs.srv import PlanToTarget
import struct
import math
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

# To force CPU usage if no GPU is available
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add Contact-GraspNet paths
# The node is in ros2_ws/src/tidybot_bringup/scripts/vision-manipulation
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if "install" in SCRIPT_DIR:
    # Running from installed location: lib/tidybot_bringup/contact_graspnet_node.py
    # Submodule should be at: lib/tidybot_bringup/contact_graspnet/
    CG_ROOT = os.path.join(SCRIPT_DIR, "contact_graspnet")
else:
    # Running from source: src/tidybot_bringup/scripts/vision-manipulation/contact_graspnet_node.py
    CG_ROOT = os.path.join(SCRIPT_DIR, "contact_graspnet")

sys.path.append(CG_ROOT)
sys.path.append(os.path.join(CG_ROOT, "contact_graspnet"))

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    import config_utils
    from contact_grasp_estimator import GraspEstimator
except ImportError as e:
    print(f"Error: Missing dependencies for Contact-GraspNet: {e}")
    # We will still define the class to avoid syntax errors, but it won't be functional without TF
    GraspEstimator = None

class ContactGraspNetNode(Node):
    def __init__(self):
        super().__init__("contact_graspnet_node")

        # --- Parameters ---
        self.declare_parameter("ckpt_dir", os.path.join(CG_ROOT, "checkpoints/scene_test_2048_bs3_hor_sigma_001"))
        self.declare_parameter("pose_topic", "/detected_grasps/pose")
        self.declare_parameter("rviz_topic", "/detected_grasps/rviz_pose")
        self.declare_parameter("use_detected_orientation", True)
        self.declare_parameter("send_plan_request", False)
        self.declare_parameter("forward_passes", 1)
        self.declare_parameter("z_range", [0.1, 1.5])
        self.declare_parameter("threshold", 0.2) 
        
        # WidowX WX250s depth from wrist to finger center is ~0.065m
        # The model assumes 0.1034m (Panda). 
        # Difference (~0.038m) is why grasps appear "above" the object.
        self.declare_parameter("gripper_depth", 0.065) 

        ckpt_dir = self.get_parameter("ckpt_dir").get_parameter_value().string_value
        self.forward_passes = self.get_parameter("forward_passes").get_parameter_value().integer_value
        self.z_range = self.get_parameter("z_range").get_parameter_value().double_array_value
        self.actual_depth = self.get_parameter("gripper_depth").get_parameter_value().double_value
        self.model_depth = 0.1034 # Constant from Contact-GraspNet Panda model

        if GraspEstimator is None:
            self.get_logger().error("Contact-GraspNet dependencies (TensorFlow) not found. Please install them and compile PointNet2 ops.")
            return

        # --- Load Model ---
        self.get_logger().info(f"Loading Contact-GraspNet model from {ckpt_dir}...")
        try:
            # Load config
            # IMPORTANT: Force batch_size=1 for inference
            config_path = os.path.join(CG_ROOT, "contact_graspnet/config.yaml")
            self.global_config = config_utils.load_config(os.path.dirname(config_path), batch_size=1)
            
            # Override thresholds for inference quality
            thresh = self.get_parameter("threshold").get_parameter_value().double_value
            self.global_config['TEST']['first_thres'] = thresh
            self.global_config['TEST']['second_thres'] = thresh
            self.global_config['DATA']['use_farthest_point'] = False # Faster inference
            
            # Build estimator
            self.grasp_estimator = GraspEstimator(self.global_config)
            self.grasp_estimator.build_network()

            # Create TF session
            self.saver = tf.train.Saver(save_relative_paths=True)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)

            # Load weights
            self.grasp_estimator.load_weights(self.sess, self.saver, ckpt_dir, mode='test')
            self.get_logger().info("Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load Contact-GraspNet model: {e}")
            return

        # --- Subscribers/Publishers ---
        # Use BEST_EFFORT to be compatible with both Reliable and Best Effort publishers
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/camera/points", self.cloud_callback, qos_profile
        )
        self.trigger_sub = self.create_subscription(
            RegionOfInterest, "/grasp_pose_request_roi", self.trigger_callback, 1
        )
        self.pose_pub = self.create_publisher(PoseStamped, self.get_parameter("pose_topic").get_parameter_value().string_value, 10)
        self.rviz_pose_pub = self.create_publisher(PoseStamped, self.get_parameter("rviz_topic").get_parameter_value().string_value, 10)

        # --- TF Buffer and Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Service Clients ---
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')
        
        self.latest_cloud = None
        self.processing = False
        self.get_logger().info("Contact-GraspNet Node Ready.")

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    def trigger_callback(self, roi_msg):
        if self.processing: return
        if self.latest_cloud is None:
            self.get_logger().warn("No cloud received yet.")
            return
            
        self.processing = True
        self.get_logger().info(f"Processing ROI request for Contact-GraspNet...")
        
        try:
            best_grasp = self.process_cloud(self.latest_cloud, roi_msg)
            if best_grasp is not None:
                # best_grasp is a 4x4 matrix
                grasp_msg = self.publish_grasp(best_grasp, self.latest_cloud.header)
                self.call_planner(grasp_msg)
        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False

    def process_cloud(self, cloud_msg, roi=None):
        # 1. Convert ROS PointCloud2 to numpy (N, 3)
        points = self.read_points_numpy(cloud_msg, roi)
        if points is None or len(points) == 0:
            self.get_logger().warn("Empty or invalid point cloud in ROI.")
            return None
            
        self.get_logger().info(f"Got {len(points)} valid points. Running inference...")
        
        # Contact-GraspNet expects pc_full and pc_segments
        # For ROI, we treat the whole thing as one segment or just pc_full
        pc_full = points
        pc_segments = {} # We can use segments if we have a segmap, but ROI crop is enough
        
        # Inference
        # pred_grasps_cam: {seg_id: [N, 4, 4]}, scores: {seg_id: [N]}, contact_pts: {seg_id: [N, 3]}
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess, pc_full, pc_segments=pc_segments, 
            local_regions=False, filter_grasps=False, forward_passes=self.forward_passes
        )
        
        # Extract best grasp from all results
        best_score = -1
        best_grasp = None
        
        for seg_id in scores:
            all_scores = scores[seg_id]
            if len(all_scores) > 0:
                self.get_logger().info(f"Detected {len(all_scores)} grasps in segment {seg_id}. Max score: {np.max(all_scores):.4f}")
                idx = np.argmax(all_scores)
                if all_scores[idx] > best_score:
                    best_score = all_scores[idx]
                    best_grasp = pred_grasps_cam[seg_id][idx]
        
        if best_grasp is None:
            self.get_logger().warn("No grasps detected above threshold.")
            return None
            
        self.get_logger().info(f"Best grasp score: {best_score:.4f}")
        return best_grasp

    def read_points_numpy(self, cloud_msg, roi=None):
        # Decode PointCloud2 (simplified for brevity, similar to pointnet_gpd_node)
        point_step = cloud_msg.point_step
        row_step = cloud_msg.row_step
        width = cloud_msg.width
        height = cloud_msg.height
        data = cloud_msg.data
        
        points = []
        fmt = 'fff'
        unpack = struct.Struct(fmt).unpack_from
        
        u_min = max(0, roi.x_offset) if roi else 0
        v_min = max(0, roi.y_offset) if roi else 0
        u_max = min(width, roi.x_offset + roi.width) if roi else width
        v_max = min(height, roi.y_offset + roi.height) if roi else height

        for v in range(v_min, v_max):
            row_offset = v * row_step
            for u in range(u_min, u_max):
                off = row_offset + u * point_step
                x, y, z = unpack(data, off)
                if math.isnan(x) or math.isnan(y) or math.isnan(z): continue
                if (self.z_range[0] < z < self.z_range[1]):
                    points.append([x, y, z])
                
        return np.array(points, dtype=np.float32)

    def publish_grasp(self, grasp_matrix, header):
        # grasp_matrix is 4x4
        # Original Contact-GraspNet grasp pose is at the gripper base (Panda).
        
        translation = grasp_matrix[:3, 3]
        rotation_matrix = grasp_matrix[:3, :3]
        
        # Correction: The model assumes a deeper gripper (model_depth).
        # We need to shift the pose forward along the approach axis (Z-axis of gripper)
        # to match our shallower gripper (actual_depth).
        # approach_vector is the 3rd column of the rotation matrix.
        approach_vector = rotation_matrix[:, 2]
        depth_offset = self.model_depth - self.actual_depth
        
        # Shift translation forward
        corrected_translation = translation + depth_offset * approach_vector
        
        quat = R.from_matrix(rotation_matrix).as_quat()
        
        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = float(corrected_translation[0])
        msg.pose.position.y = float(corrected_translation[1])
        msg.pose.position.z = float(corrected_translation[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        self.get_logger().info(f"Original Pos: {translation}, Corrected: {corrected_translation}")
        return msg

    def call_planner(self, grasp_msg):
        # Transform and call planner (similar logic to pointnet_gpd_node)
        target_frame = "base_link"
        try:
            transformed_pose_msg = self.tf_buffer.transform(
                grasp_msg, target_frame, timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f"Transform failed: {e}")
            return

        # Contact-GraspNet uses a different gripper convention than PointNetGPD
        # We might need to adjust the orientation here based on WidowX gripper
        
        send_plan_request = self.get_parameter("send_plan_request").get_parameter_value().bool_value
        if send_plan_request:
            req = PlanToTarget.Request()
            req.target_pose = transformed_pose_msg.pose
            # ... fill rest of request
            self.plan_client.call_async(req)
        else:
            self.pose_pub.publish(transformed_pose_msg)
            self.rviz_pose_pub.publish(transformed_pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ContactGraspNetNode()
    if GraspEstimator is not None:
        rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
