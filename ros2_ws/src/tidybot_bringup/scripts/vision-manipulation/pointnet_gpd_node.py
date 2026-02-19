#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, RegionOfInterest
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Bool
from tidybot_msgs.srv import PlanToTarget
import struct
import math
import tf2_ros
import tf2_geometry_msgs

# Add submodule paths
# The node is in ros2_ws/src/tidybot_bringup/scripts/vision-manipulation
# The vendored code is in ros2_ws/src/third_party/pointnet_gpd
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.environ.get("TIDYBOT_REPO_ROOT")

if REPO_ROOT:
    POINTNET_ROOT = os.path.join(REPO_ROOT, "ros2_ws/src/third_party/pointnet_gpd")
elif "install" in SCRIPT_DIR:
    # We are in install/tidybot_bringup/lib/tidybot_bringup
    # Path to src: ../../../../src
    POINTNET_ROOT = os.path.join(SCRIPT_DIR, "../../../../src/third_party/pointnet_gpd")
else:
    # We are in src/tidybot_bringup/scripts/vision-manipulation
    POINTNET_ROOT = os.path.join(SCRIPT_DIR, "../../../third_party/pointnet_gpd")

POINTNET_ROOT = os.path.abspath(POINTNET_ROOT)

# Explicitly add source paths to sys.path to bypass installation issues
sys.path.append(os.path.join(POINTNET_ROOT, "meshpy"))
sys.path.append(os.path.join(POINTNET_ROOT, "dex-net/src"))
sys.path.append(os.path.join(POINTNET_ROOT, "PointNetGPD")) # For main_test.py / model logic if needed

# dexnet imports
try:
    from dexnet.grasping import GpgGraspSamplerPcl, RobotGripper
    from autolab_core import YamlConfig
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to import dexnet: {e}")
    sys.exit(1)

# PointNetGPD imports
# We can't import main_test directly because it parses args on import.
# We'll replicate the model loading and inference logic here.

class PointNetGPDNode(Node):
    def __init__(self):
        super().__init__("pointnet_gpd_node")
        
        # --- Parameters ---
        self.declare_parameter("model_path", os.path.join(POINTNET_ROOT, "data/pointnetgpd_3class.model"))
        self.declare_parameter("config_path", os.path.join(POINTNET_ROOT, "dex-net/test/config.yaml"))
        self.declare_parameter("gripper_dir", os.path.join(POINTNET_ROOT, "dex-net/data/grippers"))
        self.declare_parameter("rviz_topic", "/detected_grasps/pose")
        
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        gripper_dir = self.get_parameter("gripper_dir").get_parameter_value().string_value
        
        # --- Load Model ---
        self.get_logger().info(f"Loading model from {model_path}...")
        self.device = torch.device("cpu") # Force CPU for now
        try:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Unwrap DataParallel if present (common in multi-gpu trained models)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
                
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

        # --- Load Gripper & Sampler ---
        self.get_logger().info("Loading gripper and sampler...")
        try:
            yaml_config = YamlConfig(config_path)
            # Override some config values for CPU/speed
            yaml_config['grasp_samples_per_surface_point'] = 2 # Reduced from default
            yaml_config['min_contact_dist'] = 0.005
            
            # Use WidowX gripper config
            gripper = RobotGripper.load("widowx", gripper_dir)
            self.sampler = GpgGraspSamplerPcl(gripper, yaml_config)
            self.get_logger().info("Sampler loaded!")
        except Exception as e:
            self.get_logger().error(f"Failed to load sampler: {e}")
            sys.exit(1)

        # --- Subscribers/Publishers ---
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/camera/points", self.cloud_callback, qos_profile
        )
        self.trigger_sub = self.create_subscription(
            RegionOfInterest, "/grasp_pose_request_roi", self.trigger_callback, 1
        )
        self.pose_pub = self.create_publisher(PoseStamped, self.get_parameter("rviz_topic").get_parameter_value().string_value, 10)
        
        # --- TF Buffer and Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Service Clients ---
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')
        self.get_logger().info("Waiting for /plan_to_target service...")
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting...")
        self.get_logger().info("Planner service connected!")

        self.latest_cloud = None
        self.processing = False
        
        self.get_logger().info("PointNetGPD Node Ready. Waiting for ROI on /grasp_pose_request_roi...")

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    def trigger_callback(self, roi_msg):
        if self.processing: return
        if self.latest_cloud is None:
            self.get_logger().warn("No cloud received yet.")
            return
            
        self.processing = True
        self.get_logger().info(f"Processing ROI request: x={roi_msg.x_offset}, y={roi_msg.y_offset}, w={roi_msg.width}, h={roi_msg.height}")
        
        try:
            best_grasp = self.process_cloud(self.latest_cloud, roi_msg)
            if best_grasp:
                # 1. Publish to RViz (Original Frame)
                grasp_msg = self.publish_grasp(best_grasp, self.latest_cloud.header)
                
                # 2. Call Planner (Transformed to base_link)
                self.call_planner(grasp_msg)
        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False

    def call_planner(self, grasp_msg):
        """Call the motion planner service with the detected grasp."""
        
        # 1. Transform pose to base_link
        target_frame = "base_link"
        try:
            # transform returns a PoseStamped
            transformed_pose_msg = self.tf_buffer.transform(
                grasp_msg, target_frame, timeout=rclpy.duration.Duration(seconds=1.0)
            )
            self.get_logger().info(f"Transformed grasp to {target_frame}")
        except Exception as e:
            self.get_logger().error(f"Could not transform grasp from {grasp_msg.header.frame_id} to {target_frame}: {e}")
            return

        # 2. Prepare and send request
        req = PlanToTarget.Request()
        req.arm_name = "right" # Default to right arm
        
        # Default to a top-down grasp orientation (fingers pointing down)
        # In base_link frame, this corresponds to:
        # qw=0.5, qx=0.5, qy=0.5, qz=-0.5
        transformed_pose_msg.pose.orientation.x = 0.5
        transformed_pose_msg.pose.orientation.y = 0.5
        transformed_pose_msg.pose.orientation.z = -0.5
        transformed_pose_msg.pose.orientation.w = 0.5
        
        req.target_pose = transformed_pose_msg.pose
        req.use_orientation = True
        req.execute = True
        req.duration = 2.0
        
        self.get_logger().info(f"Sending plan request to {req.arm_name} arm (using top-down orientation)...")
        
        # Call asynchronously
        future = self.plan_client.call_async(req)
        future.add_done_callback(self.planner_response_callback)

    def planner_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"SUCCESS: {response.message}")
            else:
                self.get_logger().warn(f"FAILED: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def process_cloud(self, cloud_msg, roi=None):
        # 1. Convert ROS PointCloud2 to numpy (N, 3)
        points = self.read_points_numpy(cloud_msg, roi)
        if points is None or len(points) == 0:
            self.get_logger().warn("Empty or invalid point cloud in ROI.")
            return None
            
        self.get_logger().info(f"Got {len(points)} valid points. Downsampling...")
        
        # 2. Downsample (simple voxel/stride)
        # Target ~2000 points for the sampler
        if len(points) > 2000:
            indices = np.random.choice(len(points), 2000, replace=False)
            points = points[indices]
            
        # 3. Sample Grasps (Geometric)
        # GpgGraspSamplerPcl.sample_grasps(point_cloud, points_for_sample, all_normal, ...)
        # Note: The sampler expects Open3D style inputs or raw arrays?
        # The python code we read uses: `all_points = point_cloud.to_array()` if it's a PCL object
        # BUT GpgGraspSamplerPcl implementation we read seems to expect numpy arrays if we tweak it,
        # or maybe we need to wrap it.
        # Let's look at the sampler code again: 
        # def sample_grasps(self, point_cloud, points_for_sample, all_normal, ...)
        # It uses o3d inside.
        
        # We need normals! The sampler calculates them internally using Open3D? 
        # Actually `cal_grasp` in kinect2grasp.py computed normals using PCL before calling sampler.
        # But `GpgGraspSamplerPcl.sample_grasps` in `grasp_sampler.py` ALSO seems to calculate normals?
        # Wait, the `GpgGraspSamplerPcl.sample_grasps` signature is:
        # (self, point_cloud, points_for_sample, all_normal, ...)
        # So we MUST compute normals first.
        
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Extract normals
        normals = np.asarray(pcd.normals)
        
        # Flip normals to point towards camera (0,0,0)
        # vector_p2cam = cam_pos - points = -points (since cam_pos is origin)
        view_dirs = -points
        
        # Check alignment: dot(normal, view_dir)
        # If dot > 0: normal faces camera (correct)
        # If dot < 0: normal faces away (flip)
        dots = np.sum(normals * view_dirs, axis=1)
        mask = dots < 0
        normals[mask] = -normals[mask]
        
        self.get_logger().info(f"Flipped {np.sum(mask)} normals to face camera.")
        
        # 4. Generate Candidates
        self.get_logger().info("Generating candidates...")
        # NOTE: passing raw numpy array as 'point_cloud' because we saw `point_cloud.to_array()` usage
        # We might need to mock a class or pass array if `to_array` is handled.
        # Let's assume we can modify the sampler or it handles it. 
        # Actually `dexnet` might expect a specific object.
        # Let's pass the points as numpy array and hope the sampler handles it 
        # (The implementation we read: `all_points = point_cloud.to_array()`)
        # We can make a dummy wrapper.
        
        class CloudWrapper:
            def __init__(self, arr): self.arr = arr
            def to_array(self): return self.arr.astype(np.float64)
            
        grasps = self.sampler.sample_grasps(CloudWrapper(points), points, normals, num_grasps=20)
        
        self.get_logger().info(f"Generated {len(grasps)} candidates.")
        if len(grasps) == 0: return None
        
        # 5. Evaluate with PointNet
        self.get_logger().info(f"Scoring {len(grasps)} candidates...")
        
        scored_grasps = []
        
        # WidowX params (approx) for cropping
        # We need points inside the gripper closing area
        hand_depth = 0.06
        hand_width = 0.074 # max width
        hand_height = 0.04
        
        for i, grasp in enumerate(grasps):
            # Grasp: [bottom_center, approach, binormal, axis, modified_center]
            center = grasp[4]
            approach = grasp[1]
            binormal = grasp[2]
            axis = grasp[3]
            
            # 1. Transform points to gripper frame
            # Rotation R = [approach, binormal, axis]
            # Grid frame: x=approach, y=binormal, z=axis
            R = np.vstack([approach, binormal, axis]).T # (3,3)
            
            # P_local = (P_global - center) @ R
            # This projects points onto the gripper axes
            local_points = (points - center) @ R
            
            # 2. Crop
            # X: [0, hand_depth] (Approach direction)
            # Y: [-width/2, width/2] (Closing direction)
            # Z: [-height/2, height/2] (Thickness direction)
            
            mask = (local_points[:, 0] > 0) & (local_points[:, 0] < hand_depth) & \
                   (local_points[:, 1] > -hand_width/2) & (local_points[:, 1] < hand_width/2) & \
                   (local_points[:, 2] > -hand_height/2) & (local_points[:, 2] < hand_height/2)
                   
            pts_in_gripper = local_points[mask]
            
            if len(pts_in_gripper) < 10: # Too few points to score
                continue
                
            # 3. Preprocess for Network (Resample to 500 points)
            num_req = 500
            if len(pts_in_gripper) >= num_req:
                idxs = np.random.choice(len(pts_in_gripper), num_req, replace=False)
            else:
                idxs = np.random.choice(len(pts_in_gripper), num_req, replace=True)
            
            input_points = pts_in_gripper[idxs] # (500, 3)
            
            # Transpose to (3, 500) and add batch dim -> (1, 3, 500)
            input_tensor = torch.from_numpy(input_points.T).float().unsqueeze(0).to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                output, _ = self.model(input_tensor) # Output: (1, NumClasses)
                # Softmax to get probabilities
                probs = torch.nn.functional.softmax(output, dim=1)
                
                # Assuming 2-class (bad, good) or 3-class (bad, good, better)
                # We want the probability of the "good" class.
                # Usually class 1 is good (2-class) or class 2 is best (3-class) based on kinect2grasp.py logic
                # kinect2grasp.py: "the best in 3 class classification is the last column" -> index 2
                
                score = probs[0, -1].item() # Take probability of highest class
                
            scored_grasps.append((score, grasp))
                
        # Sort descending by score
        scored_grasps.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_grasps:
            self.get_logger().warn("No valid grasps found after scoring.")
            return None
            
        # Log top 5 scores
        top_n = min(15, len(scored_grasps))
        self.get_logger().info(f"Top {top_n} candidates:")
        for k in range(top_n):
            self.get_logger().info(f"  Rank {k+1}: Score {scored_grasps[k][0]:.4f}")
            
        # Visualization Loop (Top 5)
        # for k in range(top_n):
        #     self.publish_grasp(scored_grasps[k][1], cloud_msg.header.frame_id)
        #     time.sleep(0.5) # Slight delay to visualize sequence
            
        # Return best
        return scored_grasps[0][1]

    def read_points_numpy(self, cloud_msg, roi=None):
        # Decode PointCloud2
        # x, y, z are float32 at offsets 0, 4, 8
        point_step = cloud_msg.point_step
        row_step = cloud_msg.row_step
        width = cloud_msg.width
        height = cloud_msg.height
        data = cloud_msg.data
        
        points = []
        fmt = 'fff'
        unpack = struct.Struct(fmt).unpack_from
        
        # Determine ROI limits
        if roi:
            u_min = max(0, roi.x_offset)
            v_min = max(0, roi.y_offset)
            u_max = min(width, roi.x_offset + roi.width)
            v_max = min(height, roi.y_offset + roi.height)
        else:
            # Fallback to full image if no ROI provided (or handle error)
            u_min, u_max = 0, width
            v_min, v_max = 0, height
        
        # Density Configuration
        # Stride 1 = Every pixel (High density) for ROI (assume we have a close crop)
        stride_u = 1
        stride_v = 1
        
        self.get_logger().info(f"Processing ROI: u[{u_min}:{u_max}], v[{v_min}:{v_max}] with stride {stride_u}")

        for v in range(v_min, v_max, stride_v):
            row_offset = v * row_step
            for u in range(u_min, u_max, stride_u):
                off = row_offset + u * point_step
                x, y, z = unpack(data, off)
                
                # Basic validation
                if math.isnan(x) or math.isnan(y) or math.isnan(z): continue
                
                # Spatial crop (workspace limits)
                # Z: 0.1m to 0.7m (robot reach)
                if (0.1 < z < 0.5):
                    points.append([x, y, z])
                
        return np.array(points, dtype=np.float32)

    def publish_grasp(self, grasp, header):
        # Grasp format from sampler: 
        # [grasp_bottom_center, approach_normal, binormal, minor_pc, grasp_bottom_center_modify]
        center = grasp[4] # Use modified center
        approach = grasp[1] # Normal (Approach)
        binormal = grasp[2] # Major PC
        axis = grasp[3] # Minor PC (Axis)
        
        # Create Pose
        # Rotation matrix [normal, binormal, axis] -> [x, y, z]
        # X=Approach, Y=Binormal, Z=Axis? 
        # Gripper convention: Z is approach?
        # Usually: Z approach, Y closing direction (binormal), X axis.
        # Let's verify standard gripper frame.
        # WidowX: X forward (approach)?
        # Let's construct a rotation matrix.
        
        # R = [approach, binormal, axis]
        R = np.column_stack((approach, binormal, axis))
        
        # Convert to Quaternion
        # Using scipy
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat() # x, y, z, w
        
        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        self.pose_pub.publish(msg)
        self.get_logger().info(f"Published grasp pose in {header.frame_id}: {center}")
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = PointNetGPDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
