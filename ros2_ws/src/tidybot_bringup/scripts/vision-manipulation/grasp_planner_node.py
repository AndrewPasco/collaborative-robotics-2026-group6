#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String # For text input
from geometry_msgs.msg import PoseStamped # For grasp poses

# For image processing
try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("Note: cv_bridge/opencv not available. Image processing disabled.")
    print("      Install with: uv pip install opencv-python cv_bridge")

# For YOLOv11 integration
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Note: ultralytics (YOLOv11) not available. Object detection disabled.")
    print("      Install with: uv pip install ultralytics")

# For GraspNet Wrapper integration
try:
    import graspnet_wrapper # This will be created separately
    GRASPNET_WRAPPER_AVAILABLE = True
except ImportError:
    GRASPNET_WRAPPER_AVAILABLE = False
    print("Note: graspnet_wrapper not available. Grasp planning will be simulated.")
    print("      Ensure graspnet_wrapper.py is in the same directory.")

class GraspPlannerNode(Node):
    """
    ROS2 node for vision-based grasp planning.

    Subscribes to camera image and depth topics, receives text input for
    target object, performs object detection/segmentation, and then
    uses GraspAnything to propose grasp poses.
    """

    def __init__(self):
        super().__init__('grasp_planner_node')
        self.get_logger().info('Initializing GraspPlannerNode...')

        # Image processing bridge
        if CV_AVAILABLE:
            self.cv_bridge = CvBridge()
        else:
            self.get_logger().error('OpenCV and CvBridge not available, cannot process images.')
            return

        # YOLOv11 Model
        if YOLO_AVAILABLE:
            # Load a pretrained YOLOv11 model (e.g., yolov11n.pt, yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt)
            self.yolo_model = YOLO('yolov11n.pt')
            self.get_logger().info('YOLOv11 model (yolov11n) loaded.')
            # Get class names from the model
            self.yolo_class_names = self.yolo_model.names
        else:
            self.get_logger().error('ultralytics (YOLOv11) not available, object detection will be simulated.')
            self.yolo_model = None
            self.yolo_class_names = {}

        # GraspNet Wrapper availability check
        self.graspnet_wrapper = None
        if GRASPNET_WRAPPER_AVAILABLE:
            # TODO: Define appropriate checkpoint_path for GraspNetWrapper
            # This path will depend on where the GraspNet model checkpoint is stored.
            checkpoint_path = "/path/to/graspnet_repo/logs/log_rs/checkpoint.tar" # Placeholder
            self.graspnet_wrapper = graspnet_wrapper.GraspNetWrapper(
                self, checkpoint_path
            )
        else:
            self.get_logger().warn('GraspNet wrapper library not found. Grasp pose generation will be simulated.')

        # QoS profile for sensor data (best effort for images)
        qos_profile_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_profile_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # --- Subscribers ---
        self.color_image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw', # Assuming this topic based on project README
            self.color_image_callback,
            qos_profile_sensor
        )
        self.depth_image_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw', # Assuming this topic
            self.depth_image_callback,
            qos_profile_sensor
        )
        # Removed camera_info_sub, GraspNet usually takes directly depth map and intrinsics
        self.object_name_sub = self.create_subscription(
            String,
            '/grasp_planner/target_object', # Custom topic for target object name
            self.object_name_callback,
            qos_profile_reliable
        )
        self.get_logger().info('Subscribed to /camera/color/image_raw, /camera/depth/image_raw, /grasp_planner/target_object')


        # --- Publishers ---
        self.grasp_pose_pub = self.create_publisher(
            PoseStamped,
            '/grasp_planner/grasp_pose',
            qos_profile_reliable
        )
        self.get_logger().info('Publishing to /grasp_planner/grasp_pose')


        # --- Internal State ---
        self.latest_color_image = None
        self.latest_depth_image = None
        # Removed latest_camera_info as GraspNet often takes K matrix directly or is pre-configured
        self.target_object_name = None
        self.processing_request = False # To prevent multiple concurrent requests

        self.get_logger().info('GraspPlannerNode initialized.')

    # Removed camera_info_callback

    def color_image_callback(self, msg: Image):
        """Callback for color image messages."""
        self.latest_color_image = msg
        # self.get_logger().debug('Received color image.')
        self.process_images_if_ready()

    def depth_image_callback(self, msg: Image):
        """Callback for depth image messages."""
        self.latest_depth_image = msg
        # self.get_logger().debug('Received depth image.')
        self.process_images_if_ready()

    def object_name_callback(self, msg: String):
        """Callback for target object name messages."""
        self.target_object_name = msg.data
        self.get_logger().info(f'Received target object: "{self.target_object_name}"')
        self.process_images_if_ready()

    def process_images_if_ready(self):
        """
        Checks if all necessary data (color, depth, object name) is available
        and initiates grasp planning if not already processing.
        """
        if (self.latest_color_image is not None and
            self.latest_depth_image is not None and
            self.target_object_name is not None and
            not self.processing_request):

            self.processing_request = True
            self.get_logger().info(f'All data ready. Initiating grasp planning for "{self.target_object_name}"...')
            self.plan_grasps()
            # Reset after processing
            self.target_object_name = None
            self.latest_color_image = None
            self.latest_depth_image = None
            # No camera info to keep

    def plan_grasps(self):
        """
        Core logic for object detection/segmentation and grasp planning.
        This is a placeholder for actual VLM and GraspAnything integration.
        """
        self.get_logger().info('Starting grasp planning process...')

        # --- Stage 1: Object Detection/Segmentation ---
        # Placeholder: Convert ROS2 Image messages to OpenCV format
        try:
            cv_image_color = self.cv_bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            cv_image_depth = self.cv_bridge.imgmsg_to_cv2(self.latest_depth_image, "16UC1") # Depth as 16-bit unsigned int
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            self.processing_request = False
            return

        self.get_logger().info(f"Detecting/segmenting '{self.target_object_name}' in image...")

        target_bbox = None
        if YOLO_AVAILABLE and self.yolo_model:
            results = self.yolo_model(cv_image_color, verbose=False) # Run YOLOv11 inference

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_class_names.get(class_id)

                    if class_name and class_name.lower() == self.target_object_name.lower():
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        target_bbox = (x1, y1, x2, y2)
                        self.get_logger().info(f"Detected '{class_name}' with bbox: {target_bbox}")
                        break
                if target_bbox:
                    break

            if target_bbox:
                x1, y1, x2, y2 = target_bbox
                cropped_color_image = cv_image_color[y1:y2, x1:x2]
                cropped_depth_image = cv_image_depth[y1:y2, x1:x2]
                self.get_logger().info(f"Object '{self.target_object_name}' detected and cropped.")
            else:
                self.get_logger().warn(f"Object '{self.target_object_name}' not detected by YOLOv11. Simulating crop.")
                # Fallback to simulated crop if object not found or YOLO not available
                height, width, _ = cv_image_color.shape
                cropped_color_image = cv_image_color[height//4:3*height//4, width//4:3*width//4]
                cropped_depth_image = cv_image_depth[height//4:3*height//4, width//4:3*width//4]
        else:
            self.get_logger().warn("YOLOv11 not available. Simulating object detection and crop.")
            height, width, _ = cv_image_color.shape
            cropped_color_image = cv_image_color[height//4:3*height//4, width//4:3*width//4]
            cropped_depth_image = cv_image_depth[height//4:3*height//4, width//4:3*width//4]

        if cropped_color_image.size == 0 or cropped_depth_image.size == 0:
            self.get_logger().error("Cropped image is empty, cannot proceed with grasp planning.")
            self.processing_request = False
            return



        # --- Stage 2: Grasp Pose Estimation with GraspNet ---
        self.get_logger().info('Sending cropped object to GraspNet for grasp pose estimation...')
        
        grasps = []
        if GRASP_ANYTHING_WRAPPER_AVAILABLE and self.grasp_anything_wrapper:
            try:
                grasps = self.grasp_anything_wrapper.predict_grasps(
                    cropped_color_image,
                    cropped_depth_image,
                    self.latest_camera_info,
                    self.target_object_name
                )
                if grasps:
                    self.get_logger().info(f'GraspAnythingWrapper returned {len(grasps)} grasp(s).')
                else:
                    self.get_logger().warn('GraspAnythingWrapper returned no grasps.')
            except Exception as e:
                self.get_logger().error(f"Error during GraspAnything prediction: {e}")
                self.get_logger().warn("Falling back to simulated grasp pose.")
        else:
            self.get_logger().warn("GraspAnything wrapper not available. Generating simulated grasp pose.")

        if grasps:
            # Publish the first viable grasp pose
            self.grasp_pose_pub.publish(grasps[0])
            self.get_logger().info(f'Published a grasp pose from GraspAnythingWrapper.')
        else:
            # Fallback to simulated grasp pose if wrapper not available or no grasps found
            simulated_grasp_pose = PoseStamped()
            simulated_grasp_pose.header.stamp = self.get_clock().now().to_msg()
            simulated_grasp_pose.header.frame_id = 'camera_color_optical_frame' # GraspAnything usually outputs in camera frame

            # Example: a pose slightly in front of the camera, pointing down
            simulated_grasp_pose.pose.position.x = 0.0
            simulated_grasp_pose.pose.position.y = 0.0
            simulated_grasp_pose.pose.position.z = 0.5 # 0.5 meters in front

            # Orientation: pointing straight down (gripper opening towards -Z in camera frame)
            # Assuming camera_color_optical_frame: X-right, Y-down, Z-forward
            # Gripper pointing down means gripper's Z-axis aligns with camera's Y-axis (or -Y)
            # A rotation of -90 degrees around camera's X-axis would make Z point down.
            # Quaternion for -90deg around X: (0.707, -0.707, 0, 0)
            simulated_grasp_pose.pose.orientation.x = 0.707
            simulated_grasp_pose.pose.pose.orientation.y = 0.0
            simulated_grasp_pose.pose.orientation.z = 0.0
            simulated_grasp_pose.pose.orientation.w = 0.707

            self.get_logger().info('Simulated grasp pose generated and published.')
            self.grasp_pose_pub.publish(simulated_grasp_pose)

        self.processing_request = False
        self.get_logger().info('Grasp planning cycle complete.')


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GraspPlannerNode stopped cleanly.')
    except Exception as e:
        node.get_logger().error(f'GraspPlannerNode encountered an error: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
