import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 1. Point Cloud Generator (depth_image_proc)
    # Converts your existing images to the point cloud GPD needs
    cloud_node = Node(
        package="depth_image_proc",
        executable="point_cloud_xyzrgb_node",
        name="depth_to_cloud",
        output="screen",
        remappings=[
            ("rgb/camera_info", "/camera/color/camera_info"),
            ("rgb/image_rect_color", "/camera/color/image_raw"),
            ("depth_registered/image_rect", "/camera/depth/image_raw"),
            ("points", "/camera/points"),  # Output topic
        ],
    )

    # 2. GPD Node
    # Corrected executable name based on your CMakeLists.txt
    gpd_config = os.path.join(
        get_package_share_directory("gpd_ros2"), "config", "ros_eigen_params.cfg"
    )

    gpd_node = Node(
        package="gpd_ros2",
        executable="gpd_ros2_detect_grasps_server",
        name="gpd_ros2",
        output="screen",
        parameters=[
            {"config_file": gpd_config},
            {"cloud_type": 0},
            {"auto_mode": True},
        ],
        remappings=[("cloud_stitched", "/camera/points")],
    )

    return LaunchDescription([cloud_node, gpd_node])
