import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    # Resolve use_sim argument
    use_sim = LaunchConfiguration('use_sim').perform(context).lower() == 'true'

    # Determine depth topic based on environment
    # Sim: We warped depth to the raw depth topic in the bridge
    # Real: RealSense driver publishes to aligned_depth_to_color
    depth_topic = "/camera/depth/image_raw" if use_sim else "/camera/aligned_depth_to_color/image_raw"

    # 1. Point Cloud Generator (depth_image_proc)
    # Converts your existing images to the point cloud GPD needs
    # Consider seeing if we can get PC directly from RS?
    cloud_node = Node(
        package="depth_image_proc",
        executable="point_cloud_xyzrgb_node",
        name="depth_to_cloud",
        output="screen",
        remappings=[
            ("rgb/camera_info", "/camera/color/camera_info"),
            ("rgb/image_rect_color", "/camera/color/image_raw"),
            ("depth_registered/image_rect", depth_topic),
            ("points", "/camera/points"),  # Output topic
        ],
    )

    # 2. PointNetGPD Node (Python)
    gpd_node = Node(
        package="tidybot_bringup",
        executable="pointnet_gpd_node.py",
        name="pointnet_gpd",
        output="screen",
    )

    return [cloud_node, gpd_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim', default_value='true',
            description='Whether to use simulation camera topics (True) or real robot topics (False)'
        ),
        OpaqueFunction(function=launch_setup)
    ])
