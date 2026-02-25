import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

"""
Launch file for the PointNetGPD bridge.

This launch file starts:
1. depth_image_proc/point_cloud_xyzrgb_node: Converts depth and RGB images to a point cloud.
2. pointnet_gpd_node.py: Performs 6-DOF grasp detection on the generated point cloud.

Arguments:
    use_sim (bool): Whether to use simulation camera topics (True) or real robot topics (False).
    use_detected_orientation (bool): If True, uses the orientation from the PointNetGPD model.
                                     If False, forces a top-down grasp. Default: False.
    send_plan_request (bool): If True, calls the /plan_to_target service directly.
                              If False, publishes the pose to the configured topic. Default: True.
    rviz_topic (string): Topic for visualizing/publishing the detected grasp. Default: /detected_grasps/pose.

Example Usage:
    # Run in simulation with default settings:
    ros2 launch tidybot_bringup gpd_bridge.launch.py use_sim:=true

    # Run on real robot and use detected orientation:
    ros2 launch tidybot_bringup gpd_bridge.launch.py use_sim:=false use_detected_orientation:=true

    # Run without sending plan requests (for debugging/coordination):
    ros2 launch tidybot_bringup gpd_bridge.launch.py send_plan_request:=false
"""


def launch_setup(context, *args, **kwargs):
    # Resolve launch configurations
    use_sim = LaunchConfiguration('use_sim').perform(context).lower() == 'true'
    use_detected_orientation = LaunchConfiguration('use_detected_orientation')
    send_plan_request = LaunchConfiguration('send_plan_request')
    rviz_topic = LaunchConfiguration('rviz_topic')

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
        parameters=[{
            'use_detected_orientation': use_detected_orientation,
            'send_plan_request': send_plan_request,
            'rviz_topic': rviz_topic,
        }]
    )

    return [cloud_node, gpd_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim', default_value='true',
            description='Whether to use simulation camera topics (True) or real robot topics (False)'
        ),
        DeclareLaunchArgument(
            'use_detected_orientation', default_value='false',
            description='If True, use detected orientation from PointNetGPD. If False, force top-down.'
        ),
        DeclareLaunchArgument(
            'send_plan_request', default_value='true',
            description='If True, calls /plan_to_target. If False, publishes to topic.'
        ),
        DeclareLaunchArgument(
            'rviz_topic', default_value='/detected_grasps/pose',
            description='Topic for visualizing/publishing the detected grasp.'
        ),
        OpaqueFunction(function=launch_setup)
    ])
