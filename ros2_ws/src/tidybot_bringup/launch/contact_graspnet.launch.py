import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    # Resolve launch configurations
    use_sim = LaunchConfiguration('use_sim').perform(context).lower() == 'true'
    ckpt_dir = LaunchConfiguration('ckpt_dir').perform(context)
    
    # Determine depth topic based on environment
    depth_topic = "/camera/depth/image_raw" if use_sim else "/camera/aligned_depth_to_color/image_raw"

    # 1. Point Cloud Generator (depth_image_proc)
    cloud_node = Node(
        package="depth_image_proc",
        executable="point_cloud_xyzrgb_node",
        name="depth_to_cloud",
        output="screen",
        remappings=[
            ("rgb/camera_info", "/camera/color/camera_info"),
            ("rgb/image_rect_color", "/camera/color/image_raw"),
            ("depth_registered/image_rect", depth_topic),
            ("points", "/camera/points"),
        ],
    )

    # 2. Contact-GraspNet Node
    cg_node = Node(
        package="tidybot_bringup",
        executable="contact_graspnet_node.py",
        name="contact_graspnet",
        output="screen",
        parameters=[{
            'ckpt_dir': ckpt_dir,
            'send_plan_request': LaunchConfiguration('send_plan_request'),
            'rviz_topic': LaunchConfiguration('rviz_topic')
        }]
    )

    return [cloud_node, cg_node]

def generate_launch_description():
    # Get default checkpoint directory
    # SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Assuming standard project structure
    default_ckpt = "/home/apasco/me326/collaborative-robotics-2026-group6/ros2_ws/src/tidybot_bringup/scripts/vision-manipulation/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim', default_value='true',
            description='Whether to use simulation camera topics (True) or real robot topics (False)'
        ),
        DeclareLaunchArgument(
            'ckpt_dir', default_value=default_ckpt,
            description='Path to Contact-GraspNet checkpoint directory.'
        ),
        DeclareLaunchArgument(
            'send_plan_request', default_value='false',
            description='If True, calls /plan_to_target. If False, publishes to topic.'
        ),
        DeclareLaunchArgument(
            'rviz_topic', default_value='/detected_grasps/rviz_pose',
            description='Topic for visualizing/publishing the detected grasp.'
        ),
        OpaqueFunction(function=launch_setup)
    ])
