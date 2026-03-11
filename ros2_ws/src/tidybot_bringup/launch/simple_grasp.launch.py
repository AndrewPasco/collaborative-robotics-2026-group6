from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    use_sim = LaunchConfiguration('use_sim').perform(context).lower() == 'false'
    depth_topic = "/camera/depth/image_raw" if use_sim else "/camera/aligned_depth_to_color/image_raw"

    # # 1. Point Cloud Generator
    # cloud_node = Node(
    #     package="depth_image_proc",
    #     executable="point_cloud_xyzrgb_node",
    #     name="depth_to_cloud",
    #     output="screen",
    #     remappings=[
    #         ("rgb/camera_info", "/camera/color/camera_info"),
    #         ("rgb/image_rect_color", "/camera/color/image_raw"),
    #         ("depth_registered/image_rect", depth_topic),
    #         ("points", "/camera/points"),
    #     ],
    # )

    cloud_node = Node(
        package="depth_image_proc",
        executable="point_cloud_xyz_node",
        name="depth_to_cloud",
        output="screen",
        remappings=[
            ("camera_info", "/camera/depth/camera_info"),
            ("image_rect", "/camera/depth/image_raw"),
            ("points", "/camera/points"),
        ],
    )

    # 2. Simple Grasp Planner
    simple_node = Node(
        package="tidybot_bringup",
        executable="simple_grasp_planner_node.py",
        name="simple_grasp_planner",
        output="screen",
        parameters=[{
            'grasp_type': LaunchConfiguration('grasp_type'),
            'table_height_buffer': 0.01,
            # 'use_sim_time': True, # REMOVE FOR REAL
        }]
    )

    return [cloud_node, simple_node]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='true'),
        DeclareLaunchArgument('grasp_type', default_value='top', description='top or side'),
        OpaqueFunction(function=launch_setup)
    ])
