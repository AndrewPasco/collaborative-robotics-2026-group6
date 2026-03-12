"""
Brain System Launch File for TidyBot2.

Launches the FULL system in one command:

1. Simulation layer (via sim.launch.py):
   - MuJoCo bridge, Robot state publisher, Arm controllers, RViz
   - Camera rate set to 1 Hz to save bandwidth

2. Brain layer (this file):
   - microphone_node   (audio hardware)
   - speech_node       (audio recording + item extraction)
   - navigation_node   (base movement)
   - manipulation_node (arm / gripper control)
   - brain_node        (state machine orchestrator) 

Usage:
ros2 launch tidybot_bringup brain.launch.py scene:=scene_task2.xml show_mujoco_viewer:=true use_rviz:=true
Without sim:

ros2 launch tidybot_bringup real.launch.py use_lidar:=false

rviz:
rviz2 -d /home/mete/collaborative-robotics-2026-group6/ros2_ws/src/tidybot_bringup/rviz/tidybot.rviz

ros2 launch tidybot_bringup brain.launch.py use_sim:=false


Run with mic:
ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'start'}"

Run with test audio:
ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'test_audio ~/collaborative-robotics-2026-group6/examples/banana.wav'}"

Manual grab pub:
ros2 topic pub --once /brain/manipulation_goal std_msgs/msg/String "{data: 'grab'}"

Skip speech:
ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'bypass banana'}"
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # ── Include sim.launch.py (simulation + visualization) ──────────
    pkg_bringup = FindPackageShare('tidybot_bringup')
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz for visualization'
    )

    declare_show_viewer = DeclareLaunchArgument(
        'show_mujoco_viewer', default_value='true',
        description='Show MuJoCo viewer window'
    )

    declare_use_sim = DeclareLaunchArgument(
        'use_sim', default_value='true',
        description='Launch simulation layer (sim.launch.py). Set to false if real robot drivers are already running.'
    )

    sim_launch = GroupAction(
        condition=IfCondition(LaunchConfiguration('use_sim')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_bringup, 'launch', 'sim.launch.py'])
                ),
                launch_arguments={
                    'camera_rate': '3.0',
                    'use_rviz': LaunchConfiguration('use_rviz'),
                    'show_mujoco_viewer': LaunchConfiguration('show_mujoco_viewer'),
                }.items(),
            )
        ]
    )

    # ── Brain-specific nodes ────────────────────────────────────────
    # microphone = Node(
    #     package='tidybot_control',
    #     executable='microphone_node',
    #     name='microphone_node',
    #     output='screen',
    # )

    speech = Node(
        package='tidybot_bringup',
        executable='speech_node.py',
        name='speech_node',
        output='screen',
        on_exit=Shutdown(),
    )

    navigation = Node(
        package='tidybot_bringup',
        executable='navigator.py',
        name='navigation_node',
        output='screen',
        parameters=[{'use_sim': LaunchConfiguration('use_sim')}],
        on_exit=Shutdown(),
    )

    manipulation_placeholder = Node(
        package='tidybot_bringup',
        # executable='manipulation_node.py',
        executable = 'manipulation_node_placeholder.py',
        name='manipulation_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_sim')),
        on_exit=Shutdown(),
    )
    manipulation = Node(
        package='tidybot_bringup',
        executable='manipulation_node.py',
        name='manipulation_node',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_sim')),
        on_exit=Shutdown(),
    )

    vision = Node(
        package='tidybot_bringup',
        executable='vision_yolo_gemini.py',
        name='vision_node',
        output='screen',
        on_exit=Shutdown(),
    )

    brain = Node(
        package='tidybot_bringup',
        executable='brain_node_task1.py',
        name='brain_node',
        output='screen',
        on_exit=Shutdown(),
    )

    # point_cloud = Node(
    #     package='tidybot_bringup',
    #     executable='point_cloud_node.py',
    #     name='point_cloud_node',
    #     output='screen',
    #     condition=UnlessCondition(LaunchConfiguration('use_sim')),
    #     on_exit=Shutdown(),
    # )

    # depth_topic = '/camera/aligned_depth_to_color/image_raw'
    point_cloud = Node(
        package="depth_image_proc",
        executable="point_cloud_xyz_node",
        name="depth_to_cloud",
        output="screen",
        arguments=['--ros-args', '--log-level', 'FATAL'],
        remappings=[
            ("camera_info", "/camera/color/camera_info"),
            ("image_rect", "/camera/depth/image_raw"),
            ("points", "/camera/points"),
        ],
    )

    grasp_planner = Node(
        package='tidybot_bringup',
        executable='simple_grasp_planner_node.py',
        name='simple_grasp_planner_node',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_sim')),
        on_exit=Shutdown(),
    )

    # ── Real-robot compatibility: arm_controller_node ────────────────
    # When use_sim is false, we need to launch the controller node ourselves 
    # to bridge ArmCommand messages to the real hardware joint topics.
    right_arm_controller = Node(
        package='tidybot_control',
        executable='arm_controller_node',
        name='right_arm_controller',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_sim')),
        parameters=[{'arm_name': 'right', 'control_rate': 50.0}],
        on_exit=Shutdown(),
    )

    left_arm_controller = Node(
        package='tidybot_control',
        executable='arm_controller_node',
        name='left_arm_controller',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_sim')),
        parameters=[{'arm_name': 'left', 'control_rate': 50.0}],
        on_exit=Shutdown(),
    )

    return LaunchDescription([
        # Arguments
        declare_use_rviz,
        declare_show_viewer,
        declare_use_sim,
        # Simulation layer
        sim_launch,
        # Brain layer
        # microphone,
        speech,
        navigation,
        vision,
        # point_cloud,
        # pc_container,
        manipulation,
        manipulation_placeholder,
        grasp_planner,
        # right_arm_controller,
        # left_arm_controller,
        brain,
    ])
