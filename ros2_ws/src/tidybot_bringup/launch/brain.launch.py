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
    ros2 launch tidybot_bringup brain.launch.py scene:=scene_pickup.xml show_mujoco_viewer:=true use_rviz:=true 
    ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'start'}"
Test audio:
    ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'test_audio /home/mete/collaborative-robotics-2026-group6/examples/banana.wav'}"
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


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
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_bringup, 'launch', 'sim.launch.py'])
        ),
        launch_arguments={
            'camera_rate': '3.0',
            'use_rviz': LaunchConfiguration('use_rviz'),
            'show_mujoco_viewer': LaunchConfiguration('show_mujoco_viewer'),
        }.items(),
    )

    # ── Brain-specific nodes ────────────────────────────────────────
    microphone = Node(
        package='tidybot_control',
        executable='microphone_node',
        name='microphone_node',
        output='screen',
    )

    speech = Node(
        package='tidybot_bringup',
        executable='speech_node.py',
        name='speech_node',
        output='screen',
    )

    navigation = Node(
        package='tidybot_bringup',
        executable='navigator.py',
        name='navigation_node',
        output='screen',
    )

    manipulation = Node(
        package='tidybot_bringup',
        executable='manipulation_node.py',
        name='manipulation_node',
        output='screen',
    )

    brain = Node(
        package='tidybot_bringup',
        executable='brain_node.py',
        name='brain_node',
        output='screen',
    )

    vision = Node(
        package='tidybot_bringup',
        executable='vision_yolo_gemini.py',
        name='vision_node',
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        declare_use_rviz,
        declare_show_viewer,
        # Simulation layer
        sim_launch,
        # Brain layer
        microphone,
        speech,
        navigation,
        vision,
        manipulation,
        brain,
    ])
