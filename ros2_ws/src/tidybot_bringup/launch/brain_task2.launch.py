"""
Brain Task 2 Launch File for TidyBot2.

Launches the FULL system for Task 2: Sequential Pick-and-Place.

1. Simulation layer (via sim.launch.py):
   - MuJoCo bridge, Robot state publisher, Arm controllers, RViz
   - Default scene: scene_task2.xml (banana + bowl)

2. Brain layer (this file):
   - microphone_node   (audio hardware)
   - speech_node       (audio recording + item extraction)
   - navigation_node   (base movement)
   - manipulation_node (arm / gripper control)
   - vision_node       (YOLO + Gemini object detection)
   - brain_node_task2  (Task 2 state machine orchestrator)

Usage:
ros2 launch tidybot_bringup brain_task2.launch.py show_mujoco_viewer:=true use_rviz:=true

Without sim:
ros2 launch tidybot_bringup brain_task2.launch.py use_sim:=false

Start task:
ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'start'}"
Test audio:
ros2 topic pub --once /brain/command std_msgs/msg/String "{data: 'test_audio_sequential /home/mete/collaborative-robotics-2026-group6/examples/banana_bowl.wav'}"
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
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

    declare_use_sim = DeclareLaunchArgument(
        'use_sim', default_value='true',
        description='Launch simulation layer (sim.launch.py). Set to false if real robot drivers are already running.'
    )

    declare_scene = DeclareLaunchArgument(
        'scene', default_value='scene_task2.xml',
        description='MuJoCo scene XML file to load (default: scene_task2.xml for banana + bowl)'
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
                    'scene': LaunchConfiguration('scene'),
                    'use_rviz': LaunchConfiguration('use_rviz'),
                    'show_mujoco_viewer': LaunchConfiguration('show_mujoco_viewer'),
                }.items(),
            )
        ]
    )

    # ── Brain-specific nodes ────────────────────────────────────────

    speech = Node(
        package='tidybot_bringup',
        executable='speech_node.py',
        name='speech_node',
        output='screen',
        # parameters=[{'use_sim_time': True}]
    )

    navigation = Node(
        package='tidybot_bringup',
        executable='navigator.py',
        name='navigation_node',
        output='screen',
        # parameters=[{'use_sim_time': True}]
    )

    manipulation = Node(
        package='tidybot_bringup',
        # executable='manipulation_node.py',             # <-- Switch to this for full IK/Grasping node
        executable='manipulation_node_placeholder.py',    # <-- Using placeholder for now
        name='manipulation_node',
        output='screen',
        # parameters=[{'use_sim_time': True}]
    )

    vision = Node(
        package='tidybot_bringup',
        executable='vision_yolo_gemini.py',
        name='vision_node',
        output='screen',
        # parameters=[{'use_sim_time': True}]
    )

    brain = Node(
        package='tidybot_bringup',
        executable='brain_node_task2.py',
        name='brain_node_task2',
        output='screen',
        # parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        # Arguments
        declare_use_rviz,
        declare_show_viewer,
        declare_use_sim,
        declare_scene,
        # Simulation layer
        sim_launch,
        # Brain layer
        speech,
        navigation,
        vision,
        manipulation,
        brain,
    ])
