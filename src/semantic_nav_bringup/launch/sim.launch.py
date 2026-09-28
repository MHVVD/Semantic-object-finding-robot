# Copyright 2026 Mahmud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Start the house simulation: Gazebo Harmonic + TurtleBot4 + ros_gz_bridge + RViz.

What it launches:
    * gz sim with worlds/<world>.sdf (headless server only, or server + GUI)
    * robot_state_publisher with urdf/turtlebot4_semantic.urdf.xacro
      (OAK-D at camera_width x camera_height @ camera_rate Hz)
    * ros_gz_sim create: spawns the robot from /robot_description
    * ros_gz_bridge (config/ros_gz_bridge.yaml): clock, lidar, RGB, depth, camera_info
    * the Create3 stack from irobot_create_*: ros2_control diff-drive controller,
      motion_control (cmd_vel), and its own bridges (cmd_vel, odom TF, hazards)
    * a static TF linking Gazebo's scoped lidar frame name to the URDF frame
    * RViz with rviz/sim.rviz

Launch arguments:
    world          World file name in worlds/ without .sdf (default: house).
    headless       true: server only (gz sim -s), no Gazebo GUI (default: true).
    rviz           Start RViz (default: true).
    render_engine  ogre2 or ogre, used by the camera sensor and GUI (default: ogre2).
    x, y, yaw      Spawn pose in the world frame (default: -1.0 -2.4 0.0).
    camera_width, camera_height, camera_rate   OAK-D image size / rate (640 480 10).
    lidar_rate     RPLIDAR scan rate in Hz (default: 10).
"""

import os
import re
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ROBOT_NAME = 'turtlebot4'


def robot_description(context):
    """Run xacro, then point the model's Sensors system at the chosen render engine."""
    pkg = get_package_share_directory('semantic_nav_bringup')
    arg = context.launch_configurations
    urdf = subprocess.check_output([
        'xacro', os.path.join(pkg, 'urdf', 'turtlebot4_semantic.urdf.xacro'),
        'gazebo:=ignition', 'namespace:=',
        f'camera_width:={arg["camera_width"]}',
        f'camera_height:={arg["camera_height"]}',
        f'camera_rate:={arg["camera_rate"]}',
        f'lidar_rate:={arg["lidar_rate"]}',
    ], text=True)
    # create3.urdf.xacro hard-codes <render_engine>ogre</render_engine> for the
    # Sensors system; substitute so ogre2 (PBR materials) can be used.
    urdf, n = re.subn(r'<render_engine>\s*ogre2?\s*</render_engine>',
                      f'<render_engine>{arg["render_engine"]}</render_engine>', urdf)
    if n != 1:
        raise RuntimeError(f'expected one <render_engine> tag in the URDF, found {n}')
    return urdf


def launch_setup(context):
    pkg = get_package_share_directory('semantic_nav_bringup')
    world = context.launch_configurations['world']
    headless = context.launch_configurations['headless'] == 'true'
    engine = context.launch_configurations['render_engine']
    world_file = os.path.join(pkg, 'worlds', world + '.sdf')

    if headless:
        # -s: server only. --headless-rendering: EGL offscreen, no X display needed.
        gz_args = f'-r -s --headless-rendering -v 3 {world_file}'
    else:
        gz_args = f'-r -v 3 --render-engine {engine} {world_file}'

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items())

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True,
                     'robot_description': robot_description(context)}])

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-name', ROBOT_NAME, '-topic', 'robot_description',
                   '-x', context.launch_configurations['x'],
                   '-y', context.launch_configurations['y'],
                   '-z', '0.0',
                   '-Y', context.launch_configurations['yaw']])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='sim_bridge',
        output='screen',
        parameters=[{'config_file': os.path.join(pkg, 'config', 'ros_gz_bridge.yaml'),
                     'use_sim_time': True}])

    create3_common = get_package_share_directory('irobot_create_common_bringup')
    create3_gz = get_package_share_directory('irobot_create_gz_bringup')
    create3 = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(create3_gz, 'launch', 'create3_ros_gz_bridge.launch.py')),
            launch_arguments={'robot_name': ROBOT_NAME, 'world': world,
                              'namespace': ''}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(create3_common, 'launch', 'create3_nodes.launch.py')),
            launch_arguments={'namespace': '', 'gazebo': 'ignition'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(create3_gz, 'launch', 'create3_gz_nodes.launch.py')),
            launch_arguments={'robot_name': ROBOT_NAME}.items()),
    ]

    # Gazebo stamps lidar scans with its scoped frame name "<model>/<link>/<sensor>".
    # This identity transform attaches it to the URDF frame (as turtlebot4_gz_bringup
    # does). Camera images need none: the sensor sets <optical_frame_id> in the URDF.
    static_tfs = [
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='rplidar_stf', output='log',
             parameters=[{'use_sim_time': True}],
             arguments=['--frame-id', 'rplidar_link',
                        '--child-frame-id', f'{ROBOT_NAME}/rplidar_link/rplidar']),
    ]

    return [gazebo, rsp, spawn, bridge, *create3, *static_tfs]


def generate_launch_description():
    pkg = get_package_share_directory('semantic_nav_bringup')
    share_parent = os.path.dirname(pkg)  # lets gz resolve package:// and model:// URIs

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='house'),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('render_engine', default_value='ogre2',
                              choices=['ogre2', 'ogre']),
        DeclareLaunchArgument('x', default_value='-1.0'),
        DeclareLaunchArgument('y', default_value='-2.4'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_rate', default_value='10'),
        DeclareLaunchArgument('lidar_rate', default_value='10'),

        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'worlds')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', share_parent),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.dirname(get_package_share_directory('turtlebot4_description'))),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.dirname(get_package_share_directory('irobot_create_description'))),

        OpaqueFunction(function=launch_setup),

        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', PathJoinSubstitution([pkg, 'rviz', 'sim.rviz'])],
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
