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
Navigate in the saved house map: Nav2 + AMCL localization on top of the simulation.

Starts (via nav2_bringup/bringup_launch.py, slam:=False):
    map_server (maps/house.yaml) + amcl            -> map -> odom transform
    planner_server, controller_server, smoother_server, behavior_server,
    bt_navigator, waypoint_follower, velocity_smoother, collision_monitor,
    route_server, docking_server, lifecycle managers
All parameters come from config/nav2_params.yaml; use_sim_time is injected.

AMCL is seeded at the map origin (set_initial_pose in the YAML) because the map
frame was created by slam_toolbox at the robot's spawn pose.

Launch arguments:
    sim          Also start sim.launch.py (default: true).
    headless     Passed to sim.launch.py (default: true).
    rviz         Start RViz with rviz/nav.rviz (default: true).
    map          Map YAML (default: maps/house.yaml).
    params_file  Nav2 parameters (default: config/nav2_params.yaml).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('semantic_nav_bringup')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless'),
                          'rviz': 'false'}.items(),
        condition=IfCondition(LaunchConfiguration('sim')))

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')),
        launch_arguments={
            'slam': 'False',
            'map': LaunchConfiguration('map'),
            'params_file': LaunchConfiguration('params_file'),
            'use_sim_time': 'True',
            'autostart': 'True',
        }.items())

    rviz = Node(package='rviz2', executable='rviz2', output='log',
                arguments=['-d', os.path.join(pkg, 'rviz', 'nav.rviz')],
                parameters=[{'use_sim_time': True}],
                condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('map', default_value=os.path.join(pkg, 'maps', 'house.yaml')),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(pkg, 'config', 'nav2_params.yaml')),
        sim, nav2, rviz,
    ])
