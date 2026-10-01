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
Autonomous exploration: unknown house -> occupancy map + semantic map, no teleop.

Starts everything:
    * the simulation (sim.launch.py, headless by default)
    * slam_toolbox online async (builds /map and map -> odom from scratch)
    * Nav2 navigation servers (planner, controller, BT navigator, ...) WITHOUT
      map_server / AMCL: the map comes from SLAM and grows as the robot explores
    * perception.launch.py: detector, projector, semantic map
    * the explorer, which sends NavigateToPose goals to frontiers until none are
      left (see the M7 report), then returns to the start

Launch arguments:
    explorer     frontier (ours, default) | explore_lite (m-explore-ros2, needs
                 its overlay sourced) | none
    headless     Passed to sim.launch.py (default: true).
    rviz         Start RViz with rviz/nav.rviz (default: true).
    nav2_params  Nav2 parameters (default: config/nav2_params.yaml).

Note: launch configurations are global across included launch files, so this
file must NOT declare an argument called params_file -- perception.launch.py
would silently inherit it (the Nav2 file) and its nodes would crash on
missing parameters. The perception parameter file is passed explicitly.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('semantic_nav_bringup')

    def include(launch_file, package_dir, **arguments):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(package_dir, 'launch', launch_file)),
            launch_arguments=arguments.items())

    slam = include('slam.launch.py', pkg, sim='true', headless=LaunchConfiguration('headless'),
                   rviz='false')
    nav2 = include('navigation_launch.py', get_package_share_directory('nav2_bringup'),
                   use_sim_time='True', autostart='True',
                   params_file=LaunchConfiguration('nav2_params'))
    params = os.path.join(pkg, 'config', 'params.yaml')
    perception = include('perception.launch.py', pkg, params_file=params)
    frontier = Node(package='semantic_nav_bringup', executable='frontier_explorer.py',
                    name='frontier_explorer', output='screen', parameters=[params],
                    condition=LaunchConfigurationEquals('explorer', 'frontier'))
    explore_lite = Node(package='explore_lite', executable='explore', name='explore_node',
                        output='screen',
                        parameters=[os.path.join(pkg, 'config', 'explore_lite.yaml')],
                        condition=LaunchConfigurationEquals('explorer', 'explore_lite'))
    rviz = Node(package='rviz2', executable='rviz2', output='log',
                arguments=['-d', os.path.join(pkg, 'rviz', 'nav.rviz')],
                parameters=[{'use_sim_time': True}],
                condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('explorer', default_value='frontier',
                              choices=['frontier', 'explore_lite', 'none']),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('nav2_params',
                              default_value=os.path.join(pkg, 'config', 'nav2_params.yaml')),
        slam, nav2, perception, frontier, explore_lite, rviz,
    ])
