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
Map the house: slam_toolbox (online async) on top of the simulation.

Drive the robot around (teleop_twist_keyboard with stamped:=true, or
scripts/mapping_drive.py), then save the map:

    ros2 run nav2_map_server map_saver_cli -f <pkg>/maps/house \
        --ros-args -p use_sim_time:=true

slam_toolbox publishes /map (OccupancyGrid) and the map -> odom transform.

Launch arguments:
    sim          Also start sim.launch.py (default: true). Set false if the
                 simulation is already running in another terminal.
    headless     Passed to sim.launch.py (default: true).
    rviz         Start RViz with rviz/nav.rviz (default: true).
    slam_params  slam_toolbox parameter file (default: config/slam_toolbox.yaml).
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

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')),
        launch_arguments={'use_sim_time': 'true',
                          'slam_params_file': LaunchConfiguration('slam_params')}.items())

    rviz = Node(package='rviz2', executable='rviz2', output='log',
                arguments=['-d', os.path.join(pkg, 'rviz', 'nav.rviz')],
                parameters=[{'use_sim_time': True}],
                condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('slam_params',
                              default_value=os.path.join(pkg, 'config', 'slam_toolbox.yaml')),
        sim, slam, rviz,
    ])
