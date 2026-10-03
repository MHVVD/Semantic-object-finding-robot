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
The whole robot in a KNOWN house: simulation + Nav2/AMCL + perception + commander.

Starts:
    navigation.launch.py   gz sim, map_server (maps/house.yaml), AMCL, Nav2, RViz
    perception.launch.py   detector -> projector -> semantic map
    commander.launch.py    GoTo service (+ voice front end with voice:=true)
The semantic map starts empty (or from map_file with load_on_start in params.yaml)
and fills as the robot drives; then:
    ros2 run semantic_nav_commander go_to fridge
For an UNKNOWN house use exploration.launch.py instead.

Launch arguments:
    headless   Passed to the simulation (default: true).
    rviz       Start RViz with rviz/nav.rviz (default: true).
    voice      Also start the voice front end (default: false).

Note: launch configurations are global across included files, so every include
gets its parameter file passed explicitly (navigation.launch.py's params_file is
the Nav2 file; see PROBLEMS_LOG #71).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg = get_package_share_directory('semantic_nav_bringup')
    params = os.path.join(pkg, 'config', 'params.yaml')

    def include(launch_file, **arguments):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', launch_file)),
            launch_arguments=arguments.items())

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('voice', default_value='false', choices=['true', 'false']),
        include('navigation.launch.py', headless=LaunchConfiguration('headless'),
                rviz=LaunchConfiguration('rviz'),
                params_file=os.path.join(pkg, 'config', 'nav2_params.yaml')),
        include('perception.launch.py', params_file=params),
        include('commander.launch.py', params_file=params, voice=LaunchConfiguration('voice')),
    ])
