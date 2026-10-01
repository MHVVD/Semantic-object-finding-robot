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
Start the commander (GoTo service) and, optionally, the voice front end.

Needs navigation.launch.py (Nav2) and perception.launch.py (semantic map)
running. Then:
    ros2 run semantic_nav_commander go_to fridge        # typed
    say "go to the fridge"                              # with voice:=true

Launch arguments:
    params_file  Parameter YAML (default: config/params.yaml).
    voice        Also start voice_command on the default microphone (default: false).
    input_wav    Feed voice_command a WAV file instead of the microphone (testing).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('semantic_nav_bringup')
    params = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('voice', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('input_wav', default_value=''),
        Node(package='semantic_nav_commander', executable='commander_node',
             name='commander_node', output='screen', parameters=[params]),
        Node(package='semantic_nav_commander', executable='voice_command',
             name='voice_command', output='screen',
             parameters=[params, {'input_wav': LaunchConfiguration('input_wav')}],
             condition=IfCondition(LaunchConfiguration('voice'))),
    ])
