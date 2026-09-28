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
Launch the four semantic-navigation nodes with the shared parameter file.

Simulation (Gazebo, SLAM, Nav2) is launched separately; later milestones add
sim/slam/nav launch files that include this one.

Launch arguments:
    params_file   Path to the parameter YAML (default: config/params.yaml).
    use_sim_time  Use the /clock topic from Gazebo (default: true).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NODES = [
    ('semantic_nav_perception', 'detector_node'),
    ('semantic_nav_perception', 'projector_node'),
    ('semantic_nav_mapping', 'semantic_map_node'),
    ('semantic_nav_commander', 'commander_node'),
]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('semantic_nav_bringup'), 'config', 'params.yaml')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    nodes = [
        Node(
            package=package,
            executable=executable,
            name=executable,
            output='screen',
            # The launch-level override comes last so it wins over the YAML.
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        )
        for package, executable in NODES
    ]

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Full path to the parameter YAML file'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation (Gazebo) clock'),
        *nodes,
    ])
