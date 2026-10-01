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
Run the perception nodes (detector, projector, optional dataset recorder) on a running sim.

Start the robot first (sim.launch.py, slam.launch.py or navigation.launch.py);
the detector subscribes to the OAK-D RGB topic and publishes
/semantic_nav/detections and /semantic_nav/detections_image (shown in the
"Detections" panel of rviz/sim.rviz and rviz/nav.rviz); the projector turns
them into map-frame observations (/semantic_nav/observations) and markers
(/semantic_nav/observation_markers, "Observations" display in RViz).

Launch arguments:
    params_file  Parameter YAML (default: config/params.yaml).
    detector     Start detector_node (default: true).
    projector    Start projector_node (default: true).
    capture      Also start capture_frames, which writes RGB frames (+ ground-truth
                 labels when the sim runs with gt_boxes:=true) to the output_dir
                 set in params_file (default: false).
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
        DeclareLaunchArgument('detector', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('projector', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('capture', default_value='false', choices=['true', 'false']),
        Node(package='semantic_nav_perception', executable='detector_node',
             name='detector_node', output='screen', parameters=[params],
             condition=IfCondition(LaunchConfiguration('detector'))),
        Node(package='semantic_nav_perception', executable='projector_node',
             name='projector_node', output='screen', parameters=[params],
             condition=IfCondition(LaunchConfiguration('projector'))),
        Node(package='semantic_nav_perception', executable='capture_frames',
             name='capture_frames', output='screen', parameters=[params],
             condition=IfCondition(LaunchConfiguration('capture'))),
    ])
