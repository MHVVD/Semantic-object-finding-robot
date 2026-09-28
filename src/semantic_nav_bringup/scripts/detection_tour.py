#!/usr/bin/env python3
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
Drive a camera tour of the house with Nav2: go to each viewpoint, optionally spin 360 deg.

Used to collect detector evaluation / fine-tuning frames: run it while
perception.launch.py capture:=true records the camera. Viewpoints (world
frame) come from config/detection_tour.yaml and are converted to the map
frame with the robot spawn pose from config/ground_truth.yaml.

Usage (with navigation.launch.py running, ideally gt_boxes:=true):
    ros2 run semantic_nav_bringup detection_tour.py [--tour FILE]
"""

import argparse
import math
import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.duration import Duration
from semantic_nav_bringup.nav_utils import quaternion_from_yaw, world_to_map
import yaml


def make_pose(navigator, x, y, yaw):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y = x, y
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
    pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
    return pose


def run_task(nav, timeout_s):
    """Wait for the current Nav2 task, cancelling it after timeout_s of sim time."""
    t0 = nav.get_clock().now()
    while not nav.isTaskComplete():
        if nav.get_clock().now() - t0 > Duration(seconds=timeout_s):
            nav.cancelTask()
    return nav.getResult()


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--tour', default=os.path.join(pkg, 'config', 'detection_tour.yaml'))
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    args, ros_args = parser.parse_known_args()
    with open(args.tour) as f:
        cfg = yaml.safe_load(f)
    with open(args.ground_truth) as f:
        s = yaml.safe_load(f)['robot_spawn']
    spawn = (s['x'], s['y'], s['yaw'])

    rclpy.init(args=ros_args)
    nav = BasicNavigator()
    nav.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
    nav.waitUntilNav2Active(localizer='amcl')
    names = {TaskResult.SUCCEEDED: 'SUCCEEDED', TaskResult.FAILED: 'FAILED',
             TaskResult.CANCELED: 'CANCELED'}
    for i, vp in enumerate(cfg['viewpoints'], start=1):
        gx, gy, gyaw = world_to_map(vp['x'], vp['y'], vp['yaw'], spawn)
        nav.goToPose(make_pose(nav, gx, gy, gyaw))
        result = names.get(run_task(nav, cfg['goal_timeout']), '?')
        msg = f'{i}/{len(cfg["viewpoints"])} {vp["name"]}: {result}'
        if vp.get('spin') and result == 'SUCCEEDED':
            nav.spin(spin_dist=2 * math.pi - 0.01,
                     time_allowance=int(cfg['spin_time_allowance']))
            msg += f', spin {names.get(run_task(nav, cfg["spin_time_allowance"] + 5), "?")}'
        nav.get_logger().info(msg)
    nav.get_logger().info('tour complete')
    nav.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
