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
Send a sequence of NavigateToPose goals across rooms and report the results.

For each goal: result (SUCCEEDED / FAILED / CANCELED), sim time and wall time
taken, path length driven, recoveries triggered, and the final position and
yaw error against the goal. The robot pose is read from TF (map -> base_link,
i.e. AMCL-corrected) - not /amcl_pose, which AMCL only republishes after it has
moved update_min_d / update_min_a and can therefore be stale at the goal.

Uses nav2_simple_commander.BasicNavigator (a thin client of the
/navigate_to_pose action). Goals come from config/nav_test_goals.yaml.

With --sim-truth (simulation only) it also queries Gazebo for the robot's true
pose and reports the AMCL localization error at each goal.

Usage (with navigation.launch.py running):
    ros2 run semantic_nav_bringup nav_goal_test.py [--goals FILE] [--csv OUT] [--sim-truth]
"""

import argparse
import csv
import math
import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from semantic_nav_bringup.nav_utils import quaternion_from_yaw, world_to_map, yaw_from_quaternion
from tf2_ros import Buffer, TransformException, TransformListener
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


def robot_pose(tf_buffer):
    """Return the current (x, y, yaw) of base_link in the map frame, or None."""
    try:
        tf = tf_buffer.lookup_transform('map', 'base_link', Time())
    except TransformException:
        return None
    t, q = tf.transform.translation, tf.transform.rotation
    return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)


def gazebo_pose(model='turtlebot4'):
    """Return the true (x, y, yaw) of a model in the Gazebo world frame (`gz model -p`)."""
    out = subprocess.run(['gz', 'model', '-m', model, '-p'],
                         capture_output=True, text=True, check=True).stdout
    xyz_line, rpy_line = out.split('Pose [')[1].splitlines()[1:3]
    xyz = [float(v) for v in xyz_line.strip(' []').split()]
    rpy = [float(v) for v in rpy_line.strip(' []').split()]
    return xyz[0], xyz[1], rpy[2]


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--goals', default=os.path.join(pkg, 'config', 'nav_test_goals.yaml'))
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    parser.add_argument('--csv', help='optional CSV output file')
    parser.add_argument('--sim-truth', action='store_true',
                        help='also report AMCL error against Gazebo ground truth')
    args, ros_args = parser.parse_known_args()
    with open(args.goals) as f:
        cfg = yaml.safe_load(f)
    with open(args.ground_truth) as f:
        s = yaml.safe_load(f)['robot_spawn']
    spawn = (s['x'], s['y'], s['yaw'])

    rclpy.init(args=ros_args)
    nav = BasicNavigator()
    nav.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
    tf_buffer = Buffer()
    TransformListener(tf_buffer, nav)
    nav.waitUntilNav2Active(localizer='amcl')

    rows = []
    for i, goal in enumerate(cfg['goals'], start=1):
        gx, gy, gyaw = world_to_map(goal['x'], goal['y'], goal['yaw'], spawn)
        nav.goToPose(make_pose(nav, gx, gy, gyaw))
        t_sim0, t_wall0 = nav.get_clock().now(), time.monotonic()
        path_len, last, recoveries = 0.0, None, 0
        while not nav.isTaskComplete():
            fb = nav.getFeedback()
            if fb:
                recoveries = fb.number_of_recoveries
                if (nav.get_clock().now() - t_sim0) > Duration(seconds=cfg['goal_timeout']):
                    nav.cancelTask()
            pose = robot_pose(tf_buffer)   # isTaskComplete() already spun the node
            if pose is not None:
                if last is not None:
                    path_len += math.dist(pose[:2], last)
                last = pose[:2]
        result = nav.getResult()
        sim_s = (nav.get_clock().now() - t_sim0).nanoseconds * 1e-9
        wall_s = time.monotonic() - t_wall0
        rclpy.spin_once(nav, timeout_sec=0.2)
        x, y, yaw = robot_pose(tf_buffer)
        err = math.hypot(x - gx, y - gy)
        yaw_err = abs(math.remainder(yaw - gyaw, math.tau))
        loc_err = float('nan')
        if args.sim_truth:
            tx, ty, _ = world_to_map(*gazebo_pose(), spawn)
            loc_err = math.hypot(x - tx, y - ty)
        name = {TaskResult.SUCCEEDED: 'SUCCEEDED', TaskResult.FAILED: 'FAILED',
                TaskResult.CANCELED: 'CANCELED'}.get(result, str(result))
        rows.append({'goal': i, 'name': goal['name'], 'result': name,
                     'sim_s': round(sim_s, 1), 'wall_s': round(wall_s, 1),
                     'path_m': round(path_len, 2), 'recoveries': recoveries,
                     'pos_err_m': round(err, 3),
                     'yaw_err_deg': round(math.degrees(yaw_err), 1),
                     'amcl_err_m': round(loc_err, 3)})
        nav.get_logger().info(str(rows[-1]))

    header = f'{"#":>2} {"goal":18s} {"result":10s} {"sim s":>7s} {"wall s":>7s} ' \
             f'{"path m":>7s} {"recov":>5s} {"err m":>6s} {"yaw err":>7s} {"amcl err":>8s}'
    print('\n' + header + '\n' + '-' * len(header))
    for r in rows:
        print(f'{r["goal"]:>2} {r["name"]:18s} {r["result"]:10s} {r["sim_s"]:7.1f} '
              f'{r["wall_s"]:7.1f} {r["path_m"]:7.2f} {r["recoveries"]:5d} '
              f'{r["pos_err_m"]:6.3f} {r["yaw_err_deg"]:7.1f} {r["amcl_err_m"]:8.3f}')
    ok = sum(r['result'] == 'SUCCEEDED' for r in rows)
    total_sim = sum(r['sim_s'] for r in rows)
    total_wall = sum(r['wall_s'] for r in rows)
    print(f'\n{ok}/{len(rows)} goals succeeded; total sim time {total_sim:.1f} s, '
          f'wall time {total_wall:.1f} s')
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    nav.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
