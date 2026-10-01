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
Navigation benchmark: N GoTo commands over random labels, scored against ground truth.

For each of --n labels (drawn uniformly from the ground-truth classes with a
fixed --seed, never the same label twice in a row) it calls the commander's
GoTo service and waits for the answer, which comes when the robot has arrived
or failed. Per command it logs one CSV row:
    label, success, message      the commander's answer
    outcome                      benchmark_stats.classify_goto (reached,
                                 wrong_place, not_in_map, no_goal, nav_failed, timeout)
    time_s                       sim seconds from request to answer
    path_m                       distance actually driven (ground-truth pose)
    final_x, final_y, final_yaw  TRUE final robot pose, map frame
    nearest, dist_m, centre_m    nearest real object of that class: distance to
                                 its footprint and to its centre
    bearing_deg                  angle between the robot's heading and that object
The robot's true pose comes from Gazebo (/sim_ground_truth_pose, world frame,
best-effort QoS) and is converted to the map frame with the spawn pose, so
AMCL errors cannot flatter the result.

Commands are chained: each starts where the previous one ended. A command
that gets no answer within --timeout sim seconds is canceled and logged as
'timeout'. Calls that fail because the stack is not up yet (no costmap, no
robot pose, Nav2 not running) are retried and not counted.

Usage (with navigation.launch.py, the semantic map node and the commander running):
    ros2 run semantic_nav_bringup nav_benchmark.py --seed 1 --csv nav_run1.csv \
        --load-map semantic_map_run1.yaml
"""

import argparse
import csv
import math
import os

from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from semantic_nav_bringup.benchmark_stats import (bearing_error, classify_goto, nearest_object,
                                                  sample_labels)
from semantic_nav_bringup.nav_utils import world_to_map, yaw_from_quaternion
from semantic_nav_interfaces.srv import GoTo, MapFile
from std_srvs.srv import Trigger

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['run', 'index', 'label', 'success', 'outcome', 'time_s', 'path_m', 'final_x',
          'final_y', 'final_yaw', 'nearest', 'dist_m', 'centre_m', 'bearing_deg', 'message']
NOT_READY = ('not running', 'no global costmap', 'robot pose unknown')


def load_truth(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'projection_error', os.path.join(HERE, 'projection_error.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_objects(path)


class Benchmark(Node):
    """Holds the clients and the latest true robot pose."""

    def __init__(self, args, spawn):
        super().__init__('nav_benchmark', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.spawn = spawn
        self.pose = None                 # (x, y, yaw) in the map frame
        self.path_m = 0.0
        self.goto = self.create_client(GoTo, args.goto_service)
        self.cancel = self.create_client(Trigger, args.cancel_service)
        self.load = self.create_client(MapFile, args.load_service)
        self.create_subscription(Odometry, '/sim_ground_truth_pose', self.on_truth,
                                 qos_profile_sensor_data)

    def on_truth(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = world_to_map(p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w), self.spawn)
        if self.pose is not None:
            self.path_m += math.dist(self.pose[:2], pose[:2])
        self.pose = pose

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep_sim(self, seconds):
        end = self.now_s() + seconds
        while rclpy.ok() and self.now_s() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call(self, client, request, timeout_s=None):
        """Call a service; returns (response or None, timed_out)."""
        future = client.call_async(request)
        start = self.now_s()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if timeout_s is not None and self.now_s() - start > timeout_s:
                return None, True
        return future.result(), False


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--csv', required=True)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--run', default='', help='run name written in every row')
    parser.add_argument('--labels', nargs='*',
                        help='labels to draw from (default: every ground-truth class)')
    parser.add_argument('--load-map', help='load this semantic map before starting')
    parser.add_argument('--reach', type=float, default=1.0,
                        help='"reached" = final footprint distance <= this (m)')
    parser.add_argument('--timeout', type=float, default=240.0, help='sim s per command')
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    parser.add_argument('--goto-service', default='/semantic_nav/go_to')
    parser.add_argument('--cancel-service', default='/semantic_nav/cancel')
    parser.add_argument('--load-service', default='/semantic_nav/load_map')
    args, ros_args = parser.parse_known_args()
    truth, spawn = load_truth(args.ground_truth)
    labels = sample_labels(args.labels or sorted({t['label'] for t in truth}), args.n, args.seed)

    rclpy.init(args=ros_args)
    node = Benchmark(args, spawn)
    log = node.get_logger()
    for client in (node.goto, node.cancel) + ((node.load,) if args.load_map else ()):
        while rclpy.ok() and not client.wait_for_service(timeout_sec=5.0):
            log.info(f'waiting for {client.srv_name} ...')
    # wait_for_service only proves the request reader is discovered; give the
    # reply path a moment too (the M6 discovery race lost first replies).
    node.sleep_sim(1.0)
    if args.load_map:
        resp, _ = node.call(node.load, MapFile.Request(path=os.path.abspath(args.load_map)))
        log.info(f'load_map: {resp.message}')
        if not resp.success:
            raise SystemExit(1)
    while rclpy.ok() and node.pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    with open(args.csv, 'w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, label in enumerate(labels):
            for _ in range(10):                  # retry while the stack comes up
                t0, path0 = node.now_s(), node.path_m
                log.info(f'[{i + 1}/{len(labels)}] GoTo {label}')
                resp, timed_out = node.call(node.goto, GoTo.Request(label=label), args.timeout)
                if timed_out or not any(s in resp.message for s in NOT_READY):
                    break
                log.warn(f'stack not ready ({resp.message}); retrying in 10 s')
                node.sleep_sim(10.0)
            if timed_out:
                node.call(node.cancel, Trigger.Request(), 30.0)
            elapsed = node.now_s() - t0
            node.sleep_sim(1.0)                  # let the true pose settle
            x, y, yaw = node.pose
            obj, dist, centre = nearest_object((x, y), label, truth)
            success = bool(resp.success) if resp else False
            message = resp.message if resp else f'no answer within {args.timeout:.0f} s'
            row = {'run': args.run, 'index': i + 1, 'label': label, 'success': int(success),
                   'outcome': classify_goto(success, message, dist, args.reach, timed_out),
                   'time_s': round(elapsed, 1), 'path_m': round(node.path_m - path0, 2),
                   'final_x': round(x, 3), 'final_y': round(y, 3), 'final_yaw': round(yaw, 3),
                   'nearest': obj['name'] if obj else '', 'dist_m': round(dist, 3),
                   'centre_m': round(centre, 3),
                   'bearing_deg': round(math.degrees(bearing_error((x, y), yaw, obj['xy'])), 1)
                   if obj else '', 'message': message}
            writer.writerow(row)
            log.info(f'  -> {row["outcome"]}: {row["time_s"]} s, {row["dist_m"]} m from '
                     f'{row["nearest"]} | {message}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
