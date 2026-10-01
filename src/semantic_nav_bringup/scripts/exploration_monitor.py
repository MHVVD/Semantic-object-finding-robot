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
Measure an exploration run: map coverage, distance driven and semantic map quality over time.

Every --period seconds (wall) it logs one CSV row:
    wall_s, sim_s          time since the monitor started
    known_m2               area of the live SLAM /map that is known (free or occupied)
    coverage               fraction of the reference map's free cells (maps/house.pgm,
                           made in M2 with the same SLAM frame) that are known now
    path_m                 distance driven (map -> base_link TF)
    objects, matched       confirmed semantic-map objects / those matching a
                           ground-truth object (semantic_map_eval rules)
    error_m                mean position error of the matched objects
It stops when --done-topic reports "complete" (our explorer), after --duration
seconds, or on Ctrl-C; then it optionally saves the semantic map (--save-map)
and prints a summary.

Usage (with exploration.launch.py running):
    ros2 run semantic_nav_bringup exploration_monitor.py --csv run.csv --save-map run.yaml
"""

import argparse
import csv
import math
import os
import signal
import time

from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
import numpy as np
from PIL import Image
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from semantic_nav_bringup.eval_utils import match_map_to_ground_truth
from semantic_nav_interfaces.msg import SemanticObjectArray
from semantic_nav_interfaces.srv import MapFile
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def reference_free_points(map_yaml):
    """World (map-frame) coordinates of every free cell of a saved map."""
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), meta['image'])))
    res, (ox, oy, _) = meta['resolution'], meta['origin']
    free = img >= 254 * (1 - meta['free_thresh'])          # trinary map: 254 = free
    rows, cols = np.nonzero(free)
    h = img.shape[0]
    return np.stack([ox + (cols + 0.5) * res, oy + (h - rows - 0.5) * res], axis=1)


def load_truth(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'projection_error', os.path.join(HERE, 'projection_error.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_objects(path)[0]


class Monitor(Node):
    """Collects the latest map, semantic map and robot pose."""

    def __init__(self, args):
        super().__init__('exploration_monitor', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid = self.objects = None
        self.done = False
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.create_subscription(SemanticObjectArray, args.semantic_map_topic,
                                 lambda m: setattr(self, 'objects', m.objects), latched)
        self.create_subscription(String, args.done_topic, self.on_status, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.save_client = self.create_client(MapFile, args.save_service)
        self.last_xy, self.path_m = None, 0.0
        self.create_timer(0.5, self.track_path)

    def on_map(self, msg):
        self.grid = msg

    def on_status(self, msg):
        self.get_logger().info(f'explorer: {msg.data}')
        if msg.data.startswith('complete'):
            self.done = True

    def track_path(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time()).transform.translation
        except TransformException:
            return
        if self.last_xy is not None:
            step = math.dist(self.last_xy, (t.x, t.y))
            if step < 1.0:                       # ignore SLAM jumps
                self.path_m += step
        self.last_xy = (t.x, t.y)

    def coverage(self, ref_points):
        g = self.grid
        if g is None:
            return 0.0, 0.0
        res, ox, oy = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y
        data = np.asarray(g.data, dtype=np.int16).reshape(g.info.height, g.info.width)
        known_m2 = float(np.count_nonzero(data >= 0)) * res * res
        mx = np.floor((ref_points[:, 0] - ox) / res).astype(int)
        my = np.floor((ref_points[:, 1] - oy) / res).astype(int)
        inside = (mx >= 0) & (mx < g.info.width) & (my >= 0) & (my < g.info.height)
        known = np.zeros(len(ref_points), dtype=bool)
        known[inside] = data[my[inside], mx[inside]] >= 0
        return known_m2, float(known.mean())


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--csv', required=True)
    parser.add_argument('--save-map', help='save the semantic map here at the end')
    parser.add_argument('--period', type=float, default=10.0)
    parser.add_argument('--duration', type=float, default=3600.0, help='wall seconds')
    parser.add_argument('--reference-map', default=os.path.join(pkg, 'maps', 'house.yaml'))
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    parser.add_argument('--done-topic', default='/semantic_nav/exploration/status')
    parser.add_argument('--semantic-map-topic', default='/semantic_nav/semantic_map')
    parser.add_argument('--save-service', default='/semantic_nav/save_map')
    parser.add_argument('--gate', type=float, default=0.5)
    args, ros_args = parser.parse_known_args()
    ref = reference_free_points(args.reference_map)
    truth = load_truth(args.ground_truth)

    # No rclpy SIGINT handler: Ctrl-C raises KeyboardInterrupt and the context stays
    # alive, so the semantic map can still be saved afterwards.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Monitor(args)
    t0_wall, t0_sim = time.monotonic(), None
    rows, next_log = [], 0.0
    try:
        with open(args.csv, 'w', newline='', buffering=1) as f:
            writer = None
            while rclpy.ok() and not node.done and time.monotonic() - t0_wall < args.duration:
                rclpy.spin_once(node, timeout_sec=0.2)
                wall = time.monotonic() - t0_wall
                if wall < next_log:
                    continue
                next_log = wall + args.period
                sim_now = node.get_clock().now().nanoseconds * 1e-9
                t0_sim = t0_sim if t0_sim is not None else sim_now
                known_m2, cov = node.coverage(ref)
                objs = node.objects or []
                mapped = [{'label': o.label, 'xy': (o.position.x, o.position.y)} for o in objs]
                matches, _, _ = match_map_to_ground_truth(mapped, truth, args.gate)
                err = float(np.mean([e for _, _, e in matches])) if matches else float('nan')
                row = {'wall_s': round(wall, 1), 'sim_s': round(sim_now - t0_sim, 1),
                       'known_m2': round(known_m2, 2), 'coverage': round(cov, 4),
                       'path_m': round(node.path_m, 2), 'objects': len(objs),
                       'matched': len(matches), 'error_m': round(err, 3)}
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                node.get_logger().info(
                    f'sim {row["sim_s"]:6.0f} s | coverage {cov:5.1%} ({known_m2:5.1f} m2) | '
                    f'driven {node.path_m:5.1f} m | objects {len(objs)} '
                    f'({len(matches)}/{len(truth)} matched)')
    except (KeyboardInterrupt, ExternalShutdownException):
        # `ros2 run` forwards SIGINT, so a second one arrives right after the first:
        # ignore it, or it would interrupt the save below.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if args.save_map and rclpy.ok() and node.save_client.wait_for_service(timeout_sec=5.0):
        future = node.save_client.call_async(MapFile.Request(path=os.path.abspath(args.save_map)))
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        if future.result():
            print(future.result().message)
    if rows:
        last = rows[-1]
        print(f'\nfinished after {last["sim_s"]:.0f} s sim / {last["wall_s"]:.0f} s wall: '
              f'coverage {last["coverage"]:.1%}, driven {last["path_m"]:.1f} m, '
              f'{last["objects"]} objects, {last["matched"]}/{len(truth)} matched, '
              f'mean error {last["error_m"]:.3f} m')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
