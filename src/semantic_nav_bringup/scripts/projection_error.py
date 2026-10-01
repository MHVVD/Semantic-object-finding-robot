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
Score projector_node's map-frame observations against config/ground_truth.yaml.

Live mode records every observation (plus the camera position from TF at its
stamp, and with --sim-truth the same observation re-placed with Gazebo's TRUE
camera pose) to --raw CSV, then analyses it. --replay analyses an existing
raw CSV without ROS, so the analysis can change without re-driving the robot.

Analysis (map frame, 2D unless noted):
  * match: the same-label object whose FOOTPRINT is nearest, within --gate m
    (a depth camera sees surfaces: on a 2 m bed that is 1 m from the centre);
    unmatched observations are outliers, reported with the object they landed on
  * off-object distance: distance from the observation to that footprint
    (0 = on the object) -- "would a semantic map put it in the right place?"
  * centre error |e_xy| and its split into range (along the camera ray) and
    lateral components; surface offset = centre-to-visible-face distance along
    the ray, so range error + surface offset ~ 0 means "the depth hit the face"
  * height error, and the same metrics with the true camera pose (removes AMCL)

Usage (with navigation + perception running):
    ros2 run semantic_nav_bringup projection_error.py --sim-truth --raw obs.csv
    ros2 run semantic_nav_bringup projection_error.py --replay obs.csv
"""

import argparse
from collections import Counter
import csv
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np
from semantic_nav_bringup.eval_utils import (box_exit_distance, distance_to_box,
                                             error_components, match_object)
from semantic_nav_bringup.nav_utils import world_to_map
import yaml

FIELDS = ['stamp', 'label', 'confidence', 'x', 'y', 'z', 'cam_x', 'cam_y',
          'true_x', 'true_y', 'true_cam_x', 'true_cam_y', 'latest_x', 'latest_y', 'tf_lag_s']


def load_objects(path):
    """Ground-truth objects in the MAP frame, with footprint half sizes."""
    with open(path) as f:
        gt = yaml.safe_load(f)
    spawn = (gt['robot_spawn']['x'], gt['robot_spawn']['y'], gt['robot_spawn']['yaw'])
    objects = []
    for o in gt['objects']:
        x, y, _ = world_to_map(o['position'][0], o['position'][1], 0.0, spawn)
        # `size` is the WORLD-axis-aligned bounding box, so in the map frame the
        # box is rotated only by the spawn yaw (not by the object's own yaw).
        objects.append({'name': o['name'], 'label': o['label'], 'xy': (x, y),
                        'z': o['position'][2], 'yaw': -spawn[2],
                        'half': (o['size'][0] / 2, o['size'][1] / 2)})
    return objects, spawn


def record(args, objects, spawn):
    """Subscribe to observations and write the raw CSV (ROS needed only here)."""
    from nav_msgs.msg import Odometry
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from semantic_nav_interfaces.msg import SemanticObjectArray
    from semantic_nav_perception.projection import quaternion_to_matrix
    from tf2_ros import Buffer, TransformException, TransformListener

    def matrix(t, q):
        m = np.eye(4)
        m[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        m[:3, 3] = (t.x, t.y, t.z)
        return m

    class Recorder(Node):
        def __init__(self, writer):
            super().__init__('projection_error', parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', value=True)])
            self.writer = writer      # rows are written as they arrive: a crash keeps them
            self.rows, self.truth, self.base_cam = [], [], None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
            self.create_subscription(SemanticObjectArray, args.topic, self.on_obs, 50)
            if args.sim_truth:
                # Gazebo's pose republisher is best-effort: a reliable subscription
                # gets nothing.
                self.create_subscription(Odometry, '/sim_ground_truth_pose', self.on_truth,
                                         qos_profile_sensor_data)

        def on_truth(self, msg):
            s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.truth.append((s, matrix(msg.pose.pose.position, msg.pose.pose.orientation)))
            del self.truth[:-400]

        def true_camera(self, stamp):
            """Return the true world <- camera pose at `stamp`, or None."""
            if self.base_cam is None:
                try:
                    tf = self.tf_buffer.lookup_transform('base_link', args.camera_frame, Time())
                except TransformException:
                    return None
                self.base_cam = matrix(tf.transform.translation, tf.transform.rotation)
            if not self.truth:
                return None
            t, world_base = min(self.truth, key=lambda s: abs(s[0] - stamp))
            return world_base @ self.base_cam if abs(t - stamp) <= 0.06 else None

        def on_obs(self, msg):
            if not msg.objects:
                return
            try:
                tf = self.tf_buffer.lookup_transform('map', args.camera_frame,
                                                     Time.from_msg(msg.header.stamp),
                                                     timeout=Duration(seconds=0.2))
            except TransformException:
                return
            map_cam = matrix(tf.transform.translation, tf.transform.rotation)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # The mistake this milestone avoids: using the LATEST transform instead
            # of the one at the image stamp. Recorded to measure what it would cost.
            latest = self.tf_buffer.lookup_transform('map', args.camera_frame, Time())
            map_cam_latest = matrix(latest.transform.translation, latest.transform.rotation)
            lag = Time.from_msg(latest.header.stamp).nanoseconds * 1e-9 - stamp
            world_cam = self.true_camera(stamp) if args.sim_truth else None
            nan = float('nan')
            true_cam = (nan, nan)
            if world_cam is not None:
                true_cam = world_to_map(world_cam[0, 3], world_cam[1, 3], 0.0, spawn)[:2]
            for obs in msg.objects:
                p = obs.position
                true_xy = (nan, nan)
                if world_cam is not None:
                    # back into the camera frame with the projector's (AMCL) TF, then
                    # out again with the true camera pose
                    p_cam = np.linalg.inv(map_cam) @ np.array([p.x, p.y, p.z, 1.0])
                    pw = world_cam @ p_cam
                    true_xy = world_to_map(pw[0], pw[1], 0.0, spawn)[:2]
                p_latest = map_cam_latest @ np.linalg.inv(map_cam) @ np.array(
                    [p.x, p.y, p.z, 1.0])
                self.rows.append(dict(zip(FIELDS, (
                    round(stamp, 3), obs.label, round(obs.confidence, 3), p.x, p.y, p.z,
                    map_cam[0, 3], map_cam[1, 3], *true_xy, *true_cam,
                    p_latest[0], p_latest[1], round(lag, 3)))))
                self.writer.writerow(self.rows[-1])

    with open(args.raw, 'w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        rclpy.init(args=args.ros_args)
        node = Recorder(writer)
        end = time.monotonic() + args.duration
        try:
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.2)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass   # Ctrl-C / SIGINT: rclpy may raise either
    node.destroy_node()
    rclpy.try_shutdown()
    return node.rows


def score(rows, objects, gate, use_truth=False, use_latest=False):
    """Match every observation; return (matched dicts, outlier dicts)."""
    matched, outliers = [], []
    by_name = {o['name']: o for o in objects}
    for r in rows:
        if use_truth:
            if math.isnan(float(r['true_x'])):
                continue
            xy, cam = (float(r['true_x']), float(r['true_y'])), \
                (float(r['true_cam_x']), float(r['true_cam_y']))
        elif use_latest:
            xy, cam = (float(r['latest_x']), float(r['latest_y'])), \
                (float(r['cam_x']), float(r['cam_y']))
        else:
            xy, cam = (float(r['x']), float(r['y'])), (float(r['cam_x']), float(r['cam_y']))
        obj, off = match_object(r['label'], xy, objects, gate)
        if obj is None:
            near = min(objects, key=lambda o: distance_to_box(xy, o['xy'], o['half'], o['yaw']))
            d = distance_to_box(xy, near['xy'], near['half'], near['yaw'])
            outliers.append({'label': r['label'],
                             'landed_on': near['name'] if d <= gate else 'nothing'})
            continue
        ray = (obj['xy'][0] - cam[0], obj['xy'][1] - cam[1])
        rng, lat = error_components(xy, obj['xy'], cam)
        surface = box_exit_distance((-ray[0], -ray[1]), obj['half'], obj['yaw'])
        matched.append({'object': obj['name'], 'off_object': off,
                        'err_xy': math.dist(xy, obj['xy']),
                        'err_z': float(r['z']) - by_name[obj['name']]['z'],
                        'range': rng, 'lateral': lat, 'surface': surface,
                        'distance': math.hypot(*ray)})
    return matched, outliers


def stats(values):
    v = np.asarray(values, dtype=float)
    return (f'median {np.median(v):6.3f}  mean {v.mean():6.3f}  '
            f'p90 {np.percentile(v, 90):6.3f}')


def analyse(rows, objects, gate):
    matched, outliers = score(rows, objects, gate)
    n = len(matched) + len(outliers)
    print(f'\n{n} observations: {len(matched)} on/near a same-label object '
          f'(footprint within {gate} m), {len(outliers)} outliers')
    if not matched:
        return
    print('\nmatched observations, AMCL pose (metres):')
    print('  off-object distance        ', stats([m['off_object'] for m in matched]),
          f'  on the object: {np.mean([m["off_object"] == 0 for m in matched]):.0%}')
    print('  centre error |e_xy|        ', stats([m['err_xy'] for m in matched]))
    print('  range error (signed)       ', stats([m['range'] for m in matched]))
    print('  range error + surface      ', stats([m['range'] + m['surface'] for m in matched]))
    print('  lateral error |e_lat|      ', stats([abs(m['lateral']) for m in matched]))
    print('  height error (signed)      ', stats([m['err_z'] for m in matched]))
    t_matched, _ = score(rows, objects, gate, use_truth=True)
    if t_matched:
        print(f'\nsame, with the TRUE camera pose (n={len(t_matched)}, '
              'localisation error removed):')
        print('  off-object distance        ', stats([m['off_object'] for m in t_matched]))
        print('  centre error |e_xy|        ', stats([m['err_xy'] for m in t_matched]))
        print('  range error + surface      ',
              stats([m['range'] + m['surface'] for m in t_matched]))
        print('  lateral error |e_lat|      ', stats([abs(m['lateral']) for m in t_matched]))

    if rows and 'latest_x' in rows[0]:
        l_matched, l_out = score(rows, objects, gate, use_latest=True)
        lags = [float(r['tf_lag_s']) for r in rows]
        print(f'\nsame, but with the LATEST TF instead of the TF at the image stamp '
              f'(lag {stats(lags)} s):')
        print(f'  outliers {len(l_out)} (vs {len(outliers)})')
        print('  off-object distance        ', stats([m['off_object'] for m in l_matched]))
        print('  lateral error |e_lat|      ', stats([abs(m['lateral']) for m in l_matched]))

    print(f'\n{"object":16s} {"n":>4s} {"dist":>5s} {"off-obj":>7s} {"|e_xy|":>6s} '
          f'{"range":>6s} {"surface":>7s} {"|lat|":>6s} {"e_z":>6s}   (medians, m)')
    for name in sorted({m['object'] for m in matched}):
        ms = [m for m in matched if m['object'] == name]

        def med(k, f=float):
            return float(np.median([f(m[k]) for m in ms]))
        print(f'{name:16s} {len(ms):4d} {med("distance"):5.2f} {med("off_object"):7.3f} '
              f'{med("err_xy"):6.3f} {med("range"):6.3f} {med("surface"):7.3f} '
              f'{med("lateral", abs):6.3f} {med("err_z"):6.3f}')
    print('\nby distance from the camera:')
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 4.5)):
        ms = [m for m in matched if lo <= m['distance'] < hi]
        if ms:
            print(f'  {lo:.1f}-{hi:.1f} m  n={len(ms):4d}  off-object '
                  f'{stats([m["off_object"] for m in ms])}')
    if outliers:
        print('\noutliers (label -> object it landed on):')
        for (label, near), c in Counter((o['label'], o['landed_on'])
                                        for o in outliers).most_common(12):
            print(f'  {c:4d}  {label:13s} -> {near}')


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    parser.add_argument('--raw', help='live mode: write raw observations here')
    parser.add_argument('--replay', help='analyse this raw CSV instead of subscribing')
    parser.add_argument('--topic', default='/semantic_nav/observations')
    parser.add_argument('--camera-frame', default='oakd_rgb_camera_optical_frame')
    parser.add_argument('--gate', type=float, default=0.5,
                        help='max distance (m) from the same-label footprint to count as a match')
    parser.add_argument('--duration', type=float, default=1e9, help='wall seconds')
    parser.add_argument('--sim-truth', action='store_true',
                        help='also record the observation placed with the true camera pose')
    args, args.ros_args = parser.parse_known_args()
    objects, spawn = load_objects(args.ground_truth)
    if args.replay:
        with open(args.replay) as f:
            rows = list(csv.DictReader(f))
    elif args.raw:
        rows = record(args, objects, spawn)
    else:
        parser.error('give --raw FILE (live) or --replay FILE')
    analyse(rows, objects, args.gate)


if __name__ == '__main__':
    main()
