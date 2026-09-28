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
Scripted teleop: drive a fixed route through the house while slam_toolbox maps.

This is the reproducible stand-in for driving with teleop_twist_keyboard. It
publishes geometry_msgs/TwistStamped on /cmd_vel exactly like a human with a
keyboard would, using a proportional go-to-point controller on the robot pose
from TF (map -> base_link, i.e. already SLAM-corrected). A lidar check stops
forward motion if something is closer than stop_distance straight ahead.

Topics:
    Publishes  /cmd_vel  geometry_msgs/TwistStamped
    Subscribes /scan     sensor_msgs/LaserScan
    TF         map -> base_link

Parameters: read from config/mapping_route.yaml (speeds, tolerances, waypoints);
the robot spawn pose comes from config/ground_truth.yaml.

Usage (with slam.launch.py running):
    ros2 run semantic_nav_bringup mapping_drive.py
"""

import argparse
import math
import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from semantic_nav_bringup.nav_utils import (front_clearance, go_to_point, world_to_map,
                                            yaw_from_quaternion)
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
import yaml


class MappingDrive(Node):
    """Follows the waypoint route, spinning 360 deg where requested."""

    def __init__(self, route, spawn):
        super().__init__('mapping_drive',
                         parameter_overrides=[rclpy.parameter.Parameter('use_sim_time',
                                                                        value=True)])
        self.cfg = route
        self.waypoints = [
            dict(wp, map=world_to_map(wp['x'], wp['y'], 0.0, spawn)) for wp in route['waypoints']]
        self.index = 0
        self.spinning = None          # accumulated yaw while spinning, or None
        self.last_yaw = None
        self.waypoint_start = None
        self.clearance = math.inf
        self.blocked_since = None     # sim time when the lidar stop engaged
        self.backup_until = None      # sim time until which we reverse
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.step)
        self.done = False

    def on_scan(self, msg):
        self.clearance = front_clearance(msg.ranges, msg.angle_min, msg.angle_increment)

    def pose(self):
        tf = self.tf_buffer.lookup_transform('map', 'base_link', Time(),
                                             timeout=Duration(seconds=0.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def advance(self, reason):
        wp = self.waypoints[self.index]
        self.get_logger().info(f'waypoint {self.index + 1}/{len(self.waypoints)} '
                               f'(world {wp["x"]:.2f}, {wp["y"]:.2f}): {reason}')
        self.index += 1
        self.spinning, self.waypoint_start = None, None
        if self.index >= len(self.waypoints):
            self.send(0.0, 0.0)
            self.get_logger().info('route complete - save the map now')
            self.done = True

    def step(self):
        if self.done:
            return
        try:
            x, y, yaw = self.pose()
        except TransformException:
            self.get_logger().info('waiting for map -> base_link', throttle_duration_sec=5.0)
            return
        now = self.get_clock().now()
        if self.waypoint_start is None:
            self.waypoint_start = now
        wp = self.waypoints[self.index]

        if self.spinning is not None:  # rotating in place to scan the room
            self.spinning += abs(math.remainder(yaw - self.last_yaw, math.tau))
            self.last_yaw = yaw
            if self.spinning >= math.tau:
                self.advance('reached + 360 deg scan')
            else:
                self.send(0.0, self.cfg['spin_speed'])
            return

        if (now - self.waypoint_start).nanoseconds * 1e-9 > self.cfg['waypoint_timeout']:
            self.advance('TIMEOUT, skipped')
            return

        v, w, dist = go_to_point((x, y, yaw), wp['map'][:2],
                                 self.cfg['linear_speed'], self.cfg['angular_speed'])
        if dist < self.cfg['waypoint_tolerance']:
            if wp.get('spin'):
                self.spinning, self.last_yaw = 0.0, yaw
            else:
                self.advance('reached')
            return
        if self.backup_until is not None:
            if now < self.backup_until:
                self.send(-0.1, 0.0)
                return
            self.backup_until, self.blocked_since = None, None
        if v > 0.0 and self.clearance < self.cfg['stop_distance']:
            v = 0.0
            self.blocked_since = self.blocked_since or now
            if (now - self.blocked_since).nanoseconds * 1e-9 > self.cfg['blocked_backup_after']:
                self.get_logger().warn('blocked - backing up')
                self.backup_until = now + Duration(seconds=self.cfg['backup_duration'])
                return
            self.get_logger().warn(f'obstacle {self.clearance:.2f} m ahead - rotating only',
                                   throttle_duration_sec=2.0)
        else:
            self.blocked_since = None
        self.send(v, w)


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description='Scripted teleop mapping drive')
    parser.add_argument('--route', default=os.path.join(pkg, 'config', 'mapping_route.yaml'))
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    args, ros_args = parser.parse_known_args()
    with open(args.route) as f:
        route = yaml.safe_load(f)
    with open(args.ground_truth) as f:
        s = yaml.safe_load(f)['robot_spawn']
    rclpy.init(args=ros_args)
    node = MappingDrive(route, (s['x'], s['y'], s['yaw']))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
