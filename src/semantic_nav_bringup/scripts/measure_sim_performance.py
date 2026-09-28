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
Measure the real-time factor and sensor rates of a running simulation.

RTF = (sim time elapsed on /clock) / (wall time elapsed). Topic rates are given
both per wall-clock second and per simulated second: a camera configured for
10 Hz should show ~10 Hz of *sim* time; its wall rate is ~10 * RTF.

Usage:
    ros2 run semantic_nav_bringup measure_sim_performance.py --duration 30
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, LaserScan

TOPICS = [
    ('/oakd/rgb/preview/image_raw', Image),
    ('/oakd/rgb/preview/depth', Image),
    ('/oakd/rgb/preview/camera_info', CameraInfo),
    ('/scan', LaserScan),
]


class Probe(Node):
    """Counts messages and records /clock against the wall clock."""

    def __init__(self):
        super().__init__('sim_performance_probe')
        self.clock_samples = []  # (wall, sim)
        self.stamps = {name: [] for name, _ in TOPICS}
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        for name, msg_type in TOPICS:
            self.create_subscription(msg_type, name, self.make_cb(name), qos_profile_sensor_data)

    def on_clock(self, msg):
        sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.clock_samples.append((time.monotonic(), sim))

    def make_cb(self, name):
        def cb(msg):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.stamps[name].append((time.monotonic(), stamp))
        return cb


def rate(samples, index):
    if len(samples) < 2:
        return float('nan')
    span = samples[-1][index] - samples[0][index]
    return (len(samples) - 1) / span if span > 0 else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--warmup', type=float, default=3.0)
    args = parser.parse_args()

    rclpy.init()
    probe = Probe()
    end_warmup = time.monotonic() + args.warmup
    while time.monotonic() < end_warmup:
        rclpy.spin_once(probe, timeout_sec=0.1)
    probe.clock_samples.clear()
    for v in probe.stamps.values():
        v.clear()
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        rclpy.spin_once(probe, timeout_sec=0.05)

    c = probe.clock_samples
    rtf = ((c[-1][1] - c[0][1]) / (c[-1][0] - c[0][0])) if len(c) > 1 else float('nan')
    print(f'measured over {args.duration:.0f} s wall')
    print(f'real-time factor: {rtf:.3f}')
    print(f'{"topic":40s} {"msgs":>6s} {"Hz (wall)":>10s} {"Hz (sim)":>9s}')
    for name, samples in probe.stamps.items():
        print(f'{name:40s} {len(samples):6d} {rate(samples, 0):10.2f} {rate(samples, 1):9.2f}')
    probe.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
