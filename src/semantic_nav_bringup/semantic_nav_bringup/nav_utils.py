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
Small, ROS-free geometry helpers for driving and evaluating navigation.

Frames: the *world* frame is Gazebo's; slam_toolbox's *map* frame starts at the
robot spawn pose, so p_map = R(-spawn_yaw) (p_world - spawn_xy).
"""

import math


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def yaw_from_quaternion(x, y, z, w):
    """Yaw (rotation about Z) of a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    """(x, y, z, w) quaternion for a pure rotation about Z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def world_to_map(x, y, yaw, spawn):
    """Express a world-frame pose in the map frame anchored at spawn=(sx, sy, syaw)."""
    sx, sy, syaw = spawn
    dx, dy = x - sx, y - sy
    c, s = math.cos(-syaw), math.sin(-syaw)
    return (c * dx - s * dy, s * dx + c * dy, wrap_angle(yaw - syaw))


def map_to_world(x, y, yaw, spawn):
    """Inverse of world_to_map."""
    sx, sy, syaw = spawn
    c, s = math.cos(syaw), math.sin(syaw)
    return (sx + c * x - s * y, sy + s * x + c * y, wrap_angle(yaw + syaw))


def go_to_point(pose, target, v_max, w_max, k_w=1.5, turn_in_place=0.5):
    """
    Proportional go-to-point controller for a differential-drive robot.

    pose = (x, y, yaw), target = (x, y). Returns (v, w, distance).
    If the heading error exceeds turn_in_place (rad) the robot rotates on the
    spot; otherwise it drives with speed scaled by cos(heading error).

    Anti-orbiting: a robot turning at w_max with speed v moves on a circle of
    radius v / w_max. A target at distance d beside the robot is only reachable
    if that radius is at most d / 2, so v is capped at w_max * d / 2. Without
    this cap the robot can circle the target forever.
    """
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    distance = math.hypot(dx, dy)
    error = wrap_angle(math.atan2(dy, dx) - pose[2])
    w = max(-w_max, min(w_max, k_w * error))
    if abs(error) > turn_in_place:
        return 0.0, w, distance
    v = min(v_max, 0.5 * w_max * distance) * math.cos(error)
    return max(0.0, v), w, distance


def front_clearance(ranges, angle_min, angle_increment, half_width=0.35, range_min=0.2):
    """
    Smallest valid range within +-half_width rad of straight ahead.

    Returns math.inf if there is no valid return in the sector.
    """
    best = math.inf
    for i, r in enumerate(ranges):
        angle = wrap_angle(angle_min + i * angle_increment)
        if abs(angle) <= half_width and range_min <= r < best and math.isfinite(r):
            best = r
    return best
