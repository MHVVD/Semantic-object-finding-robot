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

"""Unit tests for nav_utils: frames, angles, the go-to-point controller and lidar clearance."""

import math

import pytest
from semantic_nav_bringup import nav_utils as nu


def map_to_world(x, y, yaw, spawn):
    """Inverse of nav_utils.world_to_map (test helper)."""
    sx, sy, syaw = spawn
    c, s = math.cos(syaw), math.sin(syaw)
    return (sx + c * x - s * y, sy + s * x + c * y, nu.wrap_angle(yaw + syaw))


@pytest.mark.parametrize('angle, expected', [
    (0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi), (3 * math.pi, math.pi),
    (2 * math.pi + 0.1, 0.1), (-0.1, -0.1), (7.0, 7.0 - 2 * math.pi),
])
def test_wrap_angle(angle, expected):
    assert nu.wrap_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize('yaw', [0.0, 0.7, -2.5, math.pi])
def test_quaternion_roundtrip(yaw):
    q = nu.quaternion_from_yaw(yaw)
    assert math.isclose(sum(v * v for v in q), 1.0)
    assert nu.wrap_angle(nu.yaw_from_quaternion(*q)) == pytest.approx(nu.wrap_angle(yaw))


def test_world_to_map_at_spawn_is_origin():
    spawn = (-1.0, -2.4, 0.0)
    assert nu.world_to_map(-1.0, -2.4, 0.0, spawn) == pytest.approx((0.0, 0.0, 0.0))
    assert nu.world_to_map(0.0, -2.4, 0.0, spawn) == pytest.approx((1.0, 0.0, 0.0))


def test_world_to_map_with_rotated_spawn():
    # robot spawned facing +y: a point 1 m north of it is 1 m straight ahead (+x in map)
    spawn = (2.0, 3.0, math.pi / 2)
    x, y, yaw = nu.world_to_map(2.0, 4.0, math.pi / 2, spawn)
    assert (x, y, yaw) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize('spawn', [(0, 0, 0), (-1.0, -2.4, 0.3), (5, -5, -3.0)])
def test_map_world_roundtrip(spawn):
    for pose in [(1.0, 2.0, 0.5), (-3.3, 1.8, 1.5708), (0.0, 0.0, -3.0)]:
        back = map_to_world(*nu.world_to_map(*pose, spawn), spawn)
        assert back[:2] == pytest.approx(pose[:2])
        assert nu.wrap_angle(back[2] - pose[2]) == pytest.approx(0.0, abs=1e-12)


def test_go_to_point_turns_in_place_when_facing_away():
    v, w, d = nu.go_to_point((0, 0, 0), (-1, 0), v_max=0.25, w_max=0.5)
    assert v == 0.0 and abs(w) == pytest.approx(0.5) and d == pytest.approx(1.0)


def test_go_to_point_drives_straight_when_aligned():
    v, w, _ = nu.go_to_point((0, 0, 0), (2, 0), v_max=0.25, w_max=0.5)
    assert v == pytest.approx(0.25) and w == pytest.approx(0.0)


def test_go_to_point_slows_near_target_and_steers_left():
    v, w, _ = nu.go_to_point((0, 0, 0), (0.2, 0.05), v_max=0.25, w_max=0.5)
    assert 0.0 < v < 0.25   # slowed down near the target
    assert w > 0.0          # target is to the left (+y)


@pytest.mark.parametrize('distance', [0.1, 0.3, 1.0, 3.0])
def test_go_to_point_turning_radius_never_exceeds_half_distance(distance):
    # target 0.4 rad to the left, just inside the drive-while-turning cone
    target = (distance * math.cos(0.4), distance * math.sin(0.4))
    v, w, d = nu.go_to_point((0, 0, 0), target, v_max=0.25, w_max=0.5)
    assert v / 0.5 <= d / 2 + 1e-12   # radius at full turn rate <= d/2: no orbiting


def test_front_clearance_only_counts_the_front_sector():
    n = 360
    inc = 2 * math.pi / n
    ranges = [5.0] * n
    ranges[0] = 0.25          # behind the robot (angle -pi): ignored
    ranges[n // 2] = 0.8      # straight ahead (angle 0)
    ranges[n // 2 + 5] = 0.1  # ahead but below range_min: ignored as invalid
    assert nu.front_clearance(ranges, -math.pi, inc) == pytest.approx(0.8)
    assert nu.front_clearance([math.inf] * n, -math.pi, inc) == math.inf
