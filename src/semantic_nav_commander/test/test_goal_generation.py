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

"""Unit tests for goal generation against synthetic costmaps."""

import math
from types import SimpleNamespace as Obj

import pytest
from semantic_nav_commander.goal_generation import (candidate_goals, CostGrid, credible_instances,
                                                    INSCRIBED, LETHAL, local_free_space,
                                                    NO_INFORMATION,
                                                    parse_aliases,
                                                    resolve_label, yaw_to_quaternion)

RES = 0.05
N = 100                                    # 100 x 100 cells = a 5 m x 5 m room

PARAMS = {'radii': [0.8], 'n_samples': 16, 'max_cost': 252, 'max_detour': 1.5,
          'cost_weight': 1.0}


def quaternion_to_yaw(x, y, z, w):
    """Inverse of yaw_to_quaternion (test helper)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_grid(painter=None, fill=0):
    """Synthetic costmap; painter(x, y) -> cost or None (keep fill) at each cell centre."""
    data = []
    for my in range(N):
        for mx in range(N):
            x, y = (mx + 0.5) * RES, (my + 0.5) * RES
            c = painter(x, y) if painter else None
            data.append(fill if c is None else c)
    return CostGrid(N, N, RES, 0.0, 0.0, tuple(data))


def disc(cx, cy, r_lethal, r_inscribed=0.0):
    """Return a painter: a lethal circle for the object, with an inscribed ring around it."""
    def paint(x, y):
        d = math.hypot(x - cx, y - cy)
        if d <= r_lethal:
            return LETHAL
        if d <= r_inscribed:
            return INSCRIBED
        return None
    return paint


# ---------------------------------------------------------------- grid basics
def test_world_to_cell_and_outside():
    g = make_grid()
    assert g.world_to_cell(0.0, 0.0) == (0, 0)
    assert g.world_to_cell(4.999, 0.051) == (99, 1)
    assert g.world_to_cell(5.0, 1.0) is None and g.world_to_cell(-0.01, 1.0) is None
    assert g.cost(-1.0, 1.0) == NO_INFORMATION


# ---------------------------------------------------------------- candidates
def test_free_room_best_goal_faces_object_from_robot_side():
    # Object (a 0.2 m lethal disc) in the middle, robot to its west.
    g = make_grid(disc(2.5, 2.5, 0.2))
    cands = candidate_goals((2.5, 2.5), (0.5, 2.5), g, **PARAMS)
    assert len(cands) == 16                                   # all samples valid
    best = cands[0]
    assert (best.x, best.y) == pytest.approx((1.7, 2.5))      # standoff 0.8 m, robot side
    assert best.yaw == pytest.approx(0.0)                     # facing +x = the object
    for c in cands:                                           # every yaw points at it
        assert math.atan2(2.5 - c.y, 2.5 - c.x) == pytest.approx(c.yaw)
        assert math.hypot(c.x - 2.5, c.y - 2.5) == pytest.approx(0.8)


def test_lethal_and_inscribed_cells_are_rejected():
    # A second obstacle sits exactly where the best (west) candidate would be,
    # surrounded by an inscribed ring.
    obstacle = disc(1.7, 2.5, 0.1, 0.3)
    obj = disc(2.5, 2.5, 0.2)
    g = make_grid(lambda x, y: obj(x, y) if obj(x, y) is not None else obstacle(x, y))
    cands = candidate_goals((2.5, 2.5), (0.5, 2.5), g, **PARAMS)
    assert all(g.cost(c.x, c.y) not in (LETHAL, INSCRIBED) for c in cands)
    assert all(math.hypot(c.x - 1.7, c.y - 2.5) > 0.3 for c in cands)
    assert 0 < len(cands) < 16


def test_unknown_and_outside_map_are_rejected():
    # Object in a corner: half of the circle is outside the map, and a strip of
    # unknown space covers y > 0.9.
    g = make_grid(lambda x, y: NO_INFORMATION if y > 0.9 else None)
    cands = candidate_goals((0.3, 0.3), (2.0, 0.3), g, **PARAMS)
    assert cands
    assert all(0 <= c.x < 5 and 0 <= c.y <= 0.9 for c in cands)


def wall_with_door(x, y):
    # Wall along x = 3.0 (0.1 m thick) with a doorway at y > 4.4.
    return LETHAL if 2.95 <= x <= 3.05 and y < 4.4 else None


def test_object_against_wall_never_gets_a_goal_behind_the_wall():
    # The fridge point is 0.2 m in front of the wall, the robot is on the OTHER side.
    # Points behind the wall are free and only 0.8 m away in a straight line, but
    # reachable only through the door: not in the fridge's local free space.
    g = make_grid(wall_with_door)
    cands = candidate_goals((2.8, 2.5), (4.5, 2.5), g, **PARAMS)
    assert cands and all(c.x < 2.95 for c in cands)
    lax = candidate_goals((2.8, 2.5), (4.5, 2.5), g, **{**PARAMS, 'max_detour': 20.0})
    assert lax[0].x > 3.05                                    # what the check prevents


def test_far_side_of_a_small_object_is_allowed():
    # A chair-sized obstacle: going around it is a short detour, so the far side
    # (towards the robot, here) is valid.
    g = make_grid(disc(2.5, 2.5, 0.25))
    cands = candidate_goals((2.4, 2.5), (4.5, 2.5), g, **PARAMS)
    assert cands[0].x > 2.75


def test_cost_breaks_tie_between_equidistant_candidates():
    g = make_grid(lambda x, y: 200 if x > 2.5 else None)
    params = {**PARAMS, 'n_samples': 2}                       # samples at E and W only
    cands = candidate_goals((2.5, 2.5), (2.5, 4.0), g, **params)
    assert cands[0].x < 2.5 and cands[0].cost == 0
    assert cands[1].cost == 200 and cands[1].score > cands[0].score


def test_falls_back_to_next_radius_when_standoff_is_blocked():
    # An inscribed ring at 0.7-0.9 m: every 0.8 m sample would collide; 1.2 m is free.
    ring = lambda x, y: INSCRIBED if 0.7 <= math.hypot(x - 2.5, y - 2.5) <= 0.9 else None  # noqa
    g = make_grid(ring)
    cands = candidate_goals((2.5, 2.5), (0.5, 2.5), g, **{**PARAMS, 'radii': [0.8, 1.2]})
    assert cands and all(math.hypot(c.x - 2.5, c.y - 2.5) == pytest.approx(1.2)
                         for c in cands)


def test_no_valid_goal_returns_empty():
    g = make_grid(fill=LETHAL)
    assert candidate_goals((2.5, 2.5), (0.5, 0.5), g, **PARAMS) == []


def test_max_cost_threshold():
    g = make_grid(fill=150)
    assert candidate_goals((2.5, 2.5), (0.5, 0.5), g, **{**PARAMS, 'max_cost': 100}) == []
    assert candidate_goals((2.5, 2.5), (0.5, 0.5), g, **{**PARAMS, 'max_cost': 200})


def test_local_free_space_seeds_around_an_object_inside_an_obstacle():
    # Object point at the centre of a lethal disc: seeds are the free cells around it.
    g = make_grid(disc(2.5, 2.5, 0.2))
    d = local_free_space(g, (2.5, 2.5), 1.0)
    assert g.world_to_cell(2.5, 2.5) not in d                 # lethal, never reached
    assert d[g.world_to_cell(2.5, 2.8)] == pytest.approx(0.0, abs=0.06)  # a seed
    assert g.world_to_cell(2.5, 3.85) not in d                # ~1.1 m from the seed ring


def test_local_free_space_goes_around_walls():
    g = make_grid(wall_with_door)
    d = local_free_space(g, (2.8, 2.5), 10.0)
    behind = d[g.world_to_cell(3.2, 2.5)]
    assert behind > 3.5                                       # via the door at y = 4.4


# ---------------------------------------------------------------- yaw <-> quaternion
@pytest.mark.parametrize('yaw,q', [
    (0.0, (0, 0, 0, 1)),
    (math.pi / 2, (0, 0, math.sqrt(0.5), math.sqrt(0.5))),
    (math.pi, (0, 0, 1, 0)),
    (-math.pi / 2, (0, 0, -math.sqrt(0.5), math.sqrt(0.5))),
])
def test_yaw_to_quaternion(yaw, q):
    assert yaw_to_quaternion(yaw) == pytest.approx(q, abs=1e-12)


@pytest.mark.parametrize('yaw', [0.0, 0.3, -2.0, 3.0, math.pi - 1e-9])
def test_quaternion_round_trip_and_unit_norm(yaw):
    q = yaw_to_quaternion(yaw)
    assert sum(v * v for v in q) == pytest.approx(1.0)
    assert quaternion_to_yaw(*q) == pytest.approx(yaw)


def test_q_and_minus_q_are_the_same_rotation():
    q = yaw_to_quaternion(1.0)
    assert quaternion_to_yaw(*(-v for v in q)) == pytest.approx(1.0)


# ---------------------------------------------------------------- names
def test_aliases_and_label_resolution():
    aliases = parse_aliases(['fridge=refrigerator', 'Sofa = couch', 'plant=potted plant'])
    assert resolve_label('the Fridge', aliases) == 'refrigerator'
    assert resolve_label('sofa', aliases) == 'couch'
    assert resolve_label('a plant', aliases) == 'potted plant'
    assert resolve_label('dining_table', aliases) == 'dining table'
    assert parse_aliases(['']) == {}
    with pytest.raises(ValueError):
        parse_aliases(['fridge'])


def couch(x, y):
    # A couch as the SLAM costmap sees it: a closed lethal outline (x 1.5-3.5,
    # y 2.5-3.5) around an unknown interior, inscribed cells just inside and out.
    inside = 1.5 <= x <= 3.5 and 2.5 <= y <= 3.5
    on_edge = inside and (x < 1.6 or x > 3.4 or y < 2.6 or y > 3.4)
    near = 1.35 <= x <= 3.65 and 2.35 <= y <= 3.65
    if on_edge:
        return LETHAL
    if inside:
        return NO_INFORMATION if 1.75 <= x <= 3.25 and 2.75 <= y <= 3.25 else INSCRIBED
    return INSCRIBED if near else None


def test_object_point_inside_an_enclosed_obstacle():
    # The couch point (from depth, seen from the front = -y side) lies inside the
    # outline, 0.3 m behind the front face. Goals must be outside, on the front.
    g = make_grid(couch)
    cands = candidate_goals((2.4, 2.8), (2.4, 0.5), g, **PARAMS)
    assert cands, 'no goal found for an object inside an enclosed obstacle'
    assert all(c.y < 2.35 for c in cands)                      # in front, outside
    assert cands[0].yaw == pytest.approx(math.pi / 2, abs=0.3)  # facing +y = the couch


def test_credible_instances_drops_weak_phantoms():
    beds = [Obj(observation_count=129), Obj(observation_count=16)]
    assert [o.observation_count for o in credible_instances(beds, 0.25)] == [129]
    plants = [Obj(observation_count=c) for c in (93, 118, 70)]
    assert len(credible_instances(plants, 0.25)) == 3
    assert len(credible_instances(beds, 0.0)) == 2 and credible_instances([], 0.25) == []
