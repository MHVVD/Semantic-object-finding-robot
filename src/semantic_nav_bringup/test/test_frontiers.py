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

"""Unit tests for frontier detection, geodesic distances and camera coverage."""

import math

import numpy as np
import pytest
from semantic_nav_bringup.frontiers import (clusters, first_hit, frontier_goal, frontier_mask,
                                            geodesic, GridInfo, mark_seen, near_any, score,
                                            surface_mask, tiles, traversable_from_map,
                                            viewpoint)

RES = 0.1
INFO = GridInfo(RES, 0.0, 0.0, 60, 40)          # 6 m x 4 m
UNK, FREE, OCC = -1, 0, 100


def room():
    """Return a 6 x 4 m area: left 3 m explored (walls at the border), right half unknown."""
    occ = np.full((40, 60), UNK, dtype=np.int16)
    occ[:, :30] = FREE
    occ[0, :30] = occ[-1, :30] = OCC              # bottom / top walls
    occ[:, 0] = OCC                               # left wall
    return occ


# ---------------------------------------------------------------- frontiers
def test_frontier_mask_is_free_next_to_unknown():
    m = frontier_mask(room())
    assert m[10, 29] and m[30, 29]                 # last free column touches unknown
    assert not m[10, 28] and not m[10, 30]          # inside free / unknown itself
    assert not m[0, 29]                             # occupied is never a frontier


def test_clusters_split_by_gap_and_filter_small():
    occ = room()
    occ[18:22, 29] = OCC                            # a short wall stub splits the frontier
    cl = clusters(frontier_mask(occ), INFO, min_cells=3)
    # column 29, rows 1-38 are free frontier cells; the stub removes rows 18-21
    assert sorted(c.size for c in cl) == [17, 17]
    assert all(c.centroid[0] == pytest.approx(2.95) for c in cl)
    assert clusters(frontier_mask(occ), INFO, min_cells=18) == []


def test_no_frontiers_in_a_closed_room():
    occ = np.full((40, 60), FREE, dtype=np.int16)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = OCC
    assert not frontier_mask(occ).any()


# ---------------------------------------------------------------- distances and goals
def test_geodesic_goes_around_walls():
    occ = np.full((40, 60), FREE, dtype=np.int16)
    occ[0:35, 30] = OCC                             # wall at x = 3 m with a gap at the top
    trav = traversable_from_map(occ, INFO, clearance=0.0)
    d = geodesic(trav, (10, 10))
    straight = math.hypot(50 - 10, 0)
    # up 25 cells to the gap, through, down 25: ~66.6 cells instead of 40
    assert d[10, 50] == pytest.approx(66.6, abs=0.5) and d[10, 50] > straight
    assert np.isinf(d[10, 30])                      # the wall itself


def test_traversable_keeps_clearance_from_obstacles():
    occ = room()
    trav = traversable_from_map(occ, INFO, clearance=0.25)
    assert not trav[1, 10] and not trav[2, 10]      # 0.1 / 0.2 m from the bottom wall
    assert trav[5, 10]                              # 0.5 m away
    assert not trav[20, 40]                         # unknown is never traversable


def test_frontier_goal_is_reachable_cell_near_centroid():
    occ = room()
    trav = traversable_from_map(occ, INFO, clearance=0.25)
    cl = clusters(frontier_mask(occ), INFO, min_cells=3)[0]
    dist = geodesic(trav, (5, 20))
    (x, y), d = frontier_goal(cl, dist, INFO)
    assert x == pytest.approx(2.95) and y == pytest.approx(2.05, abs=0.11)
    assert d == pytest.approx(2.4, abs=0.1)         # 24 cells straight east


def test_frontier_goal_none_when_unreachable():
    occ = room()
    cl = clusters(frontier_mask(occ), INFO, min_cells=3)[0]
    assert frontier_goal(cl, np.full(occ.shape, np.inf), INFO) is None


def test_score_trades_distance_against_gain():
    near_small = score(1.0, 10, RES, gain_weight=1.0)       # 1 m away, 1 m frontier
    far_big = score(3.0, 40, RES, gain_weight=1.0)          # 3 m away, 4 m frontier
    assert far_big < near_small                             # exploration wins
    assert score(3.0, 40, RES, 0.1) > score(1.0, 10, RES, 0.1)   # exploitation wins


# ---------------------------------------------------------------- camera coverage
def closed_room():
    occ = np.full((40, 60), FREE, dtype=np.int16)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = OCC
    return occ


def test_mark_seen_respects_fov_and_range():
    occ = closed_room()
    seen = np.zeros(occ.shape, dtype=bool)
    # Camera at (1, 2) looking +x (east wall at x = 5.95), 1.25 rad FOV.
    n = mark_seen(occ, seen, INFO, 1.0, 2.0, 0.0, hfov=1.25, max_range=6.0)
    assert n > 0 and seen[20, 59]                   # straight ahead
    assert not seen[20, 0]                          # behind the camera
    seen2 = np.zeros(occ.shape, dtype=bool)
    mark_seen(occ, seen2, INFO, 1.0, 2.0, 0.0, hfov=1.25, max_range=3.0)
    assert not seen2[20, 59]                        # 5 m away: beyond max_range


def test_mark_seen_stops_at_first_obstacle():
    occ = closed_room()
    occ[15:26, 30] = OCC                            # a cabinet in the middle
    seen = np.zeros(occ.shape, dtype=bool)
    mark_seen(occ, seen, INFO, 1.0, 2.0, 0.0, hfov=0.2, max_range=6.0)
    assert seen[20, 30] and not seen[20, 59]        # wall behind the cabinet hidden


def test_first_hit():
    occ = closed_room()
    occ[20, 30] = OCC
    assert first_hit(occ, INFO, 1.05, 2.05, 5.0, 2.05) == (30, 20)
    assert first_hit(occ, INFO, 1.05, 1.05, 2.0, 1.05) is None


def test_viewpoint_faces_unseen_surface_with_line_of_sight():
    occ = closed_room()
    occ[15:26, 30] = OCC                            # cabinet: the unseen surface
    cl = clusters(occ == OCC, INFO)                 # walls + cabinet are separate clusters
    cabinet = next(c for c in cl if c.size == 11)
    trav = traversable_from_map(occ, INFO, clearance=0.25)
    dist = geodesic(trav, (10, 20))
    (x, y, yaw), d = viewpoint(cabinet, occ, dist, INFO, 1.0, 2.0)
    assert 1.0 - 0.06 <= math.hypot(x - cabinet.centroid[0], y - cabinet.centroid[1]) <= 2.06
    assert math.cos(yaw - math.atan2(cabinet.centroid[1] - y, cabinet.centroid[0] - x)) > 0.999
    assert x < 3.0                                  # the robot's side: closest reachable


def test_near_any():
    assert near_any((1.0, 1.0), [(1.2, 1.0)], 0.5)
    assert not near_any((1.0, 1.0), [(2.0, 1.0)], 0.5) and not near_any((0, 0), [], 1.0)


def test_surface_mask_excludes_wall_interiors():
    occ = closed_room()
    occ[:, 0:3] = OCC                               # a 3-cell thick west wall
    surf = surface_mask(occ)
    assert surf[20, 2] and not surf[20, 1] and not surf[20, 0]


def test_tiles_split_long_walls():
    occ = closed_room()
    t = tiles(surface_mask(occ), INFO, tile_m=1.0, min_cells=3)
    assert len(t) > 10                              # the perimeter is cut into ~1 m pieces
    assert all(c.size <= 2 * 10 for c in t)         # a 1 m tile holds at most ~2 x 10 cells
