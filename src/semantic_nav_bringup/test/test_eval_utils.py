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

"""Unit tests for the projection-error geometry."""

import math

import pytest
from semantic_nav_bringup.eval_utils import (box_exit_distance, distance_to_box, error_components,
                                             match_object)

CHAIR = {'half': (0.25, 0.25), 'yaw': 0.0}
OBJECTS = [{'name': 'chair_1', 'label': 'chair', 'xy': (0.0, 0.0), **CHAIR},
           {'name': 'chair_2', 'label': 'chair', 'xy': (2.0, 0.0), **CHAIR},
           {'name': 'bed_1', 'label': 'bed', 'xy': (0.0, 3.0), 'half': (1.0, 0.8), 'yaw': 0.0}]


def test_distance_to_box_inside_edge_corner():
    assert distance_to_box((0.1, -0.2), (0, 0), (0.5, 0.3), 0.0) == 0.0
    assert distance_to_box((0.8, 0.0), (0, 0), (0.5, 0.3), 0.0) == pytest.approx(0.3)
    assert distance_to_box((0.8, 0.7), (0, 0), (0.5, 0.3), 0.0) == pytest.approx(0.5)


def test_distance_to_box_rotated():
    # 1.0 x 0.6 box yawed 90 deg: it now extends 0.5 along map y.
    assert distance_to_box((0.0, 0.6), (0, 0), (0.5, 0.3), math.pi / 2) == pytest.approx(0.1)


def test_match_object_uses_footprint_not_centre():
    # 1.1 m from the bed centre but only 0.1 m from its footprint (half 1.0 along x).
    obj, d = match_object('bed', (1.1, 3.0), OBJECTS, 0.5)
    assert obj['name'] == 'bed_1' and d == pytest.approx(0.1)


def test_match_object_picks_closest_same_label_and_respects_gate():
    assert match_object('chair', (1.5, 0.0), OBJECTS, 0.5)[0]['name'] == 'chair_2'
    assert match_object('chair', (0.6, 0.0), OBJECTS, 0.5)[0]['name'] == 'chair_1'
    assert match_object('chair', (1.0, 2.0), OBJECTS, 0.5) == (None, None)
    assert match_object('tv', (0.0, 0.0), OBJECTS, 0.5) == (None, None)


def test_error_components_pure_range():
    # Camera at origin looking along +x at an object at (3, 0); observed at (2.7, 0).
    assert error_components((2.7, 0.0), (3.0, 0.0), (0.0, 0.0)) == pytest.approx((-0.3, 0.0))


def test_error_components_pure_lateral_left_is_positive():
    assert error_components((3.0, 0.2), (3.0, 0.0), (0.0, 0.0)) == pytest.approx((0.0, 0.2))


def test_error_components_preserve_length():
    r, lat = error_components((1.3, 2.4), (1.0, 2.0), (-2.0, -1.0))
    assert math.hypot(r, lat) == pytest.approx(0.5)


def test_box_exit_distance_axis_aligned():
    # 1.0 x 0.4 box: along x you leave after 0.5, along y after 0.2.
    assert box_exit_distance((1, 0), (0.5, 0.2), 0.0) == pytest.approx(0.5)
    assert box_exit_distance((0, -3), (0.5, 0.2), 0.0) == pytest.approx(0.2)


def test_box_exit_distance_rotated_box():
    # Same box yawed 90 deg: its long side now runs along map y.
    assert box_exit_distance((0, 1), (0.5, 0.2), math.pi / 2) == pytest.approx(0.5)


def test_box_exit_distance_diagonal_hits_nearer_face():
    # Square 1 x 1 (half 0.5) along 45 deg: exits through a corner at 0.5 * sqrt(2).
    assert box_exit_distance((1, 1), (0.5, 0.5), 0.0) == pytest.approx(0.5 * math.sqrt(2))
