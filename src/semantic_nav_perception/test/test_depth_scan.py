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

"""Unit tests for the depth -> height-filtered scan maths."""

import math

import numpy as np
import pytest
from semantic_nav_perception.depth_scan import (beam_layout, depth_points, scan_ranges,
                                                transform_points)

FX = FY = 320.0 / math.tan(0.625)
CX, CY = 320.0, 240.0
# optical (z fwd, x right, y down) -> base_link (x fwd, y left, z up), camera 0.24 m up
R_BASE_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
T_BASE_OPT = np.array([0.0, 0.0, 0.24])


def test_depth_points_deprojects_and_skips_invalid():
    depth = np.full((480, 640), np.nan)
    depth[240, 480] = 2.0
    pts = depth_points(depth, FX, FY, CX, CY, decimation=1)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx([(480 - CX) * 2.0 / FX, 0.0, 2.0])


def test_depth_points_decimation_keeps_every_kth_pixel():
    depth = np.ones((8, 8))
    assert len(depth_points(depth, FX, FY, CX, CY, decimation=4)) == 4


def test_transform_optical_to_base():
    pts = transform_points(np.array([[0.0, 0.0, 2.0]]), R_BASE_OPT, T_BASE_OPT)
    assert pts[0] == pytest.approx([2.0, 0.0, 0.24])        # 2 m ahead, at camera height


def test_scan_keeps_only_the_height_band_and_nearest_point():
    angle_min, n = -0.5, 10
    pts = np.array([[1.5, 0.0, 0.74],      # table top ahead: kept
                    [1.0, 0.0, 0.01],      # floor: dropped
                    [0.8, 0.0, 1.50],      # above the band: dropped
                    [2.5, 0.0, 0.40]])     # same bearing, farther: not the minimum
    r = scan_ranges(pts, 0.05, 1.0, angle_min, 0.1, n, 0.2, 3.0)
    beam = int((0.0 - angle_min) / 0.1)
    assert r[beam] == pytest.approx(1.5)
    assert np.isinf(np.delete(r, beam)).all()


def test_scan_bearing_bins_and_range_limits():
    pts = np.array([[1.0, 1.0, 0.5],       # 45 deg left
                    [0.1, 0.0, 0.5],       # closer than range_min
                    [5.0, 0.0, 0.5]])      # beyond range_max
    r = scan_ranges(pts, 0.05, 1.0, -1.0, 0.25, 8, 0.2, 3.0)
    assert r[int((math.pi / 4 + 1.0) / 0.25)] == pytest.approx(math.sqrt(2))
    assert np.isfinite(r).sum() == 1


def test_scan_empty_input_is_all_inf():
    assert np.isinf(scan_ranges(np.zeros((0, 3)), 0.05, 1.0, -0.5, 0.1, 10, 0.2, 3.0)).all()


def test_beam_layout_covers_the_field_of_view():
    angle_min, n = beam_layout(FX, 640, 0.01)
    assert angle_min == pytest.approx(-0.625, abs=0.01)
    assert n == 125
    assert angle_min + n * 0.01 == pytest.approx(-angle_min)
