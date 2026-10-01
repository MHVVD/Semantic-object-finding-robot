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

"""Unit tests for depth sampling, deprojection and transforms (known inputs -> outputs)."""

import math

import numpy as np
import pytest
from semantic_nav_perception.projection import (deproject, depth_to_meters, Intrinsics,
                                                project, quaternion_to_matrix, robust_depth,
                                                shrink_box, transform_point)

# The simulated OAK-D: 640x480, horizontal FOV 1.25 rad -> fx = 320 / tan(0.625).
FX = 320.0 / math.tan(0.625)
K = Intrinsics(fx=FX, fy=FX, cx=320.0, cy=240.0)


def quat_from_rpy(roll, pitch, yaw):
    """Fixed-axis roll-pitch-yaw (URDF convention) -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


# ---------------------------------------------------------------- intrinsics
def test_intrinsics_from_camera_info_k():
    k = Intrinsics.from_k([443.4, 0.0, 320.5, 0.0, 441.0, 240.5, 0.0, 0.0, 1.0])
    assert (k.fx, k.fy, k.cx, k.cy) == (443.4, 441.0, 320.5, 240.5)


def test_sim_focal_length_matches_fov():
    assert FX == pytest.approx(443.53, abs=0.01)   # what the sim CameraInfo reports


# ---------------------------------------------------------------- deprojection
def test_principal_point_lies_on_the_optical_axis():
    assert deproject(320.0, 240.0, 2.0, K) == pytest.approx((0.0, 0.0, 2.0))


def test_deproject_hand_computed():
    # 100 px right of and 100 px above the centre, 2 m away:
    # X = 100 * 2 / 443.53 = 0.451 m (right), Y = -0.451 m (up is -Y in the optical frame).
    x, y, z = deproject(420.0, 140.0, 2.0, K)
    assert x == pytest.approx(100 * 2 / FX)
    assert y == pytest.approx(-100 * 2 / FX)
    assert z == 2.0


def test_image_edge_is_half_the_fov():
    # The right image edge (u = 640) is at angle hfov/2 = 0.625 rad from the axis.
    x, _, z = deproject(640.0, 240.0, 1.0, K)
    assert math.atan2(x, z) == pytest.approx(0.625)


def test_deproject_scales_linearly_with_depth():
    p1 = np.array(deproject(100.0, 400.0, 1.0, K))
    p3 = np.array(deproject(100.0, 400.0, 3.0, K))
    np.testing.assert_allclose(p3, 3 * p1)          # same ray, 3x further


@pytest.mark.parametrize('u,v,z', [(0, 0, 0.5), (639, 479, 4.0), (123.4, 321.0, 2.2)])
def test_project_deproject_round_trip(u, v, z):
    assert project(*deproject(u, v, z, K), K) == pytest.approx((u, v))


def test_non_square_pixels_and_offset_principal_point():
    k = Intrinsics(fx=600.0, fy=300.0, cx=300.0, cy=200.0)
    assert deproject(360.0, 230.0, 3.0, k) == pytest.approx((0.3, 0.3, 3.0))


# ---------------------------------------------------------------- frames
OPTICAL_IN_LINK = quat_from_rpy(-math.pi / 2, 0.0, -math.pi / 2)   # from the URDF


@pytest.mark.parametrize('optical,link', [
    ((0, 0, 1), (1, 0, 0)),     # optical forward (+Z)  = link forward (+X)
    ((1, 0, 0), (0, -1, 0)),    # image right (+X)      = link right (-Y)
    ((0, 1, 0), (0, 0, -1)),    # image down (+Y)       = link down (-Z)
])
def test_optical_to_camera_link_axes(optical, link):
    assert transform_point(optical, (0, 0, 0), OPTICAL_IN_LINK) == pytest.approx(link, abs=1e-12)


def test_rotation_matrix_is_orthonormal():
    r = quaternion_to_matrix(*quat_from_rpy(0.3, -0.2, 1.1))
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_transform_point_rotation_then_translation():
    # Camera at (1, 2, 0.25) in the map, yawed +90 deg: its forward is map +Y.
    q = quat_from_rpy(0.0, 0.0, math.pi / 2)
    assert transform_point((3.0, 0.0, 0.0), (1.0, 2.0, 0.25), q) == pytest.approx((1, 5, 0.25))


def test_full_chain_pixel_to_map():
    # Robot at map (2, 1) facing +Y; camera 0.24 m up on it. Pixel at the image centre,
    # depth 2.5 m -> object 2.5 m in front of the camera = map (2, 3.5, 0.24).
    cam_in_map_q = quat_from_rpy(0.0, 0.0, math.pi / 2)
    p_link = transform_point(deproject(320.0, 240.0, 2.5, K), (0, 0, 0), OPTICAL_IN_LINK)
    p_map = transform_point(p_link, (2.0, 1.0, 0.24), cam_in_map_q)
    assert p_map == pytest.approx((2.0, 3.5, 0.24))


# ---------------------------------------------------------------- depth encodings
def test_depth_32fc1_invalid_values_become_nan():
    raw = np.array([[2.5, np.nan, np.inf], [-np.inf, 0.0, -1.0]], dtype=np.float32)
    out = depth_to_meters(raw, '32FC1')
    assert out[0, 0] == pytest.approx(2.5)
    assert np.isnan(out).sum() == 5


def test_depth_16uc1_is_millimetres_and_zero_is_invalid():
    out = depth_to_meters(np.array([[0, 1500, 65535]], dtype=np.uint16), '16UC1')
    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(1.5) and out[0, 2] == pytest.approx(65.535)


def test_depth_to_meters_does_not_modify_input():
    raw = np.array([[np.inf]], dtype=np.float32)
    depth_to_meters(raw, '32FC1')
    assert np.isinf(raw[0, 0])


def test_depth_unknown_encoding_rejected():
    with pytest.raises(ValueError):
        depth_to_meters(np.zeros((1, 1)), 'mono8')


# ---------------------------------------------------------------- box sampling
def test_shrink_box_half():
    # 100x60 box centred at (200, 150), shrink 0.5 -> 50x30 centred on the same point.
    assert shrink_box(200, 150, 100, 60, 0.5, 640, 480) == (175, 225, 135, 165)


def test_shrink_box_clips_to_image_and_stays_non_empty():
    assert shrink_box(5, 5, 100, 100, 0.5, 640, 480) == (0, 30, 0, 30)
    assert shrink_box(10.2, 10.2, 0.1, 0.1, 0.5, 640, 480) == (10, 11, 10, 11)


def test_median_ignores_background_but_mean_does_not():
    # A chair 2 m away covers 60 % of the patch; the wall behind it (5 m) shows
    # through the other 40 %. The median is the chair, the mean is neither.
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    depth[:4, :] = 5.0
    median, n_valid, n_total = robust_depth(depth, (0, 10, 0, 10), 0.3, 10.0, 0.2)
    assert median == pytest.approx(2.0) and (n_valid, n_total) == (100, 100)
    assert float(depth.mean()) == pytest.approx(3.2)


def test_robust_depth_skips_invalid_and_out_of_range():
    depth = np.full((4, 4), np.nan, dtype=np.float32)
    depth[0, :] = 1.0          # 4 valid pixels
    depth[1, :] = 9.0          # beyond max_depth
    median, n_valid, _ = robust_depth(depth, (0, 4, 0, 4), 0.3, 4.0, 0.2)
    assert median == pytest.approx(1.0) and n_valid == 4


def test_robust_depth_none_when_too_few_valid():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[0, 0] = 1.0                                   # 1 % valid
    assert robust_depth(depth, (0, 10, 0, 10), 0.3, 4.0, 0.2) is None
    assert robust_depth(depth, (0, 10, 0, 10), 0.3, 4.0, 0.0) is not None
