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
Depth image -> height-filtered 2D scan for the costmap. No ROS imports.

Why: the 2D lidar is ~0.2 m above the floor, so a dining table is four legs to
it and the space underneath looks free; the robot drove in and got trapped
(PROBLEMS_LOG #76, #93). The depth camera sees the table top. Every depth pixel
is deprojected (pinhole, optical frame), transformed into the robot frame
(base_link: on the floor, x forward, y left, z up), and kept if its height is
between min_height and max_height. For each bearing bin the nearest kept point
becomes the scan range; bins with no point are +inf (free as far as the
camera saw, so the costmap may clear them).
"""

import math

import numpy as np


def depth_points(depth_m, fx, fy, cx, cy, decimation):
    """
    Deproject every `decimation`-th pixel of a metric depth image (NaN = invalid).

    Returns an (N, 3) array of points in the camera OPTICAL frame
    (z forward, x right, y down): X = (u - cx) Z / fx, Y = (v - cy) Z / fy.
    """
    d = depth_m[::decimation, ::decimation]
    v, u = np.mgrid[0:depth_m.shape[0]:decimation, 0:depth_m.shape[1]:decimation]
    valid = np.isfinite(d)
    z = d[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def transform_points(points, rotation, translation):
    """Apply p' = R p + t to an (N, 3) array."""
    return points @ np.asarray(rotation).T + np.asarray(translation)


def scan_ranges(points, min_height, max_height, angle_min, angle_increment, n_beams,
                range_min, range_max):
    """
    Nearest in-band point per bearing bin, for points in the robot frame (z up).

    Beam i covers bearings [angle_min + i * inc, angle_min + (i + 1) * inc).
    Returns n_beams ranges (horizontal distance); +inf where no point fell in the bin.
    """
    ranges = np.full(n_beams, np.inf)
    if len(points) == 0:
        return ranges
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.hypot(x, y)
    keep = (z >= min_height) & (z <= max_height) & (r >= range_min) & (r <= range_max)
    if not keep.any():
        return ranges
    beam = np.floor((np.arctan2(y[keep], x[keep]) - angle_min) / angle_increment).astype(int)
    inside = (beam >= 0) & (beam < n_beams)
    np.minimum.at(ranges, beam[inside], r[keep][inside])
    return ranges


def beam_layout(fx, width, angle_increment):
    """(angle_min, n_beams) covering the camera's horizontal field of view, symmetric."""
    half_fov = math.atan((width / 2.0) / fx)
    n_beams = int(math.ceil(2.0 * half_fov / angle_increment))
    return -n_beams * angle_increment / 2.0, n_beams
