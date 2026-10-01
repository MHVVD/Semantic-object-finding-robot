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
Depth sampling, pinhole deprojection and rigid transforms. No ROS imports.

Pipeline for one 2D detection (box centre (u, v) in pixels):

    depth_to_meters()  raw depth image -> float32 metres, invalid -> NaN
    shrink_box()       central part of the box (avoid background at the edges)
    robust_depth()     median of the valid depths inside it
    deproject()        (u, v, Z) -> (X, Y, Z) in the camera OPTICAL frame
                       X = (u - cx) Z / fx,  Y = (v - cy) Z / fy
    transform_point()  optical frame -> map frame (rotation quaternion + translation)

Optical frame convention (REP 103): +Z forward along the optical axis, +X right
in the image, +Y down in the image.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels (no distortion: the sim camera is ideal)."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, k):
        """Build from the row-major 3x3 K of sensor_msgs/CameraInfo."""
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))


def depth_to_meters(depth, encoding):
    """
    Convert a depth image to float32 metres with NaN for every invalid pixel.

    32FC1: already metres; NaN / +-inf / <= 0 mean "no return".
    16UC1: millimetres (OpenNI / RealSense convention); 0 means "no return".
    """
    if encoding == '32FC1':
        out = np.array(depth, dtype=np.float32, copy=True)
    elif encoding == '16UC1':
        out = np.asarray(depth, dtype=np.float32) / 1000.0
    else:
        raise ValueError(f'unsupported depth encoding {encoding!r} (want 32FC1 or 16UC1)')
    out[~np.isfinite(out) | (out <= 0.0)] = np.nan
    return out


def shrink_box(cx, cy, w, h, factor, image_w, image_h):
    """
    Integer pixel bounds [x0, x1) x [y0, y1) of the box scaled by `factor` about its centre.

    Clipped to the image; always at least one pixel wide/high when the centre
    is inside the image.
    """
    half_w, half_h = 0.5 * w * factor, 0.5 * h * factor
    x0 = int(math.floor(cx - half_w))
    x1 = int(math.ceil(cx + half_w))
    y0 = int(math.floor(cy - half_h))
    y1 = int(math.ceil(cy + half_h))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image_w, max(x1, x0 + 1)), min(image_h, max(y1, y0 + 1))
    return x0, x1, y0, y1


def robust_depth(depth_m, bounds, min_depth, max_depth, min_valid_fraction):
    """
    Median depth of the valid pixels inside `bounds`, or None.

    Valid = finite and within [min_depth, max_depth]. Returns None when fewer
    than `min_valid_fraction` of the pixels are valid (e.g. the box is mostly
    sky beyond the far clip, or the object is out of range).
    Returns (median_m, n_valid, n_total).
    """
    x0, x1, y0, y1 = bounds
    patch = depth_m[y0:y1, x0:x1]
    n_total = patch.size
    if n_total == 0:
        return None
    valid = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
    if valid.size == 0 or valid.size < min_valid_fraction * n_total:
        return None
    return float(np.median(valid)), int(valid.size), int(n_total)


def deproject(u, v, z, k):
    """
    Pixel (u, v) at depth z (metres along the optical axis) -> (X, Y, Z) optical frame.

    Inverse of the pinhole projection u = fx X / Z + cx, v = fy Y / Z + cy.
    Note z is the Z coordinate (what a depth camera reports), not the ray length.
    """
    return ((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z)


def project(x, y, z, k):
    """Optical-frame point -> pixel (u, v). Used by the tests and the evaluation."""
    return (k.fx * x / z + k.cx, k.fy * y / z + k.cy)


def quaternion_to_matrix(x, y, z, w):
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_point(point, translation, rotation_xyzw):
    """p_target = R p_source + t, with (R, t) the pose of the source frame in the target."""
    r = quaternion_to_matrix(*rotation_xyzw)
    return tuple(float(c) for c in r @ np.asarray(point, dtype=float) + np.asarray(translation))
