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
Geometry for scoring projected observations against ground truth. No ROS imports.

Error decomposition (2D, map frame): with the camera at C, the observation at
O and the true object centre at G, the unit ray r = (G - C) / |G - C| splits
the error e = O - G into
    range error   e_r = e . r          (< 0: observation closer than the centre)
    lateral error e_l = e x r (signed, perpendicular to the ray)
A depth camera measures the object's visible SURFACE, so e_r is expected to be
about minus the distance from the centre to the box face the camera sees
(box_exit_distance).
"""

import math


def distance_to_box(xy, centre, half_size_xy, yaw):
    """Distance from a point to an oriented 2D box (0 inside it)."""
    dx, dy = xy[0] - centre[0], xy[1] - centre[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = c * dx - s * dy, s * dx + c * dy               # point in the box frame
    ox = max(abs(lx) - half_size_xy[0], 0.0)
    oy = max(abs(ly) - half_size_xy[1], 0.0)
    return math.hypot(ox, oy)


def match_object(label, xy, objects, gate):
    """
    Same-label object whose FOOTPRINT is closest to `xy`, if within `gate` metres.

    Matching by distance to the footprint, not to the centre: a depth camera
    sees an object's surface, which for a 2 m bed is 1 m from its centre.
    Each object dict needs 'label', 'xy', 'half', 'yaw'. Returns (object, distance)
    or (None, None).
    """
    best, best_d = None, None
    for obj in objects:
        if obj['label'] != label:
            continue
        d = distance_to_box(xy, obj['xy'], obj['half'], obj['yaw'])
        if d <= gate and (best_d is None or d < best_d):
            best, best_d = obj, d
    return best, best_d


def error_components(obs_xy, truth_xy, camera_xy):
    """(range_error, lateral_error) of obs vs truth along / across the camera ray."""
    rx, ry = truth_xy[0] - camera_xy[0], truth_xy[1] - camera_xy[1]
    n = math.hypot(rx, ry)
    rx, ry = rx / n, ry / n
    ex, ey = obs_xy[0] - truth_xy[0], obs_xy[1] - truth_xy[1]
    return ex * rx + ey * ry, rx * ey - ry * ex


def box_exit_distance(direction_xy, half_size_xy, yaw):
    """
    Distance from an oriented box's centre to its boundary along `direction_xy`.

    The box has half extents (hx, hy) in its own frame, rotated by `yaw` in the
    map. Seen from a camera along the ray r, the visible face is this far in
    front of the centre (travelling along -r from the centre).
    """
    dx, dy = direction_xy
    n = math.hypot(dx, dy)
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = (c * dx - s * dy) / n, (s * dx + c * dy) / n      # direction in box frame
    hx, hy = half_size_xy
    tx = hx / abs(lx) if abs(lx) > 1e-12 else math.inf
    ty = hy / abs(ly) if abs(ly) > 1e-12 else math.inf
    return min(tx, ty)


def match_map_to_ground_truth(mapped, truth, gate):
    """
    One-to-one match of semantic-map objects to ground-truth objects.

    mapped: dicts with 'label', 'xy'.  truth: dicts with 'label', 'xy', 'half', 'yaw'.
    A pair is allowed if the labels are equal and the map point is within `gate`
    metres of the object's footprint; among allowed pairs the assignment minimising
    the total centre distance is chosen (Hungarian), so one ground-truth chair can
    never be claimed by two map chairs. Returns (matches, unmatched_map,
    unmatched_truth) with matches = [(map_index, truth_index, centre_error_m)].
    """
    from scipy.optimize import linear_sum_assignment
    big = 1e6
    matches = []
    for label in sorted({m['label'] for m in mapped} | {t['label'] for t in truth}):
        mi = [i for i, m in enumerate(mapped) if m['label'] == label]
        ti = [j for j, t in enumerate(truth) if t['label'] == label]
        if not mi or not ti:
            continue
        cost = [[math.dist(mapped[i]['xy'], truth[j]['xy'])
                 if distance_to_box(mapped[i]['xy'], truth[j]['xy'], truth[j]['half'],
                                    truth[j]['yaw']) <= gate else big
                 for j in ti] for i in mi]
        rows, cols = linear_sum_assignment(cost)
        matches += [(mi[r], ti[c], cost[r][c]) for r, c in zip(rows, cols) if cost[r][c] < big]
    matched_map = {m for m, _, _ in matches}
    matched_truth = {t for _, t, _ in matches}
    return (sorted(matches),
            [i for i in range(len(mapped)) if i not in matched_map],
            [j for j in range(len(truth)) if j not in matched_truth])
