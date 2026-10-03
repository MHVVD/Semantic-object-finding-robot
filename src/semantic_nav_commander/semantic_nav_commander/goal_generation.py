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
Goal generation for "go to <object>": where should the robot stand? No ROS imports.

For an object at O, candidate robot positions are sampled on circles of radius
r (the standoff distance, then fallbacks) around O. A candidate is REJECTED if:

    * it lies outside the costmap,
    * its cell is LETHAL (254), INSCRIBED (253) or NO_INFORMATION (255) -- the
      robot's centre there would collide, or nobody knows -- or above max_cost,
    * it is not in the object's LOCAL free space: its geodesic distance (through
      non-lethal cells) from the free cells nearest the object point exceeds
      max_detour x r. A point on the far side of a wall is a free cell only a
      few centimetres from the object in a straight line, but reachable only via
      the door: rejected. The far side of a chair is a short detour: accepted.
      (A straight line-of-sight test cannot tell the object's own lethal cells
      from a wall right behind it; the depth-based object point, however, lies
      on the side the camera saw it from, which seeds the flood fill there.)

Survivors are scored  score = distance_to_robot + cost_weight * cost / 252
(close to the robot, and away from obstacles), and each gets the yaw that
faces the object: yaw = atan2(Oy - y, Ox - x).

Nav2 costmap cell values (nav2_costmap_2d/cost_values.hpp):
    0 free, 1-252 inflated (decaying with distance from obstacles),
    253 inscribed (robot footprint would touch an obstacle),
    254 lethal (obstacle), 255 no information.
"""

from dataclasses import dataclass
import heapq
import math

FREE = 0
MAX_NON_OBSTACLE = 252
INSCRIBED = 253
LETHAL = 254
NO_INFORMATION = 255


@dataclass(frozen=True)
class CostGrid:
    """A 2D costmap: row-major uint8 costs, cell (0, 0) at `origin` (lower-left corner)."""

    size_x: int
    size_y: int
    resolution: float
    origin_x: float
    origin_y: float
    data: tuple                  # len == size_x * size_y, index = my * size_x + mx

    def world_to_cell(self, x, y):
        """World (map frame) -> (mx, my), or None outside the grid."""
        mx = math.floor((x - self.origin_x) / self.resolution)
        my = math.floor((y - self.origin_y) / self.resolution)
        if 0 <= mx < self.size_x and 0 <= my < self.size_y:
            return mx, my
        return None

    def cost(self, x, y):
        """Cost at a world point; NO_INFORMATION outside the grid."""
        cell = self.world_to_cell(x, y)
        if cell is None:
            return NO_INFORMATION
        return self.data[cell[1] * self.size_x + cell[0]]


@dataclass(frozen=True)
class Candidate:
    """A goal pose candidate: position, yaw facing the object, cell cost and score."""

    x: float
    y: float
    yaw: float
    cost: int
    score: float


def yaw_to_quaternion(yaw):
    """Rotation about +z by `yaw` -> quaternion (x, y, z, w) = (0, 0, sin(yaw/2), cos(yaw/2))."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def local_free_space(grid, object_xy, max_dist, seed_search=1.0):
    """
    Geodesic distances (metres) from the object's nearest free cells, up to max_dist.

    Seeds: the cells closest to the object point where the robot could STAND
    (cost below INSCRIBED; within one cell of the minimum distance, searched up
    to `seed_search` m away). Then Dijkstra over 8-connected non-lethal, known
    cells. Seeding from standable cells matters: a couch appears as a closed
    lethal outline with an unknown / inscribed interior, and the object point
    lies inside it -- seeding from the nearest non-lethal cell would trap the
    flood fill inside the couch. Returns {(mx, my): distance}.
    """
    ox, oy = object_xy
    res = grid.resolution
    reach = int(math.ceil((max_dist + seed_search) / res)) + 1
    c0 = (math.floor((ox - grid.origin_x) / res), math.floor((oy - grid.origin_y) / res))

    def passable(mx, my):
        return (0 <= mx < grid.size_x and 0 <= my < grid.size_y and
                grid.data[my * grid.size_x + mx] not in (LETHAL, NO_INFORMATION))

    def centre(mx, my):
        return grid.origin_x + (mx + 0.5) * res, grid.origin_y + (my + 0.5) * res

    def standable(mx, my):
        return (0 <= mx < grid.size_x and 0 <= my < grid.size_y and
                grid.data[my * grid.size_x + mx] < INSCRIBED)

    seed_cells = []
    r_seed = int(math.ceil(seed_search / res))
    for my in range(c0[1] - r_seed, c0[1] + r_seed + 1):
        for mx in range(c0[0] - r_seed, c0[0] + r_seed + 1):
            if standable(mx, my):
                cx, cy = centre(mx, my)
                d = math.hypot(cx - ox, cy - oy)
                if d <= seed_search:
                    seed_cells.append((d, mx, my))
    if not seed_cells:
        return {}
    d_min = min(d for d, _, _ in seed_cells)
    dist = {}
    heap = [(0.0, mx, my) for d, mx, my in seed_cells if d <= d_min + res]
    for _, mx, my in heap:
        dist[(mx, my)] = 0.0
    heapq.heapify(heap)
    steps = [(dx, dy, res * math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
             if dx or dy]
    while heap:
        d, mx, my = heapq.heappop(heap)
        if d > dist.get((mx, my), math.inf):
            continue
        for dx, dy, step in steps:
            nx, ny, nd = mx + dx, my + dy, d + step
            if (nd <= max_dist and abs(nx - c0[0]) <= reach and abs(ny - c0[1]) <= reach and
                    passable(nx, ny) and nd < dist.get((nx, ny), math.inf)):
                dist[(nx, ny)] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return dist


def candidate_goals(object_xy, robot_xy, grid, radii, n_samples, max_cost,
                    max_detour, cost_weight):
    """
    Return the valid goal candidates around an object, best (lowest score) first.

    Radii are tried in order; the first radius that yields any valid candidate
    wins (prefer the requested standoff, fall back further out / closer in).
    Returns [] if no radius works.
    """
    ox, oy = object_xy
    for r in radii:
        local = local_free_space(grid, object_xy, max_detour * r)
        found = []
        for k in range(n_samples):
            a = 2.0 * math.pi * k / n_samples
            x, y = ox + r * math.cos(a), oy + r * math.sin(a)
            c = grid.cost(x, y)
            if c in (LETHAL, INSCRIBED, NO_INFORMATION) or c > max_cost:
                continue
            if grid.world_to_cell(x, y) not in local:
                continue
            score = math.hypot(x - robot_xy[0], y - robot_xy[1]) + \
                cost_weight * c / MAX_NON_OBSTACLE
            found.append(Candidate(x, y, math.atan2(oy - y, ox - x), int(c), score))
        if found:
            return sorted(found, key=lambda cand: cand.score)
    return []


def credible_instances(objects, min_relative_evidence):
    """
    Drop instances with far less evidence than the best-observed one of the label.

    The semantic map confirms consistent detector mistakes too (a "bed" seen 16
    times at the couch, next to the real bed seen 129 times). Keeping only
    instances with >= min_relative_evidence x the highest observation_count
    stops "nearest" from picking such a phantom. 0 disables the filter.
    """
    if not objects:
        return []
    best = max(o.observation_count for o in objects)
    return [o for o in objects if o.observation_count >= min_relative_evidence * best]


def parse_aliases(items):
    """['fridge=refrigerator', 'sofa=couch'] -> {'fridge': 'refrigerator', 'sofa': 'couch'}."""
    out = {}
    for item in items:
        if not item.strip():
            continue
        alias, sep, label = item.partition('=')
        if not sep or not alias.strip() or not label.strip():
            raise ValueError(f'alias entry {item!r} is not "spoken=label"')
        out[alias.strip().lower()] = label.strip().lower()
    return out


def resolve_label(text, aliases):
    """Normalise a requested name: lower case, drop 'the'/'a', apply aliases."""
    words = [w for w in text.lower().replace('_', ' ').split() if w not in ('the', 'a', 'an')]
    name = ' '.join(words)
    return aliases.get(name, name)
