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
Frontier and camera-coverage exploration maths on occupancy grids. No ROS imports.

Grids are numpy arrays indexed [row = y, col = x] with
    occupancy map (nav_msgs/OccupancyGrid): -1 unknown, 0..free_max free, >= occ_min occupied
    traversability: a boolean array "the robot may stand here" (from the costmap).

Phase 1 -- map frontiers (Yamauchi 1997): a FRONTIER cell is a known-free cell
with an unknown 4-neighbour; 8-connected frontier cells form clusters; driving
to a cluster reveals the unknown space behind it.

Phase 2 -- camera coverage: the lidar finishes the map long before the camera
has looked at everything. Every occupied cell (walls, furniture) is a surface
an object may stand on or against; `mark_seen` ray-casts the camera's
horizontal field of view on the map and records which occupied cells it has
seen within the useful depth range. Clusters of UNSEEN occupied cells are
"view frontiers", and `viewpoint` picks a standable spot facing one.

Choosing a target trades exploitation (go where you are -- short drive)
against exploration (go where there is most to discover):
    score = geodesic_distance - gain_weight * cluster_size_m      (lower = better)
with the geodesic distance from one Dijkstra run over the traversable grid.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math

import numpy as np

N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N8 = N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class GridInfo:
    """Grid geometry: cell (col, row) has its lower-left corner at origin + res * (col, row)."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int

    def to_cell(self, x, y):
        return (int(math.floor((x - self.origin_x) / self.resolution)),
                int(math.floor((y - self.origin_y) / self.resolution)))

    def to_world(self, col, row):
        return (self.origin_x + (col + 0.5) * self.resolution,
                self.origin_y + (row + 0.5) * self.resolution)

    def inside(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height


@dataclass
class Cluster:
    """A connected set of cells (frontier or unseen surface)."""

    cells: list            # [(col, row)]
    centroid: tuple        # world (x, y)

    @property
    def size(self):
        return len(self.cells)


def frontier_mask(occ, free_max=0):
    """Boolean mask of frontier cells: known free with an unknown 4-neighbour."""
    free = (occ >= 0) & (occ <= free_max)
    unknown = occ < 0
    neighbour_unknown = np.zeros_like(unknown)
    neighbour_unknown[1:, :] |= unknown[:-1, :]
    neighbour_unknown[:-1, :] |= unknown[1:, :]
    neighbour_unknown[:, 1:] |= unknown[:, :-1]
    neighbour_unknown[:, :-1] |= unknown[:, 1:]
    return free & neighbour_unknown


def clusters(mask, info, min_cells=1):
    """8-connected clusters of True cells with at least min_cells cells."""
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    rows, cols = np.nonzero(mask)
    for r0, c0 in zip(rows, cols):
        if seen[r0, c0]:
            continue
        queue, cells = deque([(c0, r0)]), []
        seen[r0, c0] = True
        while queue:
            c, r = queue.popleft()
            cells.append((c, r))
            for dc, dr in N8:
                nc, nr = c + dc, r + dr
                if info.inside(nc, nr) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nc, nr))
        if len(cells) >= min_cells:
            pts = np.array([info.to_world(c, r) for c, r in cells])
            out.append(Cluster(cells, tuple(pts.mean(axis=0))))
    return out


def geodesic(traversable, start):
    """Dijkstra distances (cells, 8-connected) from start over traversable; inf elsewhere."""
    h, w = traversable.shape
    dist = np.full((h, w), np.inf)
    c0, r0 = start
    if not (0 <= c0 < w and 0 <= r0 < h) or not traversable[r0, c0]:
        return dist
    dist[r0, c0] = 0.0
    heap = [(0.0, c0, r0)]
    steps = [(dc, dr, math.hypot(dc, dr)) for dc, dr in N8]
    while heap:
        d, c, r = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dc, dr, step in steps:
            nc, nr = c + dc, r + dr
            if 0 <= nc < w and 0 <= nr < h and traversable[nr, nc] and d + step < dist[nr, nc]:
                dist[nr, nc] = d + step
                heapq.heappush(heap, (d + step, nc, nr))
    return dist


def frontier_goal(cluster, dist, info):
    """
    Return the reachable cell of a frontier cluster closest to its centroid, or None.

    Returns ((x, y), geodesic distance in metres). Frontier cells next to walls
    may be untraversable (inscribed); any reachable cell of the cluster will do.
    """
    best = None
    for c, r in cluster.cells:
        if not np.isfinite(dist[r, c]):
            continue
        x, y = info.to_world(c, r)
        d_centroid = math.hypot(x - cluster.centroid[0], y - cluster.centroid[1])
        if best is None or d_centroid < best[0]:
            best = (d_centroid, (x, y), dist[r, c] * info.resolution)
    return None if best is None else (best[1], best[2])


def mark_seen(occ, seen, info, cam_x, cam_y, cam_yaw, hfov, max_range,
              occ_min=65, ray_step_rad=0.005):
    """
    Ray-cast the camera's horizontal field of view; mark occupied cells it can see.

    Each ray walks outward in half-cell steps and stops at the first occupied
    cell (marked seen if within max_range) or at an unknown cell. Returns the
    number of newly seen cells.
    """
    new = 0
    step = info.resolution / 2.0
    n_steps = int(max_range / step)
    for a in np.arange(cam_yaw - hfov / 2.0, cam_yaw + hfov / 2.0 + 1e-9, ray_step_rad):
        ca, sa = math.cos(a), math.sin(a)
        for k in range(1, n_steps + 1):
            c, r = info.to_cell(cam_x + k * step * ca, cam_y + k * step * sa)
            if not info.inside(c, r):
                break
            v = occ[r, c]
            if v < 0:
                break
            if v >= occ_min:
                if not seen[r, c]:
                    seen[r, c] = True
                    new += 1
                break
    return new


def first_hit(occ, info, x0, y0, x1, y1, occ_min=65):
    """First occupied cell on the segment (x0, y0) -> (x1, y1), or None."""
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(length / (info.resolution / 2.0)))
    for k in range(n + 1):
        t = k / n
        c, r = info.to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        if info.inside(c, r) and occ[r, c] >= occ_min:
            return (c, r)
    return None


def viewpoint(cluster, occ, dist, info, min_range, max_range, n_angles=24, occ_min=65):
    """
    Return a reachable spot from which the camera sees the (unseen) surface cluster.

    Candidates on circles around the cluster centroid at distances between
    min_range and max_range; valid if reachable (finite geodesic distance) and
    the first occupied cell on the line of sight towards the centroid belongs
    to the cluster itself (so the surface is not hidden behind another wall).
    Returns ((x, y, yaw), geodesic metres) for the closest valid one, or None.
    """
    members = set(cluster.cells)
    cx, cy = cluster.centroid
    best = None
    for rng in np.linspace(min_range, max_range, 4):
        for k in range(n_angles):
            a = 2.0 * math.pi * k / n_angles
            x, y = cx + rng * math.cos(a), cy + rng * math.sin(a)
            c, r = info.to_cell(x, y)
            if not info.inside(c, r) or not np.isfinite(dist[r, c]):
                continue
            hit = first_hit(occ, info, x, y, cx, cy, occ_min)
            if hit is None or hit not in members:
                continue
            d = dist[r, c] * info.resolution
            if best is None or d < best[1]:
                best = ((x, y, math.atan2(cy - y, cx - x)), d)
    return best


def score(geodesic_m, cluster_size_cells, resolution, gain_weight):
    """Lower is better: drive cost minus weighted information gain (cluster length)."""
    return geodesic_m - gain_weight * cluster_size_cells * resolution


def near_any(xy, points, radius):
    return any(math.hypot(xy[0] - p[0], xy[1] - p[1]) < radius for p in points)


def traversable_from_map(occ, info, clearance, free_max=0, occ_min=65):
    """Known-free cells farther than `clearance` metres from every occupied cell."""
    from scipy.ndimage import distance_transform_edt
    occupied = occ >= occ_min
    if not occupied.any():
        return (occ >= 0) & (occ <= free_max)
    dist_to_occ = distance_transform_edt(~occupied) * info.resolution
    return (occ >= 0) & (occ <= free_max) & (dist_to_occ > clearance)


def surface_mask(occ, free_max=0, occ_min=65):
    """Occupied cells with a known-free 4-neighbour: the faces a camera can see."""
    occupied = occ >= occ_min
    free = (occ >= 0) & (occ <= free_max)
    next_to_free = np.zeros_like(free)
    next_to_free[1:, :] |= free[:-1, :]
    next_to_free[:-1, :] |= free[1:, :]
    next_to_free[:, 1:] |= free[:, :-1]
    next_to_free[:, :-1] |= free[:, 1:]
    return occupied & next_to_free


def tiles(mask, info, tile_m, min_cells=1):
    """
    Group True cells into square tiles of tile_m metres -> [Cluster].

    Used for unseen surfaces instead of connected clusters: a room's walls are
    one connected ring whose centroid is the middle of the room, which is no
    place to look at. Tiles give compact targets along long walls.
    """
    rows, cols = np.nonzero(mask)
    groups = {}
    for r, c in zip(rows, cols):
        x, y = info.to_world(c, r)
        groups.setdefault((math.floor(x / tile_m), math.floor(y / tile_m)), []).append((c, r))
    out = []
    for cells in groups.values():
        if len(cells) >= min_cells:
            pts = np.array([info.to_world(c, r) for c, r in cells])
            out.append(Cluster(cells, tuple(pts.mean(axis=0))))
    return out
