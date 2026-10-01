#!/usr/bin/env python3
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
Autonomous exploration for SEMANTIC mapping: map frontiers, then camera coverage.

Role:
    Phase 1 (map): drive to frontiers of the live SLAM map (known free next to
    unknown) until none are left -- classic frontier exploration.
    Phase 2 (camera): the 12 m lidar completes the map long before the camera
    (57 deg wide, ~3 m useful depth) has looked at everything, and objects are
    found by the camera. The node ray-casts the camera's field of view on the
    map continuously and records which surface cells (occupied cells with a free
    neighbour) it has seen within camera_range_m. Unseen surface cells, grouped
    into tiles, become targets; the robot drives to a viewpoint facing each.
    When neither phase has a reachable target left, exploration is complete
    (optionally returning to the start) and "complete ..." is published.

    Target choice: score = geodesic distance - gain_weight x target size (m);
    geodesic distances come from one Dijkstra run over the map's traversable
    cells (free and > clearance_m from obstacles). The robot COMMITS to a target
    until Nav2 finishes, the target stops being a frontier (already revealed),
    or progress stalls (then it is blacklisted).

Topics / actions:
    Subscribes   <map_topic>          nav_msgs/OccupancyGrid (slam_toolbox, transient local)
    Action cli   <nav_action_name>    nav2_msgs/action/NavigateToPose
    Publishes    <status_topic>       std_msgs/String ("phase 1: ...", "complete ...")
    Publishes    <markers_topic>      visualization_msgs/MarkerArray (targets, goal)
    TF           <map_frame> <- <robot_frame>

Parameters (config/params.yaml, section frontier_explorer):
    map_topic, nav_action_name, status_topic, markers_topic, map_frame, robot_frame
    clearance_m          (double) traversable = free and farther than this from obstacles
    min_frontier_m       (double) ignore frontier clusters shorter than this
    gain_weight          (double) metres of driving one metre of target is worth
    decision_period_s    (double) how often to (re)decide
    progress_timeout_s   (double) cancel + blacklist if the robot does not move closer
    goal_timeout_s       (double) hard cap per goal (near the goal only Nav2 decides,
                         so a goal under a table could otherwise take minutes)
    coverage_budget_s    (double) sim-time budget for phase 2 (diminishing returns)
    blacklist_radius_m   (double) no new targets this close to a failed one
    camera_coverage      (bool)   run phase 2
    camera_hfov_rad, camera_range_m   camera model for the coverage ray cast
    coverage_tile_m      (double) tile size for unseen surfaces
    min_unseen_cells     (int)    ignore tiles with fewer unseen surface cells
    view_min_m, view_max_m (double) viewpoint distance from an unseen tile
    return_to_start      (bool)
    use_sim_time         (bool)
"""

import math
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from semantic_nav_bringup.frontiers import (clusters, frontier_goal, frontier_mask, geodesic,
                                            GridInfo, mark_seen, near_any, score,
                                            surface_mask, tiles, traversable_from_map,
                                            viewpoint)
from semantic_nav_bringup.nav_utils import quaternion_from_yaw, yaw_from_quaternion
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

NEAR_GOAL_M = 0.4

PARAMETERS = {
    'map_topic': Parameter.Type.STRING,
    'nav_action_name': Parameter.Type.STRING,
    'status_topic': Parameter.Type.STRING,
    'markers_topic': Parameter.Type.STRING,
    'map_frame': Parameter.Type.STRING,
    'robot_frame': Parameter.Type.STRING,
    'clearance_m': Parameter.Type.DOUBLE,
    'min_frontier_m': Parameter.Type.DOUBLE,
    'gain_weight': Parameter.Type.DOUBLE,
    'decision_period_s': Parameter.Type.DOUBLE,
    'progress_timeout_s': Parameter.Type.DOUBLE,
    'goal_timeout_s': Parameter.Type.DOUBLE,
    'coverage_budget_s': Parameter.Type.DOUBLE,
    'blacklist_radius_m': Parameter.Type.DOUBLE,
    'camera_coverage': Parameter.Type.BOOL,
    'camera_hfov_rad': Parameter.Type.DOUBLE,
    'camera_range_m': Parameter.Type.DOUBLE,
    'coverage_tile_m': Parameter.Type.DOUBLE,
    'min_unseen_cells': Parameter.Type.INTEGER,
    'view_min_m': Parameter.Type.DOUBLE,
    'view_max_m': Parameter.Type.DOUBLE,
    'return_to_start': Parameter.Type.BOOL,
}


class FrontierExplorer(Node):
    """Map-frontier + camera-coverage exploration driving Nav2."""

    def __init__(self):
        super().__init__('frontier_explorer')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        self.p = p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.map = self.info = None
        self.seen_world = set()          # seen surface cells as world-integer keys
        self.blacklist = []              # (x, y) of failed targets
        self.goal = None                 # dict: kind, xy, yaw, handle, best_dist, last_progress
        self.phase = 1
        self.start_xy = None
        self.finished = False
        self.t0 = time.monotonic()
        self.goals_sent = 0
        self.phase2_start = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, p['map_topic'], self.on_map, latched)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.nav = ActionClient(self, NavigateToPose, p['nav_action_name'])
        self.status_pub = self.create_publisher(String, p['status_topic'], latched)
        self.marker_pub = self.create_publisher(MarkerArray, p['markers_topic'], 10)
        self.create_timer(0.5, self.update_coverage)
        self.create_timer(p['decision_period_s'], self.decide)
        self.status('waiting for map, TF and Nav2')

    # -------------------------------------------------------------- inputs
    def on_map(self, msg):
        i = msg.info
        self.info = GridInfo(i.resolution, i.origin.position.x, i.origin.position.y,
                             i.width, i.height)
        self.map = np.asarray(msg.data, dtype=np.int16).reshape(i.height, i.width)

    def pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.p['map_frame'], self.p['robot_frame'],
                                                 Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def key(self, c, r):
        """World-anchored integer key of a cell (survives map growth / origin changes)."""
        x, y = self.info.to_world(c, r)
        return (round(x / self.info.resolution), round(y / self.info.resolution))

    def seen_array(self):
        seen = np.zeros(self.map.shape, dtype=bool)
        res = self.info.resolution
        for kx, ky in self.seen_world:
            c, r = self.info.to_cell(kx * res, ky * res)
            if self.info.inside(c, r):
                seen[r, c] = True
        return seen

    def update_coverage(self):
        """Ray-cast the camera field of view and remember the surfaces it has seen."""
        pose = self.pose()
        if pose is None or self.map is None:
            return
        seen = np.zeros(self.map.shape, dtype=bool)
        mark_seen(self.map, seen, self.info, pose[0], pose[1], pose[2],
                  self.p['camera_hfov_rad'], self.p['camera_range_m'])
        rows, cols = np.nonzero(seen)
        self.seen_world.update(self.key(c, r) for r, c in zip(rows, cols))

    # -------------------------------------------------------------- helpers
    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def status(self, text):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def send(self, kind, xy, yaw, size):
        goal = PoseStamped()
        goal.header.frame_id = self.p['map_frame']
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = float(xy[0]), float(xy[1])
        q = quaternion_from_yaw(yaw)
        (goal.pose.orientation.x, goal.pose.orientation.y,
         goal.pose.orientation.z, goal.pose.orientation.w) = q
        self.goal = {'kind': kind, 'xy': xy, 'yaw': yaw, 'handle': None, 'done': False,
                     'best_dist': math.inf, 'last_progress': time.monotonic(),
                     'sent': self.sim_now()}
        self.goals_sent += 1
        self.get_logger().info(f'goal {self.goals_sent}: {kind} at ({xy[0]:.2f}, {xy[1]:.2f}), '
                               f'target size {size:.1f} m')
        future = self.nav.send_goal_async(NavigateToPose.Goal(pose=goal))
        future.add_done_callback(lambda f, g=self.goal: self.on_accepted(f, g))
        self.publish_markers()

    def on_accepted(self, future, goal):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('goal rejected by Nav2 (still activating?)')
            goal['done'] = True
            return
        goal['handle'] = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_result(f, goal))

    def on_result(self, future, goal):
        status = future.result().status
        goal['done'] = True
        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(f'Nav2 aborted the {goal["kind"]} goal -> blacklisted')
            self.blacklist.append(goal['xy'])

    def cancel(self, reason, blacklist):
        g = self.goal
        self.get_logger().info(f'abandoning {g["kind"]} goal: {reason}')
        if blacklist:
            self.blacklist.append(g['xy'])
        if g['handle'] is not None:
            g['handle'].cancel_goal_async()
        g['done'] = True

    # -------------------------------------------------------------- decisions
    def decide(self):
        if self.finished or self.map is None:
            return
        pose = self.pose()
        if pose is None or not self.nav.server_is_ready():
            return
        if self.start_xy is None:
            self.start_xy = pose[:2]
        if self.goal is not None and not self.goal['done']:
            self.supervise(pose)
            return
        occ, info = self.map, self.info
        trav = traversable_from_map(occ, info, self.p['clearance_m'])
        start = info.to_cell(pose[0], pose[1])
        if info.inside(*start) and not trav[start[1], start[0]]:
            # The robot itself may sit inside the clearance band (near a wall): plan
            # from it anyway.
            trav[start[1], start[0]] = True
        dist = geodesic(trav, start)
        target = self.next_frontier(occ, info, dist, pose) if self.phase == 1 else None
        if target is None and self.phase == 1:
            self.phase = 2 if self.p['camera_coverage'] else 3
            self.status(f'phase 1 done: no reachable frontiers left after '
                        f'{time.monotonic() - self.t0:.0f} s wall -- '
                        + ('now covering unseen surfaces with the camera'
                           if self.phase == 2 else 'finishing'))
        if target is None and self.phase == 2:
            if self.phase2_start is None:
                self.phase2_start = self.sim_now()
            if self.sim_now() - self.phase2_start > self.p['coverage_budget_s']:
                self.phase = 3
                self.status(f'phase 2 done: coverage budget of '
                            f'{self.p["coverage_budget_s"]:.0f} s used up')
            else:
                target = self.next_view(occ, info, dist)
                if target is None:
                    self.phase = 3
                    self.status('phase 2 done: every reachable surface has been seen')
        if target is not None:
            self.send(*target)
            return
        self.finish(pose)

    def next_frontier(self, occ, info, dist, pose):
        min_cells = max(1, int(self.p['min_frontier_m'] / info.resolution))
        best = None
        for cl in clusters(frontier_mask(occ), info, min_cells):
            g = frontier_goal(cl, dist, info)
            if g is None or near_any(g[0], self.blacklist, self.p['blacklist_radius_m']):
                continue
            s = score(g[1], cl.size, info.resolution, self.p['gain_weight'])
            if best is None or s < best[0]:
                yaw = math.atan2(g[0][1] - pose[1], g[0][0] - pose[0])
                best = (s, ('frontier', g[0], yaw, cl.size * info.resolution))
        return None if best is None else best[1]

    def next_view(self, occ, info, dist):
        unseen = surface_mask(occ) & ~self.seen_array()
        best = None
        for t in tiles(unseen, info, self.p['coverage_tile_m'], self.p['min_unseen_cells']):
            if near_any(t.centroid, self.blacklist, self.p['blacklist_radius_m']):
                continue
            v = viewpoint(t, occ, dist, info, self.p['view_min_m'], self.p['view_max_m'])
            if v is None:
                self.blacklist.append(t.centroid)        # no place to see it from
                continue
            s = score(v[1], t.size, info.resolution, self.p['gain_weight'])
            if best is None or s < best[0]:
                (x, y, yaw), _ = v
                best = (s, ('view', (x, y), yaw, t.size * info.resolution), t.centroid)
        if best is None:
            return None
        # Blacklist the tile itself once targeted: even if the view is partial we
        # do not come back to it forever.
        self.blacklist.append(best[2])
        return best[1]

    def supervise(self, pose):
        """While navigating: abandon goals that are pointless or stuck."""
        g = self.goal
        d = math.hypot(g['xy'][0] - pose[0], g['xy'][1] - pose[1])
        now = time.monotonic()
        if d < g['best_dist'] - 0.1 or d < NEAR_GOAL_M:
            # Near the goal the robot only turns to the final heading: distance no
            # longer shrinks, but that is not "stuck" -- Nav2 will finish or abort.
            g['best_dist'], g['last_progress'] = min(d, g['best_dist']), now
        elif now - g['last_progress'] > self.p['progress_timeout_s']:
            self.cancel(f'no progress for {self.p["progress_timeout_s"]:.0f} s', True)
            return
        if self.sim_now() - g['sent'] > self.p['goal_timeout_s']:
            self.cancel(f'goal timeout ({self.p["goal_timeout_s"]:.0f} s)', True)
            return
        if g['kind'] == 'frontier':
            c, r = self.info.to_cell(*g['xy'])
            m = frontier_mask(self.map)
            r0, r1 = max(0, r - 5), min(m.shape[0], r + 6)
            c0, c1 = max(0, c - 5), min(m.shape[1], c + 6)
            if not m[r0:r1, c0:c1].any():
                # The unknown space behind it has been seen already: move on.
                self.cancel('frontier already revealed', False)

    def finish(self, pose):
        if self.phase == 3 and self.p['return_to_start'] and self.start_xy is not None and \
                math.hypot(pose[0] - self.start_xy[0], pose[1] - self.start_xy[1]) > 0.5:
            self.phase = 4
            self.send('return', self.start_xy, 0.0, 0.0)
            return
        self.finished = True
        seen_total = len(self.seen_world)
        self.status(f'complete: {time.monotonic() - self.t0:.0f} s wall, {self.goals_sent} goals, '
                    f'{len(self.blacklist)} blacklisted, {seen_total} surface cells seen')

    def publish_markers(self):
        arr = MarkerArray()
        if self.goal is not None:
            m = Marker(ns='explore_goal', id=0, type=Marker.ARROW, action=Marker.ADD)
            m.header.frame_id = self.p['map_frame']
            m.pose.position.x, m.pose.position.y = self.goal['xy']
            q = quaternion_from_yaw(self.goal['yaw'])
            m.pose.orientation.x, m.pose.orientation.y = q[0], q[1]
            m.pose.orientation.z, m.pose.orientation.w = q[2], q[3]
            m.scale.x, m.scale.y, m.scale.z = 0.5, 0.08, 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = (
                (0.1, 0.5, 1.0, 1.0) if self.goal['kind'] == 'frontier' else (1.0, 0.6, 0.0, 1.0))
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    try:
        rclpy.init(args=args)
        rclpy.spin(FrontierExplorer())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
