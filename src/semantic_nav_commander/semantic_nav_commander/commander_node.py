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
Turn "go to <label>" requests into Nav2 NavigateToPose goals.

Role:
    1. Resolve the requested name ("the fridge" -> "refrigerator" via aliases).
    2. Ask the semantic map for every confirmed instance (ListObjects).
    3. For each instance, generate goal candidates on circles around it
       (goal_generation.py: standoff radii, rejecting lethal / inscribed /
       unknown cells of the global costmap and points outside the object's
       local free space; yaw faces the object).
    4. "Nearest reachable": ask Nav2's planner (ComputePathToPose) for a path to
       the best candidates of each instance and keep the shortest PATH, not the
       shortest straight line (an object behind a wall may be far to drive).
    5. Send NavigateToPose, log feedback, and answer the service call with the
       final result. A new GoTo (or the cancel service) cancels a running one.

    The GoTo callback is a coroutine (async def): it awaits the map query, the
    planner and the navigation result without blocking the executor, so
    feedback, the cancel service and a superseding GoTo are processed meanwhile.
    A MultiThreadedExecutor plus a ReentrantCallbackGroup lets the blocking
    bits (waiting for action servers) run without stalling everything else.

Topics / services / actions:
    Service      <goto_service>        semantic_nav_interfaces/GoTo
    Service      <cancel_service>      std_srvs/Trigger (stop the current goal)
    Client       <list_objects_service> semantic_nav_interfaces/ListObjects
    Action cli   <nav_action_name>     nav2_msgs/action/NavigateToPose
    Action cli   <plan_action_name>    nav2_msgs/action/ComputePathToPose
    Subscribes   <costmap_topic>       nav2_msgs/Costmap (raw uint8 global costmap)
    TF           <goal_frame> <- <robot_frame>

Parameters (config/params.yaml, section commander_node):
    goto_service, cancel_service, list_objects_service   (string)
    nav_action_name, plan_action_name, costmap_topic     (string)
    goal_frame, robot_frame                              (string)
    standoff_radii_m     (double[]) standoff distance first, then fallbacks
    num_samples          (int)    candidates per circle
    max_goal_cost        (int)    highest acceptable costmap cost at the goal (<= 252)
    max_detour_factor    (double) local free space limit = factor x radius
    cost_weight          (double) metres of score per unit of normalised cost
    candidates_per_object (int)   planner calls per instance before giving up on it
    min_relative_evidence (double) ignore instances seen < this x the best-seen one
    label_aliases        (string[]) "spoken=label", e.g. "fridge=refrigerator"
    server_timeout_s     (double) wait for services / action servers
    feedback_period_s    (double) feedback logging period
    use_sim_time         (bool)
"""

from collections import Counter
import math
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.msg import Costmap
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from semantic_nav_commander.goal_generation import (candidate_goals, CostGrid,
                                                    credible_instances, parse_aliases,
                                                    resolve_label, yaw_to_quaternion)
from semantic_nav_interfaces.srv import GoTo, ListObjects
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {
    'goto_service': Parameter.Type.STRING,
    'cancel_service': Parameter.Type.STRING,
    'list_objects_service': Parameter.Type.STRING,
    'nav_action_name': Parameter.Type.STRING,
    'plan_action_name': Parameter.Type.STRING,
    'costmap_topic': Parameter.Type.STRING,
    'goal_frame': Parameter.Type.STRING,
    'robot_frame': Parameter.Type.STRING,
    'standoff_radii_m': Parameter.Type.DOUBLE_ARRAY,
    'num_samples': Parameter.Type.INTEGER,
    'max_goal_cost': Parameter.Type.INTEGER,
    'max_detour_factor': Parameter.Type.DOUBLE,
    'cost_weight': Parameter.Type.DOUBLE,
    'candidates_per_object': Parameter.Type.INTEGER,
    'min_relative_evidence': Parameter.Type.DOUBLE,
    'label_aliases': Parameter.Type.STRING_ARRAY,
    'server_timeout_s': Parameter.Type.DOUBLE,
    'feedback_period_s': Parameter.Type.DOUBLE,
}

STATUS = {GoalStatus.STATUS_SUCCEEDED: 'succeeded', GoalStatus.STATUS_ABORTED: 'aborted',
          GoalStatus.STATUS_CANCELED: 'canceled'}


def path_length(path):
    pts = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


class CommanderNode(Node):
    """GoTo service -> semantic map lookup -> goal generation -> Nav2."""

    def __init__(self):
        super().__init__('commander_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        self.p = p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.aliases = parse_aliases(p['label_aliases'])
        self.grid = None
        self.active_goal = None          # ClientGoalHandle of the running navigation
        self.request_seq = 0             # lets a superseded request know it was replaced

        cbg = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_subscription(Costmap, p['costmap_topic'], self.on_costmap, 1,
                                 callback_group=cbg)
        self.list_client = self.create_client(ListObjects, p['list_objects_service'],
                                              callback_group=cbg)
        self.nav_client = ActionClient(self, NavigateToPose, p['nav_action_name'],
                                       callback_group=cbg)
        self.plan_client = ActionClient(self, ComputePathToPose, p['plan_action_name'],
                                        callback_group=cbg)
        self.create_service(GoTo, p['goto_service'], self.on_goto, callback_group=cbg)
        self.create_service(Trigger, p['cancel_service'], self.on_cancel, callback_group=cbg)
        self.last_feedback_log = 0.0
        self.get_logger().info(f'ready: {p["goto_service"]} (aliases: {self.aliases})')

    # -------------------------------------------------------------- inputs
    def on_costmap(self, msg):
        m = msg.metadata
        self.grid = CostGrid(m.size_x, m.size_y, m.resolution, m.origin.position.x,
                             m.origin.position.y, tuple(msg.data))

    def robot_xy(self):
        tf = self.tf_buffer.lookup_transform(self.p['goal_frame'], self.p['robot_frame'], Time())
        return tf.transform.translation.x, tf.transform.translation.y

    def pose(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.p['goal_frame']
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        q = yaw_to_quaternion(yaw)
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = q
        return msg

    # -------------------------------------------------------------- helpers (awaitable)
    async def list_objects(self, label=''):
        return (await self.list_client.call_async(ListObjects.Request(label_filter=label))).objects

    async def plan_length(self, cand):
        """Length of the planner's path from the robot to `cand`, or None if no path."""
        goal = ComputePathToPose.Goal(goal=self.pose(cand.x, cand.y, cand.yaw), use_start=False)
        handle = await self.plan_client.send_goal_async(goal)
        if not handle.accepted:
            return None
        result = await handle.get_result_async()
        if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.path.poses:
            return None
        return path_length(result.result.path)

    def on_feedback(self, msg):
        now = time.monotonic()
        if now - self.last_feedback_log >= self.p['feedback_period_s']:
            self.last_feedback_log = now
            fb = msg.feedback
            self.get_logger().info(
                f'  distance remaining {fb.distance_remaining:.2f} m, '
                f'elapsed {fb.navigation_time.sec} s, recoveries {fb.number_of_recoveries}')

    async def cancel_active(self, reason):
        if self.active_goal is not None:
            self.get_logger().info(f'canceling the current goal ({reason})')
            await self.active_goal.cancel_goal_async()

    # -------------------------------------------------------------- services
    async def on_cancel(self, request, response):
        if self.active_goal is None:
            response.success, response.message = False, 'nothing to cancel'
            return response
        await self.cancel_active('cancel service')
        response.success, response.message = True, 'navigation canceled'
        return response

    async def on_goto(self, request, response):
        self.request_seq += 1
        seq = self.request_seq
        label = resolve_label(request.label, self.aliases)
        said = f"'{request.label}'" + (f' ({label})' if label != request.label.strip() else '')
        self.get_logger().info(f'GoTo {said}')

        def fail(message):
            self.get_logger().warn(f'GoTo {said} failed: {message}')
            response.success, response.message = False, message
            return response

        t = self.p['server_timeout_s']
        if not self.list_client.wait_for_service(timeout_sec=t):
            return fail('semantic map is not running (no ListObjects service)')
        if not (self.plan_client.wait_for_server(timeout_sec=t) and
                self.nav_client.wait_for_server(timeout_sec=t)):
            return fail('Nav2 is not running (no planner / navigate_to_pose action)')
        if self.grid is None:
            return fail(f'no global costmap received on {self.p["costmap_topic"]} yet')
        try:
            robot = self.robot_xy()
        except TransformException as e:
            return fail(f'robot pose unknown ({e})')

        objects = await self.list_objects(label)
        if not objects:
            known = Counter(o.label for o in await self.list_objects())
            listing = ', '.join(f'{n} x{c}' if c > 1 else n for n, c in sorted(known.items()))
            return fail(f'no {label} in the semantic map. Known objects: {listing or "none"}')

        credible = credible_instances(objects, self.p['min_relative_evidence'])
        for o in objects:
            if o not in credible:
                self.get_logger().info(f'  {o.label} #{o.id}: ignored, only '
                                       f'{o.observation_count} sightings')
        objects = credible
        # Nearest REACHABLE instance: shortest planned path over all instances.
        best = None
        for obj in objects:
            cands = candidate_goals(
                (obj.position.x, obj.position.y), robot, self.grid, self.p['standoff_radii_m'],
                self.p['num_samples'], self.p['max_goal_cost'], self.p['max_detour_factor'],
                self.p['cost_weight'])
            for cand in cands[:self.p['candidates_per_object']]:
                length = await self.plan_length(cand)
                if length is not None:
                    self.get_logger().info(
                        f'  {obj.label} #{obj.id} ({obj.observation_count} sightings): goal '
                        f'({cand.x:.2f}, {cand.y:.2f}) yaw {math.degrees(cand.yaw):.0f} deg, '
                        f'path {length:.2f} m')
                    if best is None or length < best[0]:
                        best = (length, obj, cand)
                    break
            else:
                self.get_logger().info(f'  {obj.label} #{obj.id}: no reachable goal '
                                       f'({len(cands)} candidates)')
        if best is None:
            return fail(f'{len(objects)} {label}(s) in the map, but no reachable goal pose '
                        'next to any of them')
        if seq != self.request_seq:
            return fail('superseded by a newer GoTo request')

        length, obj, cand = best
        await self.cancel_active('superseded by a new GoTo')
        handle = await self.nav_client.send_goal_async(
            NavigateToPose.Goal(pose=self.pose(cand.x, cand.y, cand.yaw)),
            feedback_callback=self.on_feedback)
        if not handle.accepted:
            return fail('Nav2 rejected the goal')
        self.active_goal = handle
        self.get_logger().info(f'navigating to {obj.label} #{obj.id} at '
                               f'({obj.position.x:.2f}, {obj.position.y:.2f}), '
                               f'path {length:.2f} m')
        t0 = self.get_clock().now()
        result = await handle.get_result_async()
        if self.active_goal is handle:
            self.active_goal = None
        took = (self.get_clock().now() - t0).nanoseconds * 1e-9
        status = STATUS.get(result.status, f'status {result.status}')
        where = f'{obj.label} #{obj.id} at ({obj.position.x:.2f}, {obj.position.y:.2f})'
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            err = result.result.error_msg or ''
            return fail(f'navigation to {where} {status} after {took:.0f} s {err}'.strip())
        standoff = math.dist((cand.x, cand.y), (obj.position.x, obj.position.y))
        response.success = True
        response.message = (f'arrived at {where}: standing {standoff:.2f} m from it, facing '
                            f'it; planned path {length:.1f} m, {took:.0f} s (sim time)')
        self.get_logger().info(response.message)
        return response


def main(args=None):
    # Ctrl-C under `ros2 launch` delivers SIGINT twice (terminal + launch), so the
    # interrupt can land anywhere in shutdown; catching it here keeps exits clean.
    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        executor.add_node(CommanderNode())
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
