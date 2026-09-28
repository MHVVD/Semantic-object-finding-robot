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
    Look the label up in the semantic map (ListObjects), pick the best
    instance, generate a reachable goal pose standoff_distance_m in front of
    the object facing it, and send it to Nav2.

Topics / services / actions (planned, implemented in the commander milestone):
    Service      go_to               semantic_nav_interfaces/GoTo
    Client       list_objects        semantic_nav_interfaces/ListObjects
    Action cli   navigate_to_pose    nav2_msgs/action/NavigateToPose

Parameters (all loaded from semantic_nav_bringup/config/params.yaml):
    list_objects_service  (string) Semantic map query service name.
    nav_action_name       (string) Nav2 action server name.
    standoff_distance_m   (double) Distance to stop short of the object.
    goal_frame            (string) Frame for navigation goals (map).
    use_sim_time          (bool)   Must be true in simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# Parameter names and types. No defaults on purpose: values must come from the
# YAML file so there is exactly one place to change them.
PARAMETERS = {
    'list_objects_service': Parameter.Type.STRING,
    'nav_action_name': Parameter.Type.STRING,
    'standoff_distance_m': Parameter.Type.DOUBLE,
    'goal_frame': Parameter.Type.STRING,
}


class CommanderNode(Node):
    """Skeleton node; see the module docstring for the planned interface."""

    def __init__(self):
        super().__init__('commander_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        values = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.get_logger().info(f'commander_node started with parameters: {values}')


def main(args=None):
    rclpy.init(args=args)
    node = CommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
