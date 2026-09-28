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
Fuse per-frame observations into a persistent map of object instances.

Role:
    Data association: each new observation is matched to an existing object of
    the same label within association_radius_m (and merged, updating position,
    confidence and observation_count) or starts a new object. Serves queries.

Topics / services (planned, implemented in the mapping milestone):
    Subscribes  <observations_topic>  semantic_nav_interfaces/SemanticObjectArray
    Publishes   <map_topic>           semantic_nav_interfaces/SemanticObjectArray
    Publishes   <markers_topic>       visualization_msgs/MarkerArray (RViz)
    Service     list_objects          semantic_nav_interfaces/ListObjects

Parameters (all loaded from semantic_nav_bringup/config/params.yaml):
    observations_topic     (string) Input map-frame observations.
    map_topic              (string) Output fused semantic map.
    markers_topic          (string) RViz markers.
    association_radius_m   (double) Max distance to merge with an existing object.
    min_observations       (int)    Observations before an object is published.
    use_sim_time           (bool)   Must be true in simulation.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter

# Parameter names and types. No defaults on purpose: values must come from the
# YAML file so there is exactly one place to change them.
PARAMETERS = {
    'observations_topic': Parameter.Type.STRING,
    'map_topic': Parameter.Type.STRING,
    'markers_topic': Parameter.Type.STRING,
    'association_radius_m': Parameter.Type.DOUBLE,
    'min_observations': Parameter.Type.INTEGER,
}


class SemanticMapNode(Node):
    """Skeleton node; see the module docstring for the planned interface."""

    def __init__(self):
        super().__init__('semantic_map_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        values = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.get_logger().info(f'semantic_map_node started with parameters: {values}')


def main(args=None):
    # Ctrl-C under `ros2 launch` delivers SIGINT twice (terminal + launch), so the
    # interrupt can land anywhere in shutdown; catching it here keeps exits clean.
    try:
        rclpy.init(args=args)
        rclpy.spin(SemanticMapNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
