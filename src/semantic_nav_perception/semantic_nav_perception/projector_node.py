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
Deproject 2D detections into 3D points in the map frame.

Role:
    For each bounding box, read the aligned depth image, back-project the box
    centre through the pinhole camera model (intrinsics from CameraInfo) and
    transform the resulting camera-frame point into the map frame with TF2.

Topics (planned, implemented in the perception milestone):
    Subscribes  <detections_topic>   vision_msgs/Detection2DArray
    Subscribes  <depth_topic>        sensor_msgs/Image
    Subscribes  <camera_info_topic>  sensor_msgs/CameraInfo
    Publishes   <output_topic>       semantic_nav_interfaces/SemanticObjectArray
                                     (per-frame observations, not yet fused)

Parameters (all loaded from semantic_nav_bringup/config/params.yaml):
    detections_topic   (string) Input 2D detections.
    depth_topic        (string) Depth image aligned with the RGB image.
    camera_info_topic  (string) Camera intrinsics.
    output_topic       (string) Map-frame observations.
    target_frame       (string) Fixed frame to express points in (map).
    max_depth_m        (double) Ignore depth readings beyond this range.
    use_sim_time       (bool)   Must be true in simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# Parameter names and types. No defaults on purpose: values must come from the
# YAML file so there is exactly one place to change them.
PARAMETERS = {
    'detections_topic': Parameter.Type.STRING,
    'depth_topic': Parameter.Type.STRING,
    'camera_info_topic': Parameter.Type.STRING,
    'output_topic': Parameter.Type.STRING,
    'target_frame': Parameter.Type.STRING,
    'max_depth_m': Parameter.Type.DOUBLE,
}


class ProjectorNode(Node):
    """Skeleton node; see the module docstring for the planned interface."""

    def __init__(self):
        super().__init__('projector_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        values = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.get_logger().info(f'projector_node started with parameters: {values}')


def main(args=None):
    rclpy.init(args=args)
    node = ProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
