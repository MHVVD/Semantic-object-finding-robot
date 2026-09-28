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
Run a YOLO nano ONNX model (OpenVINO, CPU) on the RGB camera stream.

Role:
    Turn raw RGB images into 2D bounding-box detections with class labels and
    confidences. This is the "AI" front end of the pipeline.

Topics (planned, implemented in the perception milestone):
    Subscribes  <image_topic>       sensor_msgs/Image
    Publishes   <detections_topic>  vision_msgs/Detection2DArray

Parameters (all loaded from semantic_nav_bringup/config/params.yaml):
    image_topic           (string) RGB image topic to subscribe to.
    detections_topic      (string) Output topic for 2D detections.
    model_path            (string) Path to the exported YOLO .onnx file.
    confidence_threshold  (double) Minimum score to keep a detection.
    use_sim_time          (bool)   Must be true in simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# Parameter names and types. No defaults on purpose: values must come from the
# YAML file so there is exactly one place to change them.
PARAMETERS = {
    'image_topic': Parameter.Type.STRING,
    'detections_topic': Parameter.Type.STRING,
    'model_path': Parameter.Type.STRING,
    'confidence_threshold': Parameter.Type.DOUBLE,
}


class DetectorNode(Node):
    """Skeleton node; see the module docstring for the planned interface."""

    def __init__(self):
        super().__init__('detector_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        values = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.get_logger().info(f'detector_node started with parameters: {values}')


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
