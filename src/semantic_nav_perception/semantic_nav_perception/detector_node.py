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
    confidences. This is the "AI" front end of the pipeline; projector_node
    later lifts each box to 3D using the depth image with the SAME timestamp,
    which is why every output carries the input image's header unchanged.

    The subscription keeps only the newest image (sensor-data QoS, depth 1),
    so if inference is slower than the camera, stale frames are dropped
    instead of queueing up and adding latency. `max_rate_hz` additionally
    skips frames (by image timestamp, i.e. sim time in simulation) to leave
    CPU for Gazebo and Nav2: the semantic map does not need 10 Hz.

Topics:
    Subscribes  <image_topic>        sensor_msgs/Image (bgr8 / rgb8)
    Publishes   <detections_topic>   vision_msgs/Detection2DArray
                                     header = image header; per detection:
                                     bbox centre/size in pixels,
                                     results[0].hypothesis.class_id = class name,
                                     results[0].hypothesis.score = confidence
    Publishes   <debug_image_topic>  sensor_msgs/Image, boxes drawn on the frame
                                     (only rendered while someone subscribes)

Parameters (config/params.yaml, section detector_node):
    image_topic, detections_topic, debug_image_topic   (string)
    model_path            (string) YOLO .onnx; relative = inside this package's share
    class_names_path      (string) YAML with `class_names` in model index order
    input_width, input_height  (int) network input size, multiples of 32
    confidence_threshold  (double) minimum class score to keep a box
    iou_threshold         (double) NMS: same-class boxes overlapping more are merged
    class_whitelist       (string[]) class names to report; ['*'] = all classes
    device                (string) OpenVINO device, e.g. CPU
    num_threads           (int) OpenVINO CPU threads, 0 = OpenVINO decides
    max_rate_hz           (double) process at most this many images per second of
                          image time; 0 = every image
    stats_period_s        (double) how often latency / FPS are logged
    use_sim_time          (bool) must be true in simulation
"""

import time

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from semantic_nav_perception.paths import resolve_path
from semantic_nav_perception.yolo import draw, whitelist_ids, YoloDetector
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import yaml

# Parameter names and types. No defaults on purpose: values must come from the
# YAML file so there is exactly one place to change them.
PARAMETERS = {
    'image_topic': Parameter.Type.STRING,
    'detections_topic': Parameter.Type.STRING,
    'debug_image_topic': Parameter.Type.STRING,
    'model_path': Parameter.Type.STRING,
    'class_names_path': Parameter.Type.STRING,
    'input_width': Parameter.Type.INTEGER,
    'input_height': Parameter.Type.INTEGER,
    'confidence_threshold': Parameter.Type.DOUBLE,
    'iou_threshold': Parameter.Type.DOUBLE,
    'class_whitelist': Parameter.Type.STRING_ARRAY,
    'device': Parameter.Type.STRING,
    'num_threads': Parameter.Type.INTEGER,
    'max_rate_hz': Parameter.Type.DOUBLE,
    'stats_period_s': Parameter.Type.DOUBLE,
}


def to_detection_msg(det, header):
    """Our Detection (xyxy pixels) -> vision_msgs/Detection2D (centre + size)."""
    msg = Detection2D()
    msg.header = header
    msg.bbox.center.position.x = (det.x1 + det.x2) / 2.0
    msg.bbox.center.position.y = (det.y1 + det.y2) / 2.0
    msg.bbox.size_x = det.x2 - det.x1
    msg.bbox.size_y = det.y2 - det.y1
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = det.label
    hyp.hypothesis.score = det.score
    msg.results.append(hyp)
    return msg


class DetectorNode(Node):
    """Subscribes to images, runs YOLO, publishes Detection2DArray + debug image."""

    def __init__(self):
        super().__init__('detector_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.p = p

        with open(resolve_path(p['class_names_path'])) as f:
            class_names = yaml.safe_load(f)['class_names']
        whitelist = [] if p['class_whitelist'] == ['*'] else p['class_whitelist']
        self.allowed_ids = whitelist_ids(class_names, whitelist)
        model_path = resolve_path(p['model_path'])
        self.detector = YoloDetector(model_path, class_names,
                                     (p['input_width'], p['input_height']),
                                     p['device'], p['num_threads'])

        self.bridge = CvBridge()
        self.det_pub = self.create_publisher(Detection2DArray, p['detections_topic'], 10)
        self.debug_pub = self.create_publisher(Image, p['debug_image_topic'],
                                               qos_profile_sensor_data)
        self.create_subscription(Image, p['image_topic'], self.on_image,
                                 qos_profile_sensor_data)
        # 10 % slack: sensor stamps jitter by a few ms (e.g. x.802 s after x.601 s),
        # and without slack a 5 Hz limit on a 10 Hz camera would often skip 2 frames.
        self.min_period_ns = int(0.9e9 / p['max_rate_hz']) if p['max_rate_hz'] > 0 else 0
        self.last_processed_ns = None    # image stamp of the last processed frame
        self.stats = []                  # per-frame timing dicts since the last report
        self.skipped = 0
        self.last_report = time.monotonic()
        self.create_timer(p['stats_period_s'], self.report)
        self.get_logger().info(
            f'{model_path} on {p["device"]} at {p["input_width"]}x{p["input_height"]}, '
            f'conf {p["confidence_threshold"]}, iou {p["iou_threshold"]}, '
            f'classes {whitelist or "all"}')

    def on_image(self, msg):
        stamp_ns = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
        if (self.last_processed_ns is not None and
                0 <= stamp_ns - self.last_processed_ns < self.min_period_ns):
            self.skipped += 1
            return
        self.last_processed_ns = stamp_ns
        t0 = time.perf_counter()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        t1 = time.perf_counter()
        dets = self.detector(image, self.p['confidence_threshold'], self.p['iou_threshold'],
                             self.allowed_ids)
        out = Detection2DArray(header=msg.header)
        out.detections = [to_detection_msg(d, msg.header) for d in dets]
        self.det_pub.publish(out)
        t2 = time.perf_counter()
        if self.debug_pub.get_subscription_count() > 0:
            debug = self.bridge.cv2_to_imgmsg(draw(image, dets), encoding='bgr8')
            debug.header = msg.header
            self.debug_pub.publish(debug)
        timing = dict(self.detector.timing)
        timing['convert'] = (t1 - t0) * 1e3
        timing['callback'] = (t2 - t0) * 1e3
        # Age of the image when its detections went out, in the node's clock
        # (sim time in simulation): transport + queueing + processing.
        age = self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)
        timing['age'] = age.nanoseconds * 1e-6
        self.stats.append(timing)

    def report(self):
        now = time.monotonic()
        elapsed, self.last_report = now - self.last_report, now
        if not self.stats:
            self.get_logger().info('no images received', throttle_duration_sec=30.0)
            return
        med = {k: float(np.median([s[k] for s in self.stats])) for k in self.stats[0]}
        self.get_logger().info(
            f'{len(self.stats) / elapsed:.1f} FPS (wall) | median ms: '
            f'infer {med["infer"]:.1f}, pre {med["pre"]:.1f}, post {med["post"]:.1f}, '
            f'callback {med["callback"]:.1f}, image age {med["age"]:.0f} | '
            f'skipped {self.skipped}')
        self.stats, self.skipped = [], 0


def main(args=None):
    # Ctrl-C under `ros2 launch` delivers SIGINT twice (terminal + launch), so the
    # interrupt can land anywhere in shutdown; catching it here keeps exits clean.
    try:
        rclpy.init(args=args)
        rclpy.spin(DetectorNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
