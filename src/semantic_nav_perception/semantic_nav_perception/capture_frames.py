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
Save RGB frames from the simulator as a YOLO dataset (for evaluation / fine-tuning).

Role:
    Record what the robot's camera sees while it is driven around, together
    with the simulator's ground-truth 2D boxes when sim.launch.py was started
    with gt_boxes:=true. The output is directly usable by Ultralytics
    (`yolo train data=<output_dir>/dataset.yaml`) and by evaluate_detector.

    A frame is saved only if at least `min_period_s` of sim time has passed
    since the last save AND the robot has moved `min_translation_m` or turned
    `min_rotation_rad` (map -> base_link TF). This avoids hundreds of identical
    frames while the robot stands still. If the TF is unavailable the motion
    check is skipped.

Topics:
    Subscribes  <image_topic>      sensor_msgs/Image (rgb8/bgr8)
    Subscribes  <gt_boxes_topic>   vision_msgs/Detection2DArray (optional, '' = off)
                                   class_id = COCO index as a string

Parameters (config/params.yaml, section capture_frames):
    image_topic, gt_boxes_topic       (string)
    output_dir                        (string) created if missing; ~ is expanded
    class_names_path                  (string) YAML with class_names (model's .yaml)
    min_period_s, min_translation_m, min_rotation_rad   (double) save gating
    map_frame, base_frame             (string) frames for the pose in frames.csv
    max_pending                       (int)    unmatched image / gt messages kept
                                               while waiting for the partner
    use_sim_time                      (bool)
"""

import csv
import math
import os

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from semantic_nav_perception.dataset import cxcywh_to_xyxy, dataset_yaml, to_yolo_line
from semantic_nav_perception.paths import resolve_path
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
import yaml

PARAMETERS = {
    'image_topic': Parameter.Type.STRING,
    'gt_boxes_topic': Parameter.Type.STRING,
    'output_dir': Parameter.Type.STRING,
    'class_names_path': Parameter.Type.STRING,
    'min_period_s': Parameter.Type.DOUBLE,
    'min_translation_m': Parameter.Type.DOUBLE,
    'min_rotation_rad': Parameter.Type.DOUBLE,
    'map_frame': Parameter.Type.STRING,
    'base_frame': Parameter.Type.STRING,
    'max_pending': Parameter.Type.INTEGER,
}


def stamp_key(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


class CaptureFrames(Node):
    """Pairs images with ground-truth boxes by timestamp and writes a YOLO dataset."""

    def __init__(self):
        super().__init__('capture_frames')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.p = p
        self.root = os.path.expanduser(p['output_dir'])
        os.makedirs(os.path.join(self.root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'labels'), exist_ok=True)
        with open(resolve_path(p['class_names_path'])) as f:
            names = yaml.safe_load(f)['class_names']
        with open(os.path.join(self.root, 'dataset.yaml'), 'w') as f:
            f.write(dataset_yaml(os.path.abspath(self.root), names))
        index = os.path.join(self.root, 'frames.csv')
        new = not os.path.exists(index)
        self.csv_file = open(index, 'a', newline='')
        self.csv = csv.writer(self.csv_file)
        if new:
            self.csv.writerow(['frame', 'stamp', 'x', 'y', 'yaw', 'n_gt_boxes'])

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.use_gt = bool(p['gt_boxes_topic'])
        self.images, self.gts = {}, {}
        self.last_save = None      # (stamp_ns, x, y, yaw)
        self.saved = 0
        self.create_subscription(Image, p['image_topic'], self.on_image, qos_profile_sensor_data)
        if self.use_gt:
            self.create_subscription(Detection2DArray, p['gt_boxes_topic'], self.on_gt, 10)
        self.get_logger().info(f'saving frames to {self.root} '
                               f'(ground truth: {p["gt_boxes_topic"] or "off"})')

    def on_image(self, msg):
        if not self.use_gt:
            self.consider(msg, None)
            return
        self.images[stamp_key(msg)] = msg
        self.pair()

    def on_gt(self, msg):
        self.gts[stamp_key(msg)] = msg
        self.pair()

    def pair(self):
        for key in sorted(set(self.images) & set(self.gts)):
            self.consider(self.images.pop(key), self.gts.pop(key))
        for pending in (self.images, self.gts):
            for key in sorted(pending)[:-self.p['max_pending']]:
                del pending[key]

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.p['map_frame'], self.p['base_frame'],
                                                 Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def should_save(self, stamp_ns, pose):
        if self.last_save is None:
            return True
        if (stamp_ns - self.last_save[0]) * 1e-9 < self.p['min_period_s']:
            return False
        if pose is None or self.last_save[1] is None:
            return True
        moved = math.hypot(pose[0] - self.last_save[1], pose[1] - self.last_save[2])
        turned = abs(math.remainder(pose[2] - self.last_save[3], math.tau))
        return moved >= self.p['min_translation_m'] or turned >= self.p['min_rotation_rad']

    def consider(self, image_msg, gt_msg):
        stamp_ns = stamp_key(image_msg)
        pose = self.robot_pose()
        if not self.should_save(stamp_ns, pose):
            return
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        h, w = image.shape[:2]
        frame = f'{stamp_ns // 10**6:010d}'   # sim time in ms: unique and sortable
        cv2.imwrite(os.path.join(self.root, 'images', frame + '.png'), image)
        n_boxes = ''
        if gt_msg is not None:
            lines = []
            for det in gt_msg.detections:
                b = det.bbox
                box = cxcywh_to_xyxy(b.center.position.x, b.center.position.y,
                                     b.size_x, b.size_y)
                lines.append(to_yolo_line(int(det.results[0].hypothesis.class_id), box, w, h))
            with open(os.path.join(self.root, 'labels', frame + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            n_boxes = len(lines)
        x, y, yaw = pose if pose else ('', '', '')
        self.csv.writerow([frame, f'{stamp_ns * 1e-9:.3f}', x, y, yaw, n_boxes])
        self.csv_file.flush()
        self.last_save = (stamp_ns, *(pose or (None, None, None)))
        self.saved += 1
        if self.saved % 10 == 0:
            self.get_logger().info(f'{self.saved} frames saved')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    try:
        rclpy.init(args=args)
        node = CaptureFrames()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if 'node' in locals():
            node.get_logger().info(f'{node.saved} frames saved to {node.root}')
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
