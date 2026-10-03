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
Turn the depth camera into a height-filtered LaserScan for the Nav2 costmaps.

Role:
    The 2D lidar sits ~0.2 m above the floor, so a dining table is four legs to
    it: the space underneath looks free, the planner sends the robot in, and the
    inflated legs and chairs then close every way out (PROBLEMS_LOG #76, #93).
    This node lets the costmap see what the camera sees between min_height_m
    and max_height_m (table tops, chair seats, counters): it decimates the depth
    image, deprojects it with the camera intrinsics, transforms the points into
    target_frame (base_link, on the floor, z up), keeps those in the height band
    and publishes the nearest one per bearing as a LaserScan (+inf where no
    point fell). The maths is in depth_scan.py.

    In nav2_params.yaml the scan feeds its OWN obstacle layer (depth_layer) in
    the LOCAL costmap: the lidar's beams pass under the table, and in a shared
    layer their clearing would erase the table top again. +inf beams (nothing
    in the band) do clear (inf_is_valid true): otherwise marks in front of a
    stopped robot never go away (#96). The table top is visible from ~1.2 m,
    so the robot stops before it leaves the view. Not in the global costmap:
    marks there persisted and closed passages in a long SLAM run (#95).

    The camera is fixed on the robot, so target_frame <- camera is looked up once
    (latest) and cached; the scan carries the depth image's stamp, and Nav2 places
    it in the map with TF at that stamp.

Topics:
    Subscribes  <depth_topic>        sensor_msgs/Image (32FC1 metres or 16UC1 mm)
    Subscribes  <camera_info_topic>  sensor_msgs/CameraInfo (intrinsics, cached)
    Publishes   <scan_topic>         sensor_msgs/LaserScan in <target_frame>
    TF          <target_frame> <- camera optical frame (static, looked up once)

Parameters (config/params.yaml, section depth_scan_node):
    depth_topic, camera_info_topic, scan_topic, target_frame (string)
    decimation          (int)    use every n-th pixel in each direction
    min_height_m, max_height_m (double) height band above the floor that counts
    range_min_m, range_max_m   (double) horizontal range limits of the scan
    angle_increment_rad (double) bearing bin width
    max_rate_hz         (double) processing cap (image stamps; 10 % slack)
    tf_timeout_s        (double) wait for the camera transform at start-up
    use_sim_time        (bool)
"""

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from semantic_nav_perception.depth_scan import (beam_layout, depth_points, scan_ranges,
                                                transform_points)
from semantic_nav_perception.projection import depth_to_meters, quaternion_to_matrix
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {
    'depth_topic': Parameter.Type.STRING,
    'camera_info_topic': Parameter.Type.STRING,
    'scan_topic': Parameter.Type.STRING,
    'target_frame': Parameter.Type.STRING,
    'decimation': Parameter.Type.INTEGER,
    'min_height_m': Parameter.Type.DOUBLE,
    'max_height_m': Parameter.Type.DOUBLE,
    'range_min_m': Parameter.Type.DOUBLE,
    'range_max_m': Parameter.Type.DOUBLE,
    'angle_increment_rad': Parameter.Type.DOUBLE,
    'max_rate_hz': Parameter.Type.DOUBLE,
    'tf_timeout_s': Parameter.Type.DOUBLE,
}


class DepthScanNode(Node):
    """Depth image -> height-band LaserScan."""

    def __init__(self):
        super().__init__('depth_scan_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        self.p = p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.info = None
        self.camera_pose = None          # (R, t): target_frame <- optical
        self.last_stamp = None
        self.min_period = 0.9 / p['max_rate_hz']
        self.pub = self.create_publisher(LaserScan, p['scan_topic'], 10)
        self.create_subscription(CameraInfo, p['camera_info_topic'],
                                 lambda m: setattr(self, 'info', m), qos_profile_sensor_data)
        self.create_subscription(Image, p['depth_topic'], self.on_depth, qos_profile_sensor_data)
        self.get_logger().info(
            f'{p["depth_topic"]} -> {p["scan_topic"]} in {p["target_frame"]}: heights '
            f'{p["min_height_m"]}-{p["max_height_m"]} m, every {p["decimation"]}th pixel')

    def camera_transform(self, frame):
        if self.camera_pose is None:
            tf = self.tf_buffer.lookup_transform(
                self.p['target_frame'], frame, Time(),
                timeout=Duration(seconds=self.p['tf_timeout_s']))
            t, q = tf.transform.translation, tf.transform.rotation
            self.camera_pose = (quaternion_to_matrix(q.x, q.y, q.z, q.w), (t.x, t.y, t.z))
        return self.camera_pose

    def on_depth(self, msg):
        if self.info is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and 0.0 <= stamp - self.last_stamp < self.min_period:
            return
        self.last_stamp = stamp
        try:
            rotation, translation = self.camera_transform(msg.header.frame_id)
        except TransformException as e:
            self.get_logger().warn(f'no transform from the camera yet: {e}',
                                   throttle_duration_sec=10.0)
            return
        p = self.p
        k = self.info.k
        depth = depth_to_meters(self.bridge.imgmsg_to_cv2(msg), msg.encoding)
        points = transform_points(depth_points(depth, k[0], k[4], k[2], k[5], p['decimation']),
                                  rotation, translation)
        angle_min, n_beams = beam_layout(k[0], msg.width, p['angle_increment_rad'])
        ranges = scan_ranges(points, p['min_height_m'], p['max_height_m'], angle_min,
                             p['angle_increment_rad'], n_beams, p['range_min_m'],
                             p['range_max_m'])
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = p['target_frame']
        scan.angle_min = float(angle_min)
        scan.angle_increment = float(p['angle_increment_rad'])
        scan.angle_max = float(angle_min + (n_beams - 1) * p['angle_increment_rad'])
        scan.range_min = float(p['range_min_m'])
        scan.range_max = float(p['range_max_m'])
        scan.ranges = ranges.astype(np.float32).tolist()
        self.pub.publish(scan)


def main(args=None):
    # Ctrl-C under `ros2 launch` delivers SIGINT twice (terminal + launch), so the
    # interrupt can land anywhere in shutdown; catching it here keeps exits clean.
    try:
        rclpy.init(args=args)
        rclpy.spin(DepthScanNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
