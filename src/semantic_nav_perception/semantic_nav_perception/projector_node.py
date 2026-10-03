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
    For each bounding box, read the aligned depth image, take the median of
    the valid depths in the central part of the box, back-project the box
    centre through the pinhole camera model (intrinsics from CameraInfo) and
    transform the resulting optical-frame point into the map frame with tf2
    AT THE IMAGE TIMESTAMP. Output = single-frame observations; merging them
    into persistent objects is semantic_map_node's job.

    Detections, depth and camera_info are paired by message_filters'
    ApproximateTimeSynchronizer. In simulation the three share the exact same
    stamp (the detector copies the RGB header; the rgbd_camera renders colour
    and depth in the same tick), so `sync_slop_s` only needs to absorb jitter
    on real hardware.

Topics:
    Subscribes  <detections_topic>   vision_msgs/Detection2DArray
    Subscribes  <depth_topic>        sensor_msgs/Image (32FC1 metres or 16UC1 mm)
    Subscribes  <camera_info_topic>  sensor_msgs/CameraInfo
    Publishes   <output_topic>       semantic_nav_interfaces/SemanticObjectArray
                                     (id 0 = not yet associated, observation_count 1,
                                     confidence = detector score, header = map + image stamp)
    Publishes   <markers_topic>      visualization_msgs/MarkerArray (sphere + label)
    TF          <target_frame> <- camera optical frame, at the image stamp

Parameters (config/params.yaml, section projector_node):
    detections_topic, depth_topic, camera_info_topic, output_topic, markers_topic (string)
    target_frame        (string) fixed frame for the output (map)
    bbox_shrink         (double) fraction of the box (per side length) sampled for depth
    min_depth_m, max_depth_m  (double) valid depth range
    min_valid_fraction  (double) minimum share of valid pixels in the sampled patch
    sync_queue_size     (int)    messages buffered per input by the synchronizer
    sync_slop_s         (double) max stamp difference to pair messages
    tf_timeout_s        (double) how long to wait for the transform at the image stamp
    marker_lifetime_s   (double) RViz marker lifetime
    stats_period_s      (double) how often the frame / projection counters are logged
    use_sim_time        (bool)
"""

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import message_filters
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from semantic_nav_interfaces.msg import SemanticObject, SemanticObjectArray
from semantic_nav_perception.projection import (deproject, depth_to_meters, Intrinsics,
                                                robust_depth, shrink_box, transform_point)
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

PARAMETERS = {
    'detections_topic': Parameter.Type.STRING,
    'depth_topic': Parameter.Type.STRING,
    'camera_info_topic': Parameter.Type.STRING,
    'output_topic': Parameter.Type.STRING,
    'markers_topic': Parameter.Type.STRING,
    'target_frame': Parameter.Type.STRING,
    'bbox_shrink': Parameter.Type.DOUBLE,
    'min_depth_m': Parameter.Type.DOUBLE,
    'max_depth_m': Parameter.Type.DOUBLE,
    'min_valid_fraction': Parameter.Type.DOUBLE,
    'sync_queue_size': Parameter.Type.INTEGER,
    'sync_slop_s': Parameter.Type.DOUBLE,
    'tf_timeout_s': Parameter.Type.DOUBLE,
    'marker_lifetime_s': Parameter.Type.DOUBLE,
    'stats_period_s': Parameter.Type.DOUBLE,
}


class ProjectorNode(Node):
    """Synchronises detections + depth + intrinsics and publishes map-frame observations."""

    def __init__(self):
        super().__init__('projector_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        self.p = p = {name: self.get_parameter(name).value for name in PARAMETERS}

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        # spin_thread=True: TF arrives on its own thread, so lookup_transform(timeout)
        # inside our callback can actually wait for data instead of deadlocking.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.obs_pub = self.create_publisher(SemanticObjectArray, p['output_topic'], 10)
        self.marker_pub = self.create_publisher(MarkerArray, p['markers_topic'], 10)

        subs = [
            message_filters.Subscriber(self, Detection2DArray, p['detections_topic'],
                                       qos_profile=10),
            message_filters.Subscriber(self, Image, p['depth_topic'],
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, p['camera_info_topic'],
                                       qos_profile=qos_profile_sensor_data),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, p['sync_queue_size'], p['sync_slop_s'])
        self.sync.registerCallback(self.on_synced)

        self.counts = {'frames': 0, 'detections': 0, 'projected': 0, 'no_depth': 0,
                       'tf_fail': 0}
        self.create_timer(p['stats_period_s'], self.report)
        self.get_logger().info(
            f'projecting {p["detections_topic"]} with {p["depth_topic"]} into '
            f'{p["target_frame"]} (shrink {p["bbox_shrink"]}, depth '
            f'{p["min_depth_m"]}-{p["max_depth_m"]} m)')

    def on_synced(self, dets, depth_msg, info):
        self.counts['frames'] += 1
        self.counts['detections'] += len(dets.detections)
        stamp = dets.header.stamp
        out = SemanticObjectArray()
        out.header.frame_id = self.p['target_frame']
        out.header.stamp = stamp

        if dets.detections:
            try:
                # The transform from the camera to the map AT THE MOMENT THE IMAGE WAS
                # TAKEN; "latest" would be off by however far the robot moved since.
                tf = self.tf_buffer.lookup_transform(
                    self.p['target_frame'], depth_msg.header.frame_id, Time.from_msg(stamp),
                    timeout=Duration(seconds=self.p['tf_timeout_s']))
            except TransformException as e:
                self.counts['tf_fail'] += 1
                self.get_logger().warn(f'no transform {depth_msg.header.frame_id} -> '
                                       f'{self.p["target_frame"]} at image time: {e}',
                                       throttle_duration_sec=5.0)
                return
            t, q = tf.transform.translation, tf.transform.rotation
            translation, rotation = (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)

            k = Intrinsics.from_k(info.k)
            raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth = depth_to_meters(raw, depth_msg.encoding)
            h, w = depth.shape[:2]
            for det in dets.detections:
                b = det.bbox
                u, v = b.center.position.x, b.center.position.y
                bounds = shrink_box(u, v, b.size_x, b.size_y, self.p['bbox_shrink'], w, h)
                sample = robust_depth(depth, bounds, self.p['min_depth_m'],
                                      self.p['max_depth_m'], self.p['min_valid_fraction'])
                if sample is None:
                    self.counts['no_depth'] += 1
                    continue
                p_map = transform_point(deproject(u, v, sample[0], k), translation, rotation)
                obj = SemanticObject()
                obj.header = out.header
                obj.label = det.results[0].hypothesis.class_id
                obj.id = 0                      # assigned by the semantic map
                obj.position = Point(x=p_map[0], y=p_map[1], z=p_map[2])
                obj.confidence = float(det.results[0].hypothesis.score)
                obj.observation_count = 1
                out.objects.append(obj)
            self.counts['projected'] += len(out.objects)

        self.obs_pub.publish(out)
        self.marker_pub.publish(self.markers(out))

    def markers(self, arr):
        """Sphere + text label per observation; they expire after marker_lifetime_s."""
        markers = MarkerArray()
        lifetime = Duration(seconds=self.p['marker_lifetime_s']).to_msg()
        for i, obj in enumerate(arr.objects):
            sphere = Marker(header=arr.header, ns='observations', id=i, type=Marker.SPHERE,
                            action=Marker.ADD, lifetime=lifetime)
            sphere.pose.position = obj.position
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.15
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.5, 0.0, 0.9
            text = Marker(header=arr.header, ns='observation_labels', id=i,
                          type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, lifetime=lifetime,
                          text=f'{obj.label} {obj.confidence:.2f}')
            text.pose.position.x, text.pose.position.y = obj.position.x, obj.position.y
            text.pose.position.z = obj.position.z + 0.25
            text.pose.orientation.w = 1.0
            text.scale.z = 0.15
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            markers.markers += [sphere, text]
        return markers

    def report(self):
        c = self.counts
        if c['frames']:
            self.get_logger().info(
                f'{c["frames"]} synced frames, {c["detections"]} detections -> '
                f'{c["projected"]} projected, {c["no_depth"]} without valid depth, '
                f'{c["tf_fail"]} frames dropped (no TF)')
        else:
            self.get_logger().info('waiting for synchronized detections/depth/camera_info',
                                   throttle_duration_sec=30.0)


def main(args=None):
    # Ctrl-C under `ros2 launch` delivers SIGINT twice (terminal + launch), so the
    # interrupt can land anywhere in shutdown; catching it here keeps exits clean.
    try:
        rclpy.init(args=args)
        rclpy.spin(ProjectorNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
