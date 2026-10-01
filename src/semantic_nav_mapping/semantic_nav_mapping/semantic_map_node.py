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
    projector_node publishes what the camera saw in ONE frame. This node keeps
    the long-term memory: each observation is associated with an existing object
    of the same label (nearest neighbour within the association radius, one-to-one
    per frame) and updates its running-mean position, or starts a new tentative
    object. An object is only published ("confirmed") after min_observations
    sightings; tentative objects not seen again within tentative_timeout_s are
    forgotten. The association logic lives in association.py (unit tested).

Topics / services:
    Subscribes  <observations_topic>    semantic_nav_interfaces/SemanticObjectArray
    Publishes   <map_topic>             semantic_nav_interfaces/SemanticObjectArray
                                        (confirmed objects; transient-local, 1 Hz)
    Publishes   <markers_topic>         visualization_msgs/MarkerArray (sphere + label)
    Service     <list_objects_service>  semantic_nav_interfaces/ListObjects
    Service     <save_service>          semantic_nav_interfaces/MapFile (to YAML)
    Service     <load_service>          semantic_nav_interfaces/MapFile (from YAML)

Parameters (config/params.yaml, section semantic_map_node):
    observations_topic, map_topic, markers_topic        (string)
    list_objects_service, save_service, load_service    (string)
    map_frame              (string) frame the observations and the map are in
    association_radius_m   (double) default gate for same-label association (x-y)
    class_radius           (string[]) per-class gates, "label=metres" (big objects)
    association_method     (string) greedy | hungarian
    min_observations       (int)    sightings before an object is confirmed
    tentative_timeout_s    (double) forget unconfirmed objects not seen this long
    publish_period_s       (double) map / marker publishing period
    map_file               (string) default YAML path for save/load (~ expanded)
    load_on_start          (bool)   load map_file at start-up if it exists
    use_sim_time           (bool)
"""

import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from semantic_nav_interfaces.msg import SemanticObject, SemanticObjectArray
from semantic_nav_interfaces.srv import ListObjects, MapFile
from semantic_nav_mapping.association import Observation, parse_class_radius, SemanticMap
from visualization_msgs.msg import Marker, MarkerArray
import yaml

PARAMETERS = {
    'observations_topic': Parameter.Type.STRING,
    'map_topic': Parameter.Type.STRING,
    'markers_topic': Parameter.Type.STRING,
    'list_objects_service': Parameter.Type.STRING,
    'save_service': Parameter.Type.STRING,
    'load_service': Parameter.Type.STRING,
    'map_frame': Parameter.Type.STRING,
    'association_radius_m': Parameter.Type.DOUBLE,
    'class_radius': Parameter.Type.STRING_ARRAY,
    'association_method': Parameter.Type.STRING,
    'min_observations': Parameter.Type.INTEGER,
    'tentative_timeout_s': Parameter.Type.DOUBLE,
    'publish_period_s': Parameter.Type.DOUBLE,
    'map_file': Parameter.Type.STRING,
    'load_on_start': Parameter.Type.BOOL,
}


class SemanticMapNode(Node):
    """Maintains the semantic map and serves it."""

    def __init__(self):
        super().__init__('semantic_map_node')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        self.p = p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.map = SemanticMap(p['association_radius_m'], p['min_observations'],
                               p['tentative_timeout_s'], p['association_method'],
                               parse_class_radius(p['class_radius']))
        self.observations_received = 0

        # Transient local: a node that starts later (RViz, the commander) still gets
        # the latest map without waiting for the next publish.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(SemanticObjectArray, p['map_topic'], latched)
        self.marker_pub = self.create_publisher(MarkerArray, p['markers_topic'], latched)
        self.create_subscription(SemanticObjectArray, p['observations_topic'],
                                 self.on_observations, 10)
        self.create_service(ListObjects, p['list_objects_service'], self.on_list)
        self.create_service(MapFile, p['save_service'], self.on_save)
        self.create_service(MapFile, p['load_service'], self.on_load)
        self.create_timer(p['publish_period_s'], self.publish)

        if p['load_on_start'] and os.path.exists(self.path(p['map_file'])):
            n = self.load(self.path(p['map_file']))
            self.get_logger().info(f'loaded {n} objects from {self.path(p["map_file"])}')
        self.get_logger().info(
            f'semantic map: radius {p["association_radius_m"]} m '
            f'{self.map.class_radius or ""}, {p["association_method"]}, confirm after '
            f'{p["min_observations"]} observations, forget tentative after '
            f'{p["tentative_timeout_s"]} s')

    # -------------------------------------------------------------- input
    def on_observations(self, msg):
        if msg.header.frame_id != self.p['map_frame']:
            self.get_logger().warn(f'ignoring observations in frame {msg.header.frame_id!r} '
                                   f'(expected {self.p["map_frame"]!r})',
                                   throttle_duration_sec=10.0)
            return
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        obs = [Observation(o.label, (o.position.x, o.position.y, o.position.z), o.confidence)
               for o in msg.objects]
        self.observations_received += len(obs)
        # Called for empty frames too: the clock still advances, so stale tentative
        # objects are pruned even while nothing is detected.
        self.map.update(obs, stamp)

    # -------------------------------------------------------------- output
    def to_msgs(self, tracks):
        out = []
        for t in tracks:
            o = SemanticObject(label=t.label, id=t.track_id, confidence=float(t.confidence),
                               observation_count=t.count)
            o.header.frame_id = self.p['map_frame']
            o.header.stamp = Time(nanoseconds=int(t.last_seen * 1e9)).to_msg()
            o.position.x, o.position.y, o.position.z = (float(v) for v in t.position)
            out.append(o)
        return out

    def publish(self):
        tracks = self.map.confirmed()
        arr = SemanticObjectArray(objects=self.to_msgs(tracks))
        arr.header.frame_id = self.p['map_frame']
        arr.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(arr)
        self.get_logger().info(
            f'{len(tracks)} confirmed / {len(self.map.tracks) - len(tracks)} tentative objects '
            f'from {self.observations_received} observations', throttle_duration_sec=30.0)
        markers = MarkerArray()
        # DELETEALL first, so objects removed by a load disappear from RViz too.
        markers.markers.append(Marker(action=Marker.DELETEALL))
        for obj in arr.objects:
            sphere = Marker(header=arr.header, ns='objects', id=obj.id, type=Marker.SPHERE,
                            action=Marker.ADD)
            sphere.pose.position = obj.position
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.1, 0.8, 0.2, 0.8
            text = Marker(header=arr.header, ns='labels', id=obj.id,
                          type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                          text=f'{obj.label} #{obj.id} ({obj.observation_count})')
            text.pose.position.x, text.pose.position.y = obj.position.x, obj.position.y
            text.pose.position.z = obj.position.z + 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            markers.markers += [sphere, text]
        self.marker_pub.publish(markers)

    # -------------------------------------------------------------- services
    def on_list(self, request, response):
        response.objects = self.to_msgs(self.map.confirmed(request.label_filter))
        return response

    @staticmethod
    def path(p):
        return os.path.expandvars(os.path.expanduser(p))

    def on_save(self, request, response):
        path = self.path(request.path or self.p['map_file'])
        data = {'frame_id': self.p['map_frame'],
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': {k: self.p[k] for k in (
                    'association_radius_m', 'class_radius', 'association_method',
                    'min_observations', 'tentative_timeout_s')},
                **self.map.to_dict()}
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except OSError as e:
            response.success, response.message = False, f'cannot write {path}: {e}'
            return response
        response.success, response.object_count = True, len(data['objects'])
        response.message = f'saved {len(data["objects"])} objects to {path}'
        self.get_logger().info(response.message)
        return response

    def load(self, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if data.get('frame_id', self.p['map_frame']) != self.p['map_frame']:
            raise ValueError(f'map is in frame {data["frame_id"]!r}, '
                             f'node uses {self.p["map_frame"]!r}')
        return self.map.load_dict(data)

    def on_load(self, request, response):
        path = self.path(request.path or self.p['map_file'])
        try:
            n = self.load(path)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
            response.success, response.message = False, f'cannot load {path}: {e}'
            return response
        self.publish()
        response.success, response.object_count = True, n
        response.message = f'loaded {n} objects from {path}'
        self.get_logger().info(response.message)
        return response


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
