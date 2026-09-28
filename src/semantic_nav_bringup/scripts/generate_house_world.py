#!/usr/bin/env python3
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
Generate worlds/house.sdf: a 10 m x 8 m, four-room house furnished with COCO objects.

Floor plan (world frame, metres, +x right, +y up, origin at the house centre):

    y=4  +-----------------+-----------------------------------+
         |   BATHROOM      |              BEDROOM              |
         |  toilet, sink   |   bed, tv, chair, potted plant    |
    y=0  +---  ---+--------+---  ---+--------------------------+
         |        door              door    |                  |
         |      LIVING ROOM                 |     KITCHEN      |
         |  couch, tv, chair, 2 plants      door fridge, sink, |
         |                                  |  table, 4 chairs |
    y=-4 +----------------------------------+------------------+
        x=-5            x=-1.5            x=0.5               x=5

Objects are named <coco_label>_<n> (spaces -> underscores) so that
extract_ground_truth.py can read the ground truth back out of the SDF.

Every COCO object also gets gz-sim's Label system with its COCO class index
(COCO_IDS below). Labels are invisible to the RGB camera; they only feed the
optional ground-truth boundingbox_camera (sim.launch.py gt_boxes:=true) that
the detector evaluation uses.

Run from the package root and commit the output:
    python3 scripts/generate_house_world.py
"""

import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'worlds', 'house.sdf')

FUEL = 'https://fuel.gazebosim.org/1.0/OpenRobotics/models/'
WALL_H = 2.4
WALL_T = 0.12
PI = math.pi

# Wall segments: (x0, y0, x1, y1). Door gaps are simply left out.
WALLS = [
    # outer shell
    (-5.0, -4.0, 5.0, -4.0), (-5.0, 4.0, 5.0, 4.0),
    (-5.0, -4.0, -5.0, 4.0), (5.0, -4.0, 5.0, 4.0),
    # y = 0 partition, doors at x in [-4.2, -3.2] (bathroom) and [-1.2, -0.2] (bedroom)
    (-5.0, 0.0, -4.2, 0.0), (-3.2, 0.0, -1.2, 0.0), (-0.2, 0.0, 5.0, 0.0),
    # x = 0.5 partition (living | kitchen), 1.4 m opening at y in [-2.8, -1.4]
    (0.5, -4.0, 0.5, -2.8), (0.5, -1.4, 0.5, 0.0),
    # x = -1.5 partition (bathroom | bedroom), solid
    (-1.5, 0.0, -1.5, 4.0),
]

# Floor tiles per room: (name, x0, y0, x1, y1, rgb)
FLOORS = [
    ('living', -5.0, -4.0, 0.5, 0.0, (0.55, 0.40, 0.26)),    # oak
    ('kitchen', 0.5, -4.0, 5.0, 0.0, (0.78, 0.77, 0.74)),    # light tile
    ('bathroom', -5.0, 0.0, -1.5, 4.0, (0.70, 0.78, 0.82)),  # blue tile
    ('bedroom', -1.5, 0.0, 5.0, 4.0, (0.62, 0.48, 0.34)),    # walnut
]

# COCO-80 class indices (same order as YOLO's class list) of the classes placed
# below. Used as gz Label values, so ground-truth boxes carry the COCO id.
COCO_IDS = {'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60,
            'toilet': 61, 'tv': 62, 'sink': 71, 'refrigerator': 72}

# Objects: (name, uri, x, y, z, yaw). Poses were tuned by rendering the world.
OBJECTS = [
    # living room
    ('couch_1', FUEL + 'Sofa', -2.6, -1.3, 0.0, -PI / 2),
    ('tv_stand_living', FUEL + 'TVStand', -2.6, -3.7, 0.0, 0.0),
    ('tv_1', 'model://tv', -2.6, -3.75, 0.45, PI / 2),
    ('chair_1', FUEL + 'Chair', -4.4, -2.0, 0.0, 0.0),
    ('potted_plant_1', 'model://potted_plant', -4.6, -3.6, 0.0, 0.0),
    ('potted_plant_2', 'model://potted_plant', 0.1, -3.6, 0.0, 1.0),
    # kitchen
    ('refrigerator_1', FUEL + 'Refrigerator', 4.5, -3.5, 0.0, PI),
    ('sink_1', FUEL + 'KitchenSink', 4.55, -2.3, 0.0, PI),
    ('dining_table_1', FUEL + 'Dining Table', 2.5, -1.8, 0.0, PI / 2),
    ('chair_2', FUEL + 'Dining Chair', 1.9, -0.95, 0.0, PI),
    ('chair_3', FUEL + 'Dining Chair', 3.1, -0.95, 0.0, PI),
    ('chair_4', FUEL + 'Dining Chair', 1.9, -2.65, 0.0, 0.0),
    ('chair_5', FUEL + 'Dining Chair', 3.1, -2.65, 0.0, 0.0),
    # bathroom
    ('toilet_1', FUEL + 'Toilet', -4.0, 3.55, 0.0, -PI / 2),
    ('sink_2', FUEL + 'BathroomSink', -2.6, 3.94, 0.55, 0.0),
    # bedroom
    ('bed_1', FUEL + 'Bed', 2.0, 2.75, 0.0, 0.0),
    ('tv_stand_bedroom', FUEL + 'TVStand', 2.0, 0.3, 0.0, PI),
    ('tv_2', 'model://tv', 2.0, 0.35, 0.45, PI / 2),
    ('chair_6', FUEL + 'Chair', -0.9, 3.4, 0.0, -PI / 4),
    ('potted_plant_3', 'model://potted_plant', 4.6, 0.4, 0.0, 2.0),
]


def fmt(*values):
    return ' '.join(f'{v:.4f}'.rstrip('0').rstrip('.') if v else '0' for v in values)


def wall_model(i, x0, y0, x1, y1):
    length = math.hypot(x1 - x0, y1 - y0) + WALL_T  # overlap at corners
    yaw = math.atan2(y1 - y0, x1 - x0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = fmt(length, WALL_T, WALL_H)
    return f"""
    <model name="wall_{i}">
      <static>true</static>
      <pose>{fmt(cx, cy, WALL_H / 2, 0, 0, yaw)}</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
          <material>
            <ambient>0.85 0.83 0.78 1</ambient><diffuse>0.92 0.90 0.85 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
          </material>
        </visual>
      </link>
    </model>"""


def floor_model(name, x0, y0, x1, y1, rgb):
    size = fmt(x1 - x0, y1 - y0, 0.01)
    col = fmt(*rgb)
    return f"""
    <model name="floor_{name}">
      <static>true</static>
      <pose>{fmt((x0 + x1) / 2, (y0 + y1) / 2, 0.005, 0, 0, 0)}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
          <material><ambient>{col} 1</ambient><diffuse>{col} 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular></material>
        </visual>
      </link>
    </model>"""


def coco_label(name):
    """'dining_table_1' -> 'dining table'."""
    return name.rsplit('_', 1)[0].replace('_', ' ')


def include(name, uri, x, y, z, yaw):
    label = COCO_IDS.get(coco_label(name))
    plugin = '' if label is None else f"""
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
        <label>{label}</label>
      </plugin>"""
    return f"""
    <include>
      <name>{name}</name>
      <uri>{uri}</uri>
      <pose>{fmt(x, y, z, 0, 0, yaw)}</pose>
      <!-- Force static: several Fuel models (Chair, Sofa, Toilet, KitchenSink) are
           dynamic with mesh collisions, which cost ~40% real-time factor. -->
      <static>true</static>{plugin}
    </include>"""


def main():
    body = ''.join(floor_model(*f) for f in FLOORS)
    body += ''.join(wall_model(i, *w) for i, w in enumerate(WALLS))
    body += ''.join(include(*o) for o in OBJECTS)
    sdf = f"""<?xml version="1.0"?>
<!-- GENERATED by scripts/generate_house_world.py - edit the script, not this file. -->
<sdf version="1.9">
  <world name="house">
    <physics name="3ms" type="ignored">
      <max_step_size>0.003</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>333.3333</real_time_update_rate>
    </physics>
    <!-- The Sensors system is loaded by the robot model (create3.urdf.xacro). -->
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>

    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.75 0.8 0.85 1</background>
      <shadows>false</shadows>
      <grid>false</grid>
    </scene>

    <!-- One light only (laptop GPU budget); no shadows. -->
    <light name="sun" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.85 0.85 0.82 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <material><ambient>0.5 0.55 0.45 1</ambient><diffuse>0.5 0.55 0.45 1</diffuse></material>
        </visual>
      </link>
    </model>
{body}
  </world>
</sdf>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(sdf)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
