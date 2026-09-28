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

"""Sanity checks on the generated house world and its ground-truth file (no meshes needed)."""

import os
import xml.etree.ElementTree as ET

from semantic_nav_bringup import world_tools as wt
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(PKG, 'worlds', 'house.sdf')
GROUND_TRUTH = os.path.join(PKG, 'config', 'ground_truth.yaml')


def labelled_includes():
    world = ET.parse(WORLD).getroot().find('world')
    out = {}
    for inc in world.findall('include'):
        name = inc.findtext('name')
        if wt.label_from_name(name):
            out[name] = wt.parse_pose(inc.findtext('pose'))
    return out


def test_all_target_classes_present():
    labels = {wt.label_from_name(n) for n in labelled_includes()}
    assert labels == set(wt.COCO_LABELS)


def test_some_classes_have_several_instances():
    counts = {}
    for name in labelled_includes():
        label = wt.label_from_name(name)
        counts[label] = counts.get(label, 0) + 1
    assert counts['chair'] >= 3          # data association must separate them
    assert sum(1 for c in counts.values() if c >= 2) >= 3


def test_objects_inside_the_house():
    for name, (x, y, *_rest) in labelled_includes().items():
        assert -5.0 < x < 5.0 and -4.0 < y < 4.0, name


def test_ground_truth_file_matches_world():
    with open(GROUND_TRUTH) as f:
        gt = yaml.safe_load(f)
    assert {o['name'] for o in gt['objects']} == set(labelled_includes())
    for obj in gt['objects']:
        x, y, *_ = labelled_includes()[obj['name']]
        # bbox centre should be close to the model origin in the ground plane
        assert abs(obj['position'][0] - x) < 1.0 and abs(obj['position'][1] - y) < 1.0
        assert obj['position'][2] > 0.0
