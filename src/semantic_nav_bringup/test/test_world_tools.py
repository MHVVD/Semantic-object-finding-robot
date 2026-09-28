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

"""Unit tests for the pose math, naming convention and bounding boxes in world_tools."""

import math
import os
import textwrap

import numpy as np
import pytest

from semantic_nav_bringup import world_tools as wt


def test_parse_pose_defaults_and_errors():
    assert wt.parse_pose(None) == (0.0,) * 6
    assert wt.parse_pose(' 1 2 3 0 0 1.5 ') == (1.0, 2.0, 3.0, 0.0, 0.0, 1.5)
    with pytest.raises(ValueError):
        wt.parse_pose('1 2 3')


def test_rotation_is_orthonormal():
    r = wt.rotation_rpy(0.3, -0.7, 2.1)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_yaw_rotates_x_axis_to_y_axis():
    t = wt.pose_to_matrix((0, 0, 0, 0, 0, math.pi / 2))
    np.testing.assert_allclose(wt.transform_points(t, [[1, 0, 0]]), [[0, 1, 0]], atol=1e-12)


def test_rpy_order_is_extrinsic_xyz():
    # roll 90 deg then yaw 90 deg (fixed axes): +Y -> +Z (roll) -> stays +Z (yaw)
    r = wt.rotation_rpy(math.pi / 2, 0, math.pi / 2)
    np.testing.assert_allclose(r @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    # and +X -> +X (roll) -> +Y (yaw)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_pose_composition_translation_then_rotation():
    world_model = wt.pose_to_matrix((2, 0, 0, 0, 0, math.pi / 2))
    model_link = wt.pose_to_matrix((1, 0, 0.5, 0, 0, 0))
    point = wt.transform_points(world_model @ model_link, [[0, 0, 0]])
    # the link's +1 m in x is rotated to +1 m in world y
    np.testing.assert_allclose(point, [[2, 1, 0.5]], atol=1e-12)
    assert wt.yaw_of(world_model) == pytest.approx(math.pi / 2)


@pytest.mark.parametrize('name, label', [
    ('chair_1', 'chair'),
    ('dining_table_12', 'dining table'),
    ('potted_plant_3', 'potted plant'),
    ('tv_stand_living', None),   # not <label>_<digits>
    ('wall_3', None),            # not a COCO label we use
    ('turtlebot4', None),
])
def test_label_from_name(name, label):
    assert wt.label_from_name(name) == label


def test_box_corners_extent():
    corners = wt.box_corners((2.0, 1.0, 0.5))
    np.testing.assert_allclose(corners.min(axis=0), [-1, -0.5, -0.25])
    np.testing.assert_allclose(corners.max(axis=0), [1, 0.5, 0.25])


def test_world_objects_bbox_centre(tmp_path):
    """A box resting on the floor, rotated 90 deg: centre and size come out right."""
    model_dir = tmp_path / 'models' / 'crate'
    model_dir.mkdir(parents=True)
    (model_dir / 'model.sdf').write_text(textwrap.dedent("""\
        <sdf version="1.9"><model name="crate"><link name="link">
          <visual name="v"><pose>0 0 0.25 0 0 0</pose>
            <geometry><box><size>2 1 0.5</size></box></geometry></visual>
        </link></model></sdf>"""))
    world = tmp_path / 'w.sdf'
    world.write_text(textwrap.dedent("""\
        <sdf version="1.9"><world name="w">
          <include><name>chair_1</name><uri>model://crate</uri>
                   <pose>3 4 0 0 0 1.5707963267948966</pose></include>
          <include><name>wall_0</name><uri>model://crate</uri></include>
        </world></sdf>"""))
    resolver = wt.ModelResolver(model_paths=[str(tmp_path / 'models')])
    objects = list(wt.world_objects(str(world), resolver))
    assert [o['name'] for o in objects] == ['chair_1']
    assert objects[0]['label'] == 'chair'
    np.testing.assert_allclose(objects[0]['position'], [3, 4, 0.25], atol=1e-3)
    np.testing.assert_allclose(objects[0]['size'], [1, 2, 0.5], atol=1e-3)  # rotated


def test_obj_loader_and_scale(tmp_path):
    obj = tmp_path / 'm.obj'
    obj.write_text('v 0 0 0\nv 1 2 3\nvn 0 0 1\nf 1 2 2\n')
    np.testing.assert_allclose(wt.load_obj_vertices(str(obj)), [[0, 0, 0], [1, 2, 3]])


def test_fuel_uri_resolution(tmp_path):
    cache = tmp_path / 'fuel'
    (cache / 'fuel.gazebosim.org' / 'openrobotics' / 'models' / 'dining table' / '1').mkdir(
        parents=True)
    (cache / 'fuel.gazebosim.org' / 'openrobotics' / 'models' / 'dining table' / '3').mkdir()
    resolver = wt.ModelResolver(fuel_cache=str(cache))
    path = resolver.resolve('https://fuel.gazebosim.org/1.0/OpenRobotics/models/Dining Table')
    assert path.endswith(os.path.join('dining table', '3'))  # newest version wins
