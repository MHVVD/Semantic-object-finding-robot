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

"""Smoke tests: the generated Python types exist and have the agreed fields."""

from semantic_nav_interfaces.msg import SemanticObject, SemanticObjectArray
from semantic_nav_interfaces.srv import GoTo, ListObjects, MapFile


def test_semantic_object_fields():
    obj = SemanticObject()
    obj.header.frame_id = 'map'
    obj.label = 'refrigerator'
    obj.id = 7
    obj.position.x, obj.position.y, obj.position.z = 1.5, -2.0, 0.9
    obj.confidence = 0.8
    obj.observation_count = 12
    assert obj.label == 'refrigerator'
    assert obj.position.y == -2.0
    assert abs(obj.confidence - 0.8) < 1e-6  # float32 round-trip


def test_semantic_object_array_holds_objects():
    arr = SemanticObjectArray()
    arr.objects = [SemanticObject(label='chair'), SemanticObject(label='tv')]
    assert [o.label for o in arr.objects] == ['chair', 'tv']


def test_goto_service_shape():
    req = GoTo.Request(label='refrigerator')
    res = GoTo.Response(success=True, message='ok')
    assert req.label == 'refrigerator'
    assert res.success and res.message == 'ok'


def test_list_objects_service_shape():
    req = ListObjects.Request()
    assert req.label_filter == ''  # empty filter means "all"
    res = ListObjects.Response(objects=[SemanticObject(label='sink')])
    assert res.objects[0].label == 'sink'


def test_map_file_service_shape():
    req = MapFile.Request()
    assert req.path == ''  # empty path means "the node's map_file parameter"
    res = MapFile.Response(success=True, message='saved', object_count=14)
    assert res.success and res.object_count == 14
