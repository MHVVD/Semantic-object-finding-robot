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

"""Resolve file paths given as ROS parameters."""

import os


def resolve_path(path, package='semantic_nav_perception'):
    """
    Expand ~ and $VARS; a relative path is taken relative to `package`'s share dir.

    So the YAML can say `models/yolo11n.onnx` instead of an absolute path that
    only exists on one machine.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return path
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory(package), path)
