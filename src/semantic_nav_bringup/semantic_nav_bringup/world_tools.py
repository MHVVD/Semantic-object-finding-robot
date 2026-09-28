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
Pure-Python helpers for reading ground truth out of a Gazebo world SDF.

The world convention is that every object we want to evaluate against is an
``<include>`` (or ``<model>``) whose name is ``<coco_label>_<n>``, with spaces in
the COCO label written as underscores, e.g. ``dining_table_1``, ``chair_3``.

For each such object we compute the axis-aligned bounding box of its visual
meshes in the world frame. The box centre is the ground-truth position: it is
what a detector + depth deprojection will estimate, whereas the model origin is
usually on the floor.

Nothing here imports ROS so it can be unit tested with plain pytest.
"""

import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

# COCO class names we place in the world. Used to parse object names.
COCO_LABELS = (
    'chair', 'couch', 'bed', 'refrigerator', 'tv', 'dining table',
    'toilet', 'sink', 'potted plant',
)

_NAME_RE = re.compile(r'^(?P<label>[a-z_]+?)_(?P<index>\d+)$')


# --------------------------------------------------------------------------- poses
def parse_pose(text):
    """Parse an SDF ``<pose>`` string 'x y z roll pitch yaw' into 6 floats."""
    if text is None or not text.strip():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [float(v) for v in text.split()]
    if len(values) != 6:
        raise ValueError(f'pose needs 6 values, got {len(values)}: {text!r}')
    return tuple(values)


def rotation_rpy(roll, pitch, yaw):
    """
    Rotation matrix for SDF/URDF roll-pitch-yaw.

    SDF uses fixed-axis (extrinsic) X-Y-Z rotations, i.e.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def pose_to_matrix(pose):
    """Convert (x, y, z, roll, pitch, yaw) into a 4x4 homogeneous transform."""
    x, y, z, roll, pitch, yaw = pose
    t = np.eye(4)
    t[:3, :3] = rotation_rpy(roll, pitch, yaw)
    t[:3, 3] = (x, y, z)
    return t


def transform_points(t, points):
    """Apply a 4x4 transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=float)
    return points @ t[:3, :3].T + t[:3, 3]


def yaw_of(t):
    """Yaw (rotation about +Z) of a 4x4 transform, in radians."""
    return math.atan2(t[1, 0], t[0, 0])


# ------------------------------------------------------------------------- labels
def label_from_name(name):
    """
    Map an object name like 'dining_table_2' to its COCO label ('dining table').

    Returns None for names that do not follow the convention or whose label is
    not one of COCO_LABELS (walls, floor, the robot, ...).
    """
    match = _NAME_RE.match(name)
    if not match:
        return None
    label = match.group('label').replace('_', ' ')
    return label if label in COCO_LABELS else None


# ------------------------------------------------------------------------- meshes
def load_obj_vertices(path):
    """Read the vertex positions ('v x y z' lines) of a Wavefront OBJ file."""
    vertices = []
    with open(path, errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(v) for v in line.split()[1:4]])
    return np.array(vertices, dtype=float).reshape(-1, 3)


def load_dae_vertices(path):
    """
    Read all vertex positions of a COLLADA file, in metres, Z-up.

    Scene-graph node transforms are applied (pycollada's bound geometry) and the
    <unit meter="..."> scale is honoured, as Gazebo's mesh loader does.
    """
    import collada  # imported lazily: only needed for .dae models

    mesh = collada.Collada(path, ignore=[collada.common.DaeUnsupportedError,
                                         collada.common.DaeBrokenRefError])
    chunks = []
    for geometry in mesh.scene.objects('geometry'):
        for primitive in geometry.primitives():
            if primitive.vertex is not None and len(primitive.vertex):
                chunks.append(np.asarray(primitive.vertex, dtype=float))
    vertices = np.vstack(chunks) if chunks else np.zeros((0, 3))
    unit = mesh.assetInfo.unitmeter or 1.0
    vertices = vertices * unit
    if mesh.assetInfo.upaxis == 'Y_UP':
        vertices = vertices[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    return vertices


def load_mesh_vertices(path):
    """Dispatch on file extension; returns an (N, 3) array in metres."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        return load_obj_vertices(path)
    if ext == '.dae':
        return load_dae_vertices(path)
    raise ValueError(f'unsupported mesh format: {path}')


def box_corners(size):
    """Return the 8 corners of an origin-centred box of the given (x, y, z) size."""
    hx, hy, hz = (s / 2.0 for s in size)
    return np.array([[sx * hx, sy * hy, sz * hz]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def cylinder_corners(radius, length):
    """Return the corners of a cylinder's bounding box (cylinder axis along Z)."""
    return box_corners((2 * radius, 2 * radius, length))


# ------------------------------------------------------------------ model lookup
class ModelResolver:
    """
    Turn an SDF model URI into a local model directory.

    Handles 'model://name' (searched in GZ_SIM_RESOURCE_PATH-style paths) and
    Fuel URLs 'https://fuel.gazebosim.org/1.0/<owner>/models/<name>[/<ver>]',
    which Gazebo caches under ~/.gz/fuel/fuel.gazebosim.org/<owner>/models/<name>/<ver>.
    """

    def __init__(self, model_paths=(), fuel_cache=None):
        self.model_paths = list(model_paths)
        self.fuel_cache = fuel_cache or os.path.expanduser('~/.gz/fuel')

    def resolve(self, uri):
        if uri.startswith('model://'):
            name = uri[len('model://'):].split('/')[0]
            for base in self.model_paths:
                candidate = os.path.join(base, name)
                if os.path.isfile(os.path.join(candidate, 'model.sdf')):
                    return candidate
            raise FileNotFoundError(f'{uri} not found in {self.model_paths}')
        match = re.match(r'https?://([^/]+)/1\.0/([^/]+)/models/([^/]+)(?:/(\d+|tip))?', uri)
        if match:
            host, owner, name, version = match.groups()
            name = name.replace('%20', ' ')
            base = os.path.join(self.fuel_cache, host.lower(), owner.lower(), 'models',
                                name.lower())
            if not os.path.isdir(base):
                raise FileNotFoundError(
                    f'{uri} is not in the Fuel cache ({base}); run the world once '
                    'or `gz fuel download -u <uri>` first')
            versions = sorted((v for v in os.listdir(base) if v.isdigit()), key=int)
            chosen = version if version and version.isdigit() else versions[-1]
            return os.path.join(base, chosen)
        raise ValueError(f'unsupported model uri: {uri}')

    def resolve_mesh(self, mesh_uri, model_dir):
        """Resolve a <mesh><uri> that appears inside a model's model.sdf."""
        if mesh_uri.startswith('model://'):
            name, _, rest = mesh_uri[len('model://'):].partition('/')
            return os.path.join(self.resolve('model://' + name), rest)
        match = re.match(r'https?://[^/]+/1\.0/[^/]+/models/[^/]+/[^/]+/files/(.*)', mesh_uri)
        if match:  # Fuel-absolute URL pointing at a file of this same model
            return os.path.join(model_dir, match.group(1))
        return os.path.join(model_dir, mesh_uri)


# ---------------------------------------------------------------------- bounding
def model_visual_points(model_elem, model_dir, resolver):
    """
    Collect every visual's geometry points, expressed in the model frame.

    Supports <mesh>, <box> and <cylinder> visuals and link/visual poses.
    """
    chunks = []
    for link in model_elem.findall('link'):
        t_model_link = pose_to_matrix(parse_pose(link.findtext('pose')))
        for visual in link.findall('visual'):
            t_link_visual = pose_to_matrix(parse_pose(visual.findtext('pose')))
            geometry = visual.find('geometry')
            if geometry is None:
                continue
            if geometry.find('mesh') is not None:
                mesh = geometry.find('mesh')
                path = resolver.resolve_mesh(mesh.findtext('uri').strip(), model_dir)
                points = load_mesh_vertices(path)
                scale = mesh.findtext('scale')
                if scale:
                    points = points * np.array([float(s) for s in scale.split()])
            elif geometry.find('box') is not None:
                size = [float(s) for s in geometry.find('box').findtext('size').split()]
                points = box_corners(size)
            elif geometry.find('cylinder') is not None:
                cyl = geometry.find('cylinder')
                points = cylinder_corners(float(cyl.findtext('radius')),
                                          float(cyl.findtext('length')))
            else:
                continue
            chunks.append(transform_points(t_model_link @ t_link_visual, points))
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def load_model_sdf(model_dir):
    """Return the <model> element of a model directory's model.sdf."""
    root = ET.parse(os.path.join(model_dir, 'model.sdf')).getroot()
    model = root.find('model')
    if model is None:
        raise ValueError(f'no <model> in {model_dir}/model.sdf')
    return model


def world_objects(world_path, resolver):
    """
    Yield ground-truth dicts for every labelled object in a world SDF.

    Each dict has: name, label, position [x, y, z] (bbox centre, world frame),
    yaw (model yaw, rad), size [dx, dy, dz] (axis-aligned bbox extent).
    """
    root = ET.parse(world_path).getroot()
    world = root.find('world')
    for elem in list(world.findall('include')) + list(world.findall('model')):
        name = elem.findtext('name') if elem.tag == 'include' else elem.get('name')
        label = label_from_name(name or '')
        if label is None:
            continue
        t_world_model = pose_to_matrix(parse_pose(elem.findtext('pose')))
        if elem.tag == 'include':
            model_dir = resolver.resolve(elem.findtext('uri').strip())
            model_elem = load_model_sdf(model_dir)
            # A <pose> directly inside the model.sdf offsets the model frame.
            t_world_model = t_world_model @ pose_to_matrix(
                parse_pose(model_elem.findtext('pose')))
        else:
            model_dir, model_elem = os.path.dirname(world_path), elem
        points = transform_points(t_world_model,
                                  model_visual_points(model_elem, model_dir, resolver))
        if not len(points):
            raise ValueError(f'{name}: no visual geometry found')
        lo, hi = points.min(axis=0), points.max(axis=0)
        yield {
            'name': name,
            'label': label,
            'position': [round(float(v), 3) for v in (lo + hi) / 2.0],
            'yaw': round(yaw_of(t_world_model), 4),
            'size': [round(float(v), 3) for v in hi - lo],
        }
