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
Box formats and the captured-dataset layout. No ROS imports.

Box conventions used in this package:
    xyxy   (x1, y1, x2, y2) pixel corners, top-left / bottom-right
    cxcywh (cx, cy, w, h) pixel centre + size -- what vision_msgs/BoundingBox2D holds
    yolo   (cx, cy, w, h) divided by image width/height -> [0, 1]; one text line
           "<class_id> <cx> <cy> <w> <h>" per object in labels/<frame>.txt

Dataset directory written by capture_frames and read by evaluate_detector:
    <root>/images/<frame>.png      RGB frames
    <root>/labels/<frame>.txt      YOLO labels from the simulator's ground-truth
                                   boxes (only when gt boxes were available)
    <root>/frames.csv              frame, stamp, robot x, y, yaw (map frame)
    <root>/dataset.yaml            Ultralytics dataset file (COCO-80 names)
"""


def cxcywh_to_xyxy(cx, cy, w, h):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def xyxy_to_cxcywh(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def to_yolo_line(class_id, box_xyxy, image_w, image_h):
    """Convert xyxy pixels to 'id cx cy w h' normalised to [0, 1] (clipped to the image)."""
    x1, y1, x2, y2 = box_xyxy
    x1, x2 = max(0.0, x1), min(float(image_w), x2)
    y1, y2 = max(0.0, y1), min(float(image_h), y2)
    cx, cy, w, h = xyxy_to_cxcywh(x1, y1, x2, y2)
    return (f'{int(class_id)} {cx / image_w:.6f} {cy / image_h:.6f} '
            f'{w / image_w:.6f} {h / image_h:.6f}')


def from_yolo_line(line, image_w, image_h):
    """Inverse of to_yolo_line -> (class_id, xyxy pixels)."""
    parts = line.split()
    cls = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return cls, cxcywh_to_xyxy(cx * image_w, cy * image_h, w * image_w, h * image_h)


def dataset_yaml(root, class_names):
    """Ultralytics dataset description; val = train until real splits exist."""
    lines = [f'path: {root}', 'train: images', 'val: images', 'names:']
    lines += [f'  {i}: {name}' for i, name in enumerate(class_names)]
    return '\n'.join(lines) + '\n'
