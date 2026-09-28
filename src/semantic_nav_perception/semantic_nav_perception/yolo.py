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
YOLO (v8 / 11) pre- and post-processing plus an OpenVINO runner. No ROS here.

Pipeline for one BGR frame (H x W x 3, uint8):

    letterbox()      resize keeping aspect ratio, pad to W x H with grey (114)
    to_blob()        BGR->RGB, HWC->CHW, /255, add batch dim -> float32 1x3xHxW
    <network>        -> 1 x (4 + C) x N   (N = 8400 for 640x640; anchor-free)
    decode()         rows -> (cx, cy, w, h) boxes, best class + its score,
                     confidence filter, optional class whitelist
    nms()            class-aware non-maximum suppression (IoU threshold)
    unletterbox()    boxes back to original pixel coordinates, clipped

Everything except YoloDetector is pure numpy/OpenCV so it can be unit tested
without OpenVINO or a model file.
"""

from dataclasses import dataclass
import time

import cv2
import numpy as np

PAD_VALUE = 114  # grey used by Ultralytics during training; keep it identical


@dataclass(frozen=True)
class Letterbox:
    """How an image was mapped into the square network input."""

    scale: float   # network pixels per original pixel
    pad_x: float   # left padding in network pixels
    pad_y: float   # top padding in network pixels


@dataclass
class Detection:
    """One detection in original image pixels (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str = ''


def letterbox(image, size):
    """
    Resize `image` into a canvas of `size` = (width, height) without distorting it.

    The image is scaled by the largest factor that fits both dimensions and
    centred; the leftover border is filled with PAD_VALUE. An int `size` means
    a square canvas.
    """
    out_w, out_h = (size, size) if isinstance(size, int) else size
    h, w = image.shape[:2]
    scale = min(out_h / h, out_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (out_w - new_w) / 2.0, (out_h - new_h) / 2.0
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    canvas = np.full((out_h, out_w, 3), PAD_VALUE, dtype=np.uint8)
    # Bilinear, like Ultralytics' own LetterBox: match training preprocessing.
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(image, (new_w, new_h),
                                                            interpolation=cv2.INTER_LINEAR)
    return canvas, Letterbox(scale, float(left), float(top))


def to_blob(canvas_bgr):
    """uint8 BGR HxWx3 -> float32 RGB 1x3xHxW in [0, 1]."""
    rgb = canvas_bgr[:, :, ::-1]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def decode(output, conf_threshold, allowed_ids=None):
    """
    Turn the raw 1 x (4 + C) x N head output into candidate boxes.

    YOLOv8/11 are anchor-free and have no separate objectness score: each of
    the N rows carries (cx, cy, w, h) in network pixels followed by C class
    probabilities (already sigmoid-ed). A row's confidence is its best class
    probability.

    Returns (boxes_xyxy [K,4], scores [K], class_ids [K]).
    """
    preds = output[0].T                      # N x (4 + C)
    class_scores = preds[:, 4:]
    if allowed_ids is not None:
        mask = np.zeros(class_scores.shape[1], dtype=bool)
        mask[list(allowed_ids)] = True
        class_scores = np.where(mask, class_scores, 0.0)
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    keep = scores >= conf_threshold
    cx, cy, w, h = preds[keep, :4].T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes.astype(np.float32), scores[keep].astype(np.float32), class_ids[keep]


def iou(box, boxes):
    """Intersection over union of one xyxy box with each of an array of boxes."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes, scores, class_ids, iou_threshold, max_detections=100):
    """
    Class-aware greedy non-maximum suppression.

    Repeatedly keep the highest-scoring box and drop every remaining box of the
    SAME class that overlaps it by more than `iou_threshold`. Returns the kept
    indices, best first.
    """
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size and len(keep) < max_detections:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        same = class_ids[rest] == class_ids[best]
        overlap = iou(boxes[best], boxes[rest]) > iou_threshold
        order = rest[~(same & overlap)]
    return np.array(keep, dtype=int)


def unletterbox(boxes, lb, image_shape):
    """Map xyxy boxes from network pixels back to original image pixels."""
    h, w = image_shape[:2]
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - lb.pad_x) / lb.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - lb.pad_y) / lb.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def postprocess(output, lb, image_shape, conf_threshold, iou_threshold,
                class_names, allowed_ids=None):
    """Run decode -> nms -> unletterbox and return a list of Detection."""
    boxes, scores, ids = decode(output, conf_threshold, allowed_ids)
    if len(scores) == 0:
        return []
    keep = nms(boxes, scores, ids, iou_threshold)
    boxes = unletterbox(boxes[keep], lb, image_shape)
    return [Detection(*map(float, b), float(s), int(c), class_names[int(c)])
            for b, s, c in zip(boxes, scores[keep], ids[keep])]


def whitelist_ids(class_names, whitelist):
    """Class names -> set of indices; empty whitelist means every class."""
    if not whitelist:
        return None
    unknown = set(whitelist) - set(class_names)
    if unknown:
        raise ValueError(f'class_whitelist has names the model does not know: {sorted(unknown)}')
    return {class_names.index(name) for name in whitelist}


def draw(image, detections):
    """Return a copy of `image` with boxes and "label score" captions."""
    out = image.copy()
    for d in detections:
        color = tuple(int(c) for c in np.random.default_rng(d.class_id).integers(64, 256, 3))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        text = f'{d.label} {d.score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(p1[1], th + 4)
        cv2.rectangle(out, (p1[0], y - th - 4), (p1[0] + tw + 2, y), color, -1)
        cv2.putText(out, text, (p1[0] + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


class YoloDetector:
    """
    Load a YOLO ONNX model with OpenVINO and run it on BGR images.

    The ONNX graph has a dynamic input shape; it is reshaped once to
    1 x 3 x height x width (`input_size` = (width, height)) and compiled
    static, which lets OpenVINO pick the fastest kernels for exactly that size.
    A 640x480 camera image fits a 640x480 input with zero padding, which is
    25% less work than the 640x640 square used in training.
    """

    def __init__(self, model_path, class_names, input_size, device='CPU',
                 num_threads=0, cache_dir=''):
        try:  # imported here so the math above (and its tests) need no OpenVINO
            import openvino as ov
        except ImportError as e:
            raise RuntimeError('OpenVINO is not installed: pip install --user openvino '
                               '(see README, "Detection")') from e
        width, height = (input_size, input_size) if isinstance(input_size, int) else input_size
        if width % 32 or height % 32:
            raise ValueError('input size must be a multiple of 32 (the YOLO stride), '
                             f'got {input_size}')
        self.class_names = list(class_names)
        self.input_size = (int(width), int(height))
        core = ov.Core()
        if cache_dir:
            core.set_property({'CACHE_DIR': cache_dir})
        model = core.read_model(model_path)
        model.reshape([1, 3, self.input_size[1], self.input_size[0]])
        config = {'PERFORMANCE_HINT': 'LATENCY'}
        if num_threads > 0 and device == 'CPU':
            config['INFERENCE_NUM_THREADS'] = int(num_threads)
        self.compiled = core.compile_model(model, device, config)
        self.request = self.compiled.create_infer_request()
        n_out = self.compiled.output(0).get_partial_shape()[1].get_length()
        if n_out != 4 + len(self.class_names):
            raise ValueError(f'model outputs {n_out - 4} classes but {len(self.class_names)} '
                             'class names were given')
        self.timing = {}  # last call's stage durations in milliseconds

    def __call__(self, image_bgr, conf_threshold, iou_threshold, allowed_ids=None):
        """Detect objects in one BGR image; stage timings go to self.timing."""
        t0 = time.perf_counter()
        canvas, lb = letterbox(image_bgr, self.input_size)
        blob = to_blob(canvas)
        t1 = time.perf_counter()
        output = self.request.infer({0: blob})[self.compiled.output(0)]
        t2 = time.perf_counter()
        dets = postprocess(output, lb, image_bgr.shape, conf_threshold, iou_threshold,
                           self.class_names, allowed_ids)
        t3 = time.perf_counter()
        self.timing = {'pre': (t1 - t0) * 1e3, 'infer': (t2 - t1) * 1e3,
                       'post': (t3 - t2) * 1e3, 'total': (t3 - t0) * 1e3}
        return dets
