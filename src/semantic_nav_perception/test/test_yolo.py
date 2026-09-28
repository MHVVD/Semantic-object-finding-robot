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

"""Unit tests for YOLO pre/post-processing (no OpenVINO, no model file)."""

import numpy as np
import pytest
from semantic_nav_perception.yolo import (decode, iou, Letterbox, letterbox, nms,
                                          PAD_VALUE, postprocess, to_blob, unletterbox,
                                          whitelist_ids)


def fake_output(rows, n_classes=3):
    """Build a 1 x (4 + C) x N head output from (cx, cy, w, h, class, score) rows."""
    out = np.zeros((1, 4 + n_classes, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


# ---------------------------------------------------------------- letterbox
def test_letterbox_square_from_landscape():
    image = np.full((480, 640, 3), 7, dtype=np.uint8)
    canvas, lb = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3)
    assert lb == Letterbox(1.0, 0.0, 80.0)          # 160 px of padding split 80 / 80
    assert (canvas[:80] == PAD_VALUE).all() and (canvas[560:] == PAD_VALUE).all()
    assert (canvas[80:560] == 7).all()


def test_letterbox_exact_fit_has_no_padding():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, lb = letterbox(image, (640, 480))
    assert canvas.shape == (480, 640, 3)
    assert lb == Letterbox(1.0, 0.0, 0.0)


def test_letterbox_downscale_keeps_aspect_ratio():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, lb = letterbox(image, 320)
    assert lb.scale == pytest.approx(0.5)
    assert lb.pad_y == 40.0 and lb.pad_x == 0.0     # 640x480 -> 320x240, 80 px pad


def test_to_blob_layout_and_range():
    canvas = np.zeros((2, 3, 3), dtype=np.uint8)
    canvas[..., 0] = 255                            # blue channel in BGR
    blob = to_blob(canvas)
    assert blob.shape == (1, 3, 2, 3) and blob.dtype == np.float32
    assert (blob[0, 2] == 1.0).all()                # blue ends up last (RGB order)
    assert (blob[0, :2] == 0.0).all()


def test_unletterbox_inverts_letterbox_mapping():
    lb = Letterbox(scale=0.5, pad_x=0.0, pad_y=40.0)
    net = np.array([[10.0, 50.0, 110.0, 140.0]], dtype=np.float32)
    orig = unletterbox(net, lb, (480, 640))
    np.testing.assert_allclose(orig, [[20.0, 20.0, 220.0, 200.0]])


def test_unletterbox_clips_to_image():
    lb = Letterbox(1.0, 0.0, 80.0)
    orig = unletterbox(np.array([[-5.0, 60.0, 700.0, 600.0]]), lb, (480, 640))
    np.testing.assert_allclose(orig, [[0.0, 0.0, 640.0, 480.0]])


# ---------------------------------------------------------------- decode
def test_decode_converts_centre_size_and_filters_confidence():
    out = fake_output([(100, 50, 20, 10, 1, 0.9), (0, 0, 5, 5, 0, 0.1)])
    boxes, scores, ids = decode(out, 0.5)
    np.testing.assert_allclose(boxes, [[90, 45, 110, 55]])
    np.testing.assert_allclose(scores, [0.9])
    assert ids.tolist() == [1]


def test_decode_picks_best_class_per_row():
    out = fake_output([(10, 10, 4, 4, 0, 0.6)])
    out[0, 4 + 2, 0] = 0.8
    _, scores, ids = decode(out, 0.5)
    assert ids.tolist() == [2] and scores[0] == pytest.approx(0.8)


def test_decode_whitelist_falls_back_to_best_allowed_class():
    out = fake_output([(10, 10, 4, 4, 0, 0.9)])
    out[0, 4 + 1, 0] = 0.7                          # second-best class is allowed
    _, scores, ids = decode(out, 0.5, allowed_ids={1})
    assert ids.tolist() == [1] and scores[0] == pytest.approx(0.7)
    assert len(decode(out, 0.5, allowed_ids={2})[1]) == 0


# ---------------------------------------------------------------- IoU / NMS
def test_iou_values():
    box = np.array([0, 0, 10, 10], dtype=float)
    others = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou(box, others), [1.0, 50 / 150, 0.0])


def test_nms_suppresses_same_class_overlap_only():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [1, 0, 11, 10], [50, 50, 60, 60]],
                     dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    ids = np.array([0, 0, 1, 0])
    keep = nms(boxes, scores, ids, iou_threshold=0.5)
    # box 1 overlaps box 0 (same class) -> gone; box 2 is another class -> kept.
    assert keep.tolist() == [0, 2, 3]


def test_nms_threshold_is_strict():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)   # IoU = 1/3
    scores, ids = np.array([0.9, 0.8]), np.array([0, 0])
    assert nms(boxes, scores, ids, iou_threshold=0.34).tolist() == [0, 1]
    assert nms(boxes, scores, ids, iou_threshold=0.33).tolist() == [0]


def test_nms_respects_max_detections():
    boxes = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(5)], dtype=float)
    keep = nms(boxes, np.linspace(0.9, 0.5, 5), np.zeros(5, dtype=int), 0.5, max_detections=3)
    assert keep.tolist() == [0, 1, 2]


# ---------------------------------------------------------------- end to end
def test_postprocess_end_to_end():
    names = ['a', 'b', 'c']
    lb = Letterbox(scale=0.5, pad_x=0.0, pad_y=40.0)   # 640x480 image in a 320 canvas
    out = fake_output([(60, 90, 100, 100, 2, 0.95),  # -> (20, 0, 220, 200) original
                       (62, 90, 100, 100, 2, 0.90),  # duplicate, removed by NMS
                       (200, 200, 20, 20, 0, 0.30)])  # below threshold
    dets = postprocess(out, lb, (480, 640), 0.5, 0.5, names)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == 'c' and d.score == pytest.approx(0.95)
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((20, 0, 220, 200))


def test_postprocess_empty():
    assert postprocess(fake_output([]), Letterbox(1, 0, 0), (480, 640), 0.5, 0.5, 'abc') == []


def test_whitelist_ids():
    names = ['person', 'chair', 'tv']
    assert whitelist_ids(names, []) is None
    assert whitelist_ids(names, ['tv', 'chair']) == {1, 2}
    with pytest.raises(ValueError):
        whitelist_ids(names, ['sofa'])
