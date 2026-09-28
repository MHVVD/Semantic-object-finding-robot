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

"""Unit tests for box formats and detection metrics."""

import math

import pytest
from semantic_nav_perception.dataset import (cxcywh_to_xyxy, dataset_yaml, from_yolo_line,
                                             to_yolo_line, xyxy_to_cxcywh)
from semantic_nav_perception.evaluation import (accumulate, average_precision, box_iou,
                                                ClassStats, match_frame)


# ---------------------------------------------------------------- formats
def test_box_format_round_trip():
    box = (10.0, 20.0, 110.0, 70.0)
    assert xyxy_to_cxcywh(*box) == (60.0, 45.0, 100.0, 50.0)
    assert cxcywh_to_xyxy(*xyxy_to_cxcywh(*box)) == box


def test_yolo_line_round_trip():
    line = to_yolo_line(56, (64.0, 48.0, 320.0, 240.0), 640, 480)
    assert line == '56 0.300000 0.300000 0.400000 0.400000'
    cls, box = from_yolo_line(line, 640, 480)
    assert cls == 56 and box == pytest.approx((64.0, 48.0, 320.0, 240.0))


def test_yolo_line_clips_to_image():
    cls, box = from_yolo_line(to_yolo_line(1, (-20, -10, 100, 50), 640, 480), 640, 480)
    assert box == pytest.approx((0.0, 0.0, 100.0, 50.0), abs=1e-3)  # 6-decimal text


def test_dataset_yaml_lists_names_by_index():
    text = dataset_yaml('/data', ['person', 'bicycle'])
    assert 'path: /data' in text and '  0: person' in text and '  1: bicycle' in text


# ---------------------------------------------------------------- matching
A = (0.0, 0.0, 100.0, 100.0)
B = (200.0, 200.0, 300.0, 300.0)


def test_box_iou():
    assert box_iou(A, A) == 1.0
    assert box_iou(A, B) == 0.0
    assert box_iou(A, (50.0, 0.0, 150.0, 100.0)) == pytest.approx(1 / 3)


def test_match_tp_fp_fn():
    gts = [(1, A), (2, B)]
    preds = [(1, (5.0, 5.0, 100.0, 100.0), 0.9),    # TP for GT 0
             (1, B, 0.8)]                           # right place, wrong class -> FP
    m = match_frame(gts, preds)
    assert m.pred_status == ['tp', 'fp']
    assert m.gt_matched == [True, False]            # GT 1 is a false negative


def test_duplicate_detection_is_false_positive():
    m = match_frame([(1, A)], [(1, A, 0.6), (1, (2.0, 2.0, 100.0, 100.0), 0.9)])
    assert m.pred_status == ['fp', 'tp']            # higher score claims the GT first


def test_low_iou_is_not_a_match():
    m = match_frame([(1, A)], [(1, (60.0, 0.0, 160.0, 100.0), 0.9)], iou_match=0.5)
    assert m.pred_status == ['fp'] and m.gt_matched == [False]


def test_small_ground_truth_is_ignored():
    tiny = (0.0, 0.0, 10.0, 50.0)                   # 10 px wide
    m = match_frame([(1, tiny)], [(1, tiny, 0.9)], min_box_px=20)
    assert m.gt_ignored == [True] and m.pred_status == ['ignore']
    stats = {}
    accumulate(stats, [(1, tiny)], [(1, tiny, 0.9)], m)
    assert stats == {}                              # neither GT nor prediction counted


# ---------------------------------------------------------------- AP / P / R
def test_average_precision_perfect_and_empty():
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([], [], 3) == 0.0
    assert math.isnan(average_precision([0.5], [False], 0))


def test_average_precision_hand_computed():
    # Ranked: TP, FP, TP with 2 GT. PR points: (0.5, 1), (0.5, 0.5), (1.0, 2/3).
    # Envelope: precision 1 up to recall 0.5, then 2/3 -> AP = 0.5*1 + 0.5*2/3.
    ap = average_precision([0.9, 0.8, 0.7], [True, False, True], 2)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_average_precision_missed_ground_truth_caps_recall():
    assert average_precision([0.9], [True], 4) == pytest.approx(0.25)


def test_class_stats_at_threshold():
    s = ClassStats(n_gt=4, scores=[0.9, 0.7, 0.5, 0.3], is_tp=[True, False, True, True])
    tp, fp, fn, prec, rec = s.at_threshold(0.6)
    assert (tp, fp, fn) == (1, 1, 3)
    assert prec == pytest.approx(0.5) and rec == pytest.approx(0.25)
    assert s.at_threshold(0.0)[:3] == (3, 1, 1)
