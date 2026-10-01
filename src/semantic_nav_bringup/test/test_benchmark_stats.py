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

"""Unit tests for the benchmark scoring and statistics."""

import math

import pytest
from semantic_nav_bringup.benchmark_stats import (bearing_error, classify_goto, mean_var,
                                                  median, nearest_object, per_class_metrics,
                                                  sample_labels)
from semantic_nav_bringup.eval_utils import match_map_to_ground_truth

BOX = {'half': (0.25, 0.25), 'yaw': 0.0}
TRUTH = [{'name': 'chair_1', 'label': 'chair', 'xy': (0.0, 0.0), **BOX},
         {'name': 'chair_2', 'label': 'chair', 'xy': (3.0, 0.0), **BOX},
         {'name': 'sink_1', 'label': 'sink', 'xy': (0.0, 5.0), **BOX},
         {'name': 'tv_1', 'label': 'tv', 'xy': (5.0, 5.0), **BOX}]


def metrics(mapped):
    matches, _, _ = match_map_to_ground_truth(mapped, TRUTH, 0.5)
    return per_class_metrics(mapped, TRUTH, matches)


def test_per_class_counts_precision_recall():
    mapped = [{'label': 'chair', 'xy': (0.1, 0.0)},     # TP, 0.1 m
              {'label': 'chair', 'xy': (0.0, 0.3)},     # duplicate of chair_1 -> FP
              {'label': 'tv', 'xy': (5.0, 5.2)},        # TP, 0.2 m
              {'label': 'tv', 'xy': (9.0, 9.0)}]        # nothing there -> FP
    rows = metrics(mapped)
    chair = rows['chair']
    assert (chair['truth'], chair['mapped'], chair['tp'], chair['fp'], chair['fn']) == \
        (2, 2, 1, 1, 1)
    assert chair['precision'] == pytest.approx(0.5)
    assert chair['recall'] == pytest.approx(0.5)
    assert rows['tv']['precision'] == pytest.approx(0.5)
    assert rows['tv']['recall'] == pytest.approx(1.0)
    assert rows['tv']['mean_error'] == pytest.approx(0.2)
    total = rows['ALL']
    assert (total['tp'], total['fp'], total['fn']) == (2, 2, 2)
    assert total['precision'] == pytest.approx(0.5)
    assert total['recall'] == pytest.approx(0.5)
    assert total['median_error'] == pytest.approx(0.15)


def test_class_never_mapped_has_undefined_precision_and_zero_recall():
    rows = metrics([{'label': 'chair', 'xy': (0.0, 0.0)}])
    assert rows['sink']['precision'] is None          # no claims -> nothing to be precise about
    assert rows['sink']['recall'] == 0.0
    assert rows['sink']['mean_error'] is None


def test_class_absent_from_truth_has_undefined_recall():
    rows = metrics([{'label': 'bed', 'xy': (1.0, 1.0)}])
    assert rows['bed']['recall'] is None
    assert rows['bed']['precision'] == 0.0
    assert rows['bed']['fp'] == 1


def test_median_even_odd_empty():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert median([]) is None


def test_mean_var_is_sample_variance_and_skips_undefined():
    mean, var, n = mean_var([1.0, 2.0, 3.0, None, float('nan')])
    assert (mean, n) == (2.0, 3)
    assert var == pytest.approx(1.0)                  # sum of squares 2 / (3 - 1)
    assert mean_var([5.0]) == (5.0, None, 1)          # one run: no variance
    assert mean_var([None]) == (None, None, 0)


def test_nearest_object_uses_footprint_and_label():
    obj, d_foot, d_centre = nearest_object((1.0, 0.0), 'chair', TRUTH)
    assert obj['name'] == 'chair_1'
    assert d_foot == pytest.approx(0.75)
    assert d_centre == pytest.approx(1.0)
    assert nearest_object((0.0, 0.0), 'bed', TRUTH) == (None, math.inf, math.inf)


def test_bearing_error():
    assert bearing_error((0, 0), 0.0, (1, 0)) == pytest.approx(0.0)
    assert bearing_error((0, 0), 0.0, (0, 1)) == pytest.approx(math.pi / 2)
    assert bearing_error((0, 0), math.radians(170), (-1, -0.1)) == \
        pytest.approx(math.radians(15.711), abs=1e-4)   # wraps across +-pi


def test_classify_goto():
    assert classify_goto(True, 'arrived', 0.4, 1.0) == 'reached'
    assert classify_goto(True, 'arrived', 2.5, 1.0) == 'wrong_place'
    assert classify_goto(False, 'no sink in the semantic map. Known objects: tv', 9, 1.0) == \
        'not_in_map'
    assert classify_goto(False, '1 tv(s) in the map, but no reachable goal pose next to any',
                         9, 1.0) == 'no_goal'
    assert classify_goto(False, 'navigation to tv #3 aborted after 40 s', 9, 1.0) == 'nav_failed'
    assert classify_goto(False, '', 0.2, 1.0, timed_out=True) == 'timeout'


def test_sample_labels_reproducible_and_no_immediate_repeats():
    labels = ['a', 'b', 'c']
    first = sample_labels(labels, 50, seed=7)
    assert first == sample_labels(labels, 50, seed=7)
    assert first != sample_labels(labels, 50, seed=8)
    assert len(first) == 50 and set(first) <= set(labels)
    assert all(x != y for x, y in zip(first, first[1:]))
    assert sample_labels(['only'], 3, seed=1) == ['only'] * 3
