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
Scoring and statistics for the M8 benchmarks. No ROS imports.

Semantic map (one run):
    TP_c  map objects of class c matched one-to-one to a ground-truth c
          (eval_utils.match_map_to_ground_truth: footprint gate + Hungarian)
    FP_c  map objects of class c left unmatched (wrong place, duplicate, or
          hallucinated)
    FN_c  ground-truth objects of class c left unmatched
    precision_c = TP_c / (TP_c + FP_c)   "of what the map claims, how much is real"
    recall_c    = TP_c / (TP_c + FN_c)   "of what is real, how much is in the map"
A ratio with a zero denominator is undefined (None), not 0 or 1: a map with no
sinks has no sink precision to report.

Navigation (one GoTo): the robot's TRUE final pose is compared with the
nearest ground-truth object of the requested class. The commander's own
"succeeded" only means Nav2 reached the pose the commander chose; if that pose
was next to a false positive, the robot is in the wrong place.

Repeated runs: `mean_var` gives the mean and the SAMPLE variance (n - 1).
"""

import math
import random

from semantic_nav_bringup.eval_utils import distance_to_box

OUTCOMES = ('reached', 'wrong_place', 'not_in_map', 'no_goal', 'nav_failed', 'timeout')


def ratio(num, den):
    return num / den if den else None


def per_class_metrics(mapped, truth, matches):
    """
    Per-class TP/FP/FN, precision, recall and position errors for one map.

    mapped: [{'label', ...}], truth: [{'label', ...}],
    matches: [(map_index, truth_index, error_m)] from match_map_to_ground_truth.
    Returns {label: row} plus the pooled row under key 'ALL'; a row has
    truth, mapped, tp, fp, fn, precision, recall, errors, mean_error, median_error.
    """
    labels = sorted({m['label'] for m in mapped} | {t['label'] for t in truth})
    rows = {}
    for label in labels + ['ALL']:
        def keep(lbl, label=label):
            return label == 'ALL' or lbl == label
        errors = sorted(e for _, t, e in matches if keep(truth[t]['label']))
        n_truth = sum(keep(t['label']) for t in truth)
        n_mapped = sum(keep(m['label']) for m in mapped)
        tp = len(errors)
        rows[label] = {
            'truth': n_truth, 'mapped': n_mapped, 'tp': tp,
            'fp': n_mapped - tp, 'fn': n_truth - tp,
            'precision': ratio(tp, n_mapped), 'recall': ratio(tp, n_truth),
            'errors': errors,
            'mean_error': sum(errors) / tp if tp else None,
            'median_error': median(errors),
        }
    return rows


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def mean_var(values):
    """(mean, sample variance, n) over the defined values (None / NaN skipped)."""
    v = [x for x in values if x is not None and not math.isnan(x)]
    n = len(v)
    if n == 0:
        return None, None, 0
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / (n - 1) if n > 1 else None
    return mean, var, n


def nearest_object(xy, label, truth):
    """
    Closest ground-truth object of `label` to a robot position.

    Returns (object, footprint_distance, centre_distance) or (None, inf, inf).
    """
    best = (None, math.inf, math.inf)
    for t in truth:
        if t['label'] != label:
            continue
        d = distance_to_box(xy, t['xy'], t['half'], t['yaw'])
        if d < best[1]:
            best = (t, d, math.dist(xy, t['xy']))
    return best


def bearing_error(robot_xy, robot_yaw, target_xy):
    """|angle| between the robot's heading and the direction to target, in [0, pi]."""
    a = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]) - robot_yaw
    return abs(math.atan2(math.sin(a), math.cos(a)))


def classify_goto(success, message, final_distance, reach_m, timed_out=False):
    """
    One of OUTCOMES for a GoTo call.

    reached      the commander succeeded AND the robot ended within reach_m of
                 the footprint of a real object of that class
    wrong_place  the commander succeeded but no real object of that class is
                 within reach_m (it drove to a false positive / misplaced object)
    not_in_map   the semantic map has no confirmed instance of the class
    no_goal      instances exist but no reachable goal pose next to any of them
    nav_failed   Nav2 aborted / canceled, or anything else that failed
    timeout      no answer within the benchmark's time limit (goal canceled)
    """
    if timed_out:
        return 'timeout'
    if success:
        return 'reached' if final_distance <= reach_m else 'wrong_place'
    if 'in the semantic map' in message:
        return 'not_in_map'
    if 'no reachable goal' in message:
        return 'no_goal'
    return 'nav_failed'


def sample_labels(labels, n, seed):
    """Draw n labels uniformly with replacement, never the same twice in a row."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        label = rng.choice(labels)
        if len(labels) > 1 and out and label == out[-1]:
            continue
        out.append(label)
    return out
