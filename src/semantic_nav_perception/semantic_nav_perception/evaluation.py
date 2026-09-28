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
Detection metrics: matching predictions to ground truth, precision/recall, AP50.

Definitions (the usual PASCAL VOC / COCO ones, at a single IoU threshold):
    A prediction is a TRUE POSITIVE if it overlaps a not-yet-matched ground-truth
    box of the SAME class with IoU >= iou_match; predictions are matched in
    descending score order, so each ground-truth box absorbs at most one
    prediction (duplicates become false positives).
    Otherwise it is a FALSE POSITIVE. Unmatched ground truth = FALSE NEGATIVE.

    Ground-truth boxes that are too small (shorter side < min_box_px) are
    "ignored": they are neither required (no FN) nor penalised (a prediction on
    them is neither TP nor FP). This follows COCO's crowd/ignore regions and
    stops sub-20-pixel slivers at the image edge from dominating the result.

    AP50 = area under the precision-recall curve built by sweeping the score
    threshold, using all-point interpolation (precision made monotonically
    decreasing before integrating).
"""

from dataclasses import dataclass, field

import numpy as np


def box_iou(a, b):
    """Intersection over union of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FrameMatch:
    """Outcome of matching one frame's predictions against its ground truth."""

    pred_status: list      # per prediction: 'tp', 'fp' or 'ignore'
    gt_matched: list       # per ground-truth box: True if a prediction matched it
    gt_ignored: list       # per ground-truth box: True if too small to count


def match_frame(gts, preds, iou_match=0.5, min_box_px=0.0):
    """
    Match one frame.

    gts:   list of (class_id, xyxy)
    preds: list of (class_id, xyxy, score)
    """
    ignored = [min(b[2] - b[0], b[3] - b[1]) < min_box_px for _, b in gts]
    matched = [False] * len(gts)
    status = [''] * len(preds)
    for i in sorted(range(len(preds)), key=lambda k: -preds[k][2]):
        cls, box, _ = preds[i]
        best, best_iou = None, iou_match
        for j, (gcls, gbox) in enumerate(gts):
            if gcls != cls or matched[j]:
                continue
            o = box_iou(box, gbox)
            if o >= best_iou:
                best, best_iou = j, o
        if best is None:
            status[i] = 'fp'
        elif ignored[best]:
            matched[best] = True
            status[i] = 'ignore'
        else:
            matched[best] = True
            status[i] = 'tp'
    return FrameMatch(status, matched, ignored)


def average_precision(scores, is_tp, n_gt):
    """All-point interpolated AP from per-prediction scores and TP flags."""
    if n_gt == 0:
        return float('nan')
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind='stable')
    tp = np.asarray(is_tp, dtype=float)[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # Envelope: precision at recall r = best precision at any recall >= r.
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


@dataclass
class ClassStats:
    """Accumulated results for one class over a dataset."""

    n_gt: int = 0
    scores: list = field(default_factory=list)   # every non-ignored prediction
    is_tp: list = field(default_factory=list)
    gt_heights: list = field(default_factory=list)

    def at_threshold(self, conf):
        """(TP, FP, FN, precision, recall) using only predictions with score >= conf."""
        s = np.asarray(self.scores)
        t = np.asarray(self.is_tp, dtype=bool)
        keep = s >= conf if len(s) else np.zeros(0, dtype=bool)
        tp = int(np.sum(t[keep]))
        fp = int(np.sum(~t[keep]))
        fn = self.n_gt - tp
        precision = tp / (tp + fp) if tp + fp else float('nan')
        recall = tp / self.n_gt if self.n_gt else float('nan')
        return tp, fp, fn, precision, recall

    def ap(self):
        return average_precision(self.scores, self.is_tp, self.n_gt)


def accumulate(stats, gts, preds, match):
    """Add one frame's match to per-class `stats` (dict class_id -> ClassStats)."""
    for (cls, box), ignored in zip(gts, match.gt_ignored):
        if not ignored:
            s = stats.setdefault(cls, ClassStats())
            s.n_gt += 1
            s.gt_heights.append(box[3] - box[1])
    for (cls, _, score), status in zip(preds, match.pred_status):
        if status == 'ignore':
            continue
        s = stats.setdefault(cls, ClassStats())
        s.scores.append(score)
        s.is_tp.append(status == 'tp')
