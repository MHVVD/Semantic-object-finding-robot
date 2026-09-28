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
Score the detector against simulator ground truth, per class (offline, no ROS graph).

Input is a dataset written by capture_frames with ground-truth labels. The
detector runs with exactly the code the node uses (yolo.YoloDetector) at a low
threshold, twice per frame:

  1. restricted to the evaluated classes exactly like the node's whitelist
     (best class chosen among the whitelisted ones) -> the metrics;
     --filter-after-argmax instead picks the best of all 80 classes and then
     drops non-whitelisted boxes (Ultralytics' `classes=` behaviour)
  2. with all 80 classes -> what missed objects were taken for

From these it derives:

  * per class: GT count, median GT box height, AP50, and TP / FP / FN,
    precision and recall at the operating confidence (--conf)
  * what each missed object was detected as instead ("seen as"), which is
    invisible when the node's whitelist filters other classes away
  * recall by GT box height (a proxy for distance)

Usage:
    ros2 run semantic_nav_perception evaluate_detector --data ~/semantic_nav_data/frames \
        [--model models/yolo11n.onnx --names models/yolo11n.yaml --size 640 480 \
         --conf 0.4 --out DIR]
"""

import argparse
from collections import Counter, defaultdict
import csv
import glob
import os

import cv2
import numpy as np
from semantic_nav_perception.dataset import from_yolo_line
from semantic_nav_perception.evaluation import accumulate, box_iou, match_frame
from semantic_nav_perception.paths import resolve_path
from semantic_nav_perception.yolo import draw, YoloDetector
import yaml

HEIGHT_BINS = [(20, 40), (40, 80), (80, 160), (160, 10000)]
LOW_CONF = 0.01  # keep nearly every box so AP sees the whole precision-recall curve


def load_labels(path, w, h):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [from_yolo_line(line, w, h) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--data', required=True, help='capture_frames output directory')
    parser.add_argument('--model', default='models/yolo11n.onnx')
    parser.add_argument('--names', default=None, help='class-name YAML (default: model .yaml)')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 480], metavar=('W', 'H'))
    parser.add_argument('--conf', type=float, default=0.4, help='operating confidence')
    parser.add_argument('--nms-iou', type=float, default=0.5)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--min-box-px', type=float, default=20.0,
                        help='GT boxes with shorter side below this are ignored')
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--out', help='write metrics.csv and example images here')
    parser.add_argument('--examples', type=int, default=12)
    parser.add_argument('--filter-after-argmax', action='store_true',
                        help='whitelist after choosing the best of all classes')
    parser.add_argument('--classes', nargs='*',
                        help='class names to evaluate (default: every class in the labels)')
    args = parser.parse_args()

    model = resolve_path(args.model)
    names_path = resolve_path(args.names) if args.names else os.path.splitext(model)[0] + '.yaml'
    with open(names_path) as f:
        names = yaml.safe_load(f)['class_names']
    detector = YoloDetector(model, names, tuple(args.size), args.device)

    frames = sorted(glob.glob(os.path.join(os.path.expanduser(args.data), 'images', '*.png')))
    label_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.data), 'labels', '*.txt')))
    if args.classes:
        evaluated = {names.index(n) for n in args.classes}
    else:  # the classes that exist in the world = the node's whitelist
        evaluated = {int(line.split()[0])
                     for p in label_files for line in open(p) if line.strip()}
    stats = {}
    seen_as = defaultdict(Counter)            # GT class -> what covered it instead
    fp_kind = defaultdict(Counter)            # predicted class -> 'background' / GT class
    by_height = defaultdict(lambda: [0, 0])   # bin -> [matched, total]
    examples = []                             # (n_errors, frame, image, preds, gts)
    n_frames = 0
    for path in frames:
        image = cv2.imread(path)
        h, w = image.shape[:2]
        gts = load_labels(path.replace('/images/', '/labels/')[:-4] + '.txt', w, h)
        if gts is None:
            continue
        n_frames += 1
        all_dets = detector(image, LOW_CONF, args.nms_iou)          # all 80 classes
        all_preds = [(d.class_id, (d.x1, d.y1, d.x2, d.y2), d.score) for d in all_dets]
        if args.filter_after_argmax:
            dets = [d for d in all_dets if d.class_id in evaluated]
        else:  # what detector_node does with its class_whitelist
            dets = detector(image, LOW_CONF, args.nms_iou, evaluated)
        preds = [(d.class_id, (d.x1, d.y1, d.x2, d.y2), d.score) for d in dets]
        gts = [g for g in gts if g[0] in evaluated]
        match = match_frame(gts, preds, args.match_iou, args.min_box_px)
        accumulate(stats, gts, preds, match)

        confident = [p for p in preds if p[2] >= args.conf]
        op = match_frame(gts, confident, args.match_iou, args.min_box_px)
        for (cls, box), hit, ign in zip(gts, op.gt_matched, op.gt_ignored):
            if ign:
                continue
            height = box[3] - box[1]
            for lo, hi in HEIGHT_BINS:
                if lo <= height < hi:
                    by_height[(lo, hi)][1] += 1
                    by_height[(lo, hi)][0] += hit
            if not hit:
                others = [(p[2], names[p[0]]) for p in all_preds
                          if p[0] != cls and p[2] >= 0.25 and box_iou(p[1], box) >= 0.5]
                seen_as[cls][max(others)[1] if others else 'nothing'] += 1
        for (cls, box, _), st in zip(confident, op.pred_status):
            if st != 'fp':
                continue
            overlap = [(box_iou(box, g), gc) for gc, g in gts if gc != cls]
            best = max(overlap, default=(0.0, None))
            fp_kind[cls][names[best[1]] if best[0] >= 0.3 else 'background'] += 1
        n_err = sum(st == 'fp' for st in op.pred_status) + sum(
            not m and not i for m, i in zip(op.gt_matched, op.gt_ignored))
        if args.out and n_err:
            examples.append((n_err, os.path.basename(path), image,
                             [d for d in dets if d.score >= args.conf], gts))

    print(f'\n{n_frames} labelled frames, model {os.path.basename(model)} '
          f'{args.size[0]}x{args.size[1]}, conf {args.conf}, match IoU {args.match_iou}, '
          f'GT boxes < {args.min_box_px:.0f} px ignored\n')
    header = (f'{"class":14s} {"GT":>4s} {"med h":>6s} {"AP50":>5s} {"TP":>4s} {"FP":>4s} '
              f'{"FN":>4s} {"prec":>5s} {"recall":>6s}  missed objects seen as')
    print(header + '\n' + '-' * len(header))
    rows, aps = [], []
    for cls in sorted(stats, key=lambda c: names[c]):
        s = stats[cls]
        tp, fp, fn, prec, rec = s.at_threshold(args.conf)
        ap = s.ap()
        if s.n_gt:
            aps.append(ap)
        med_h = float(np.median(s.gt_heights)) if s.gt_heights else float('nan')
        missed = ', '.join(f'{k} {v}' for k, v in seen_as[cls].most_common(3))
        print(f'{names[cls]:14s} {s.n_gt:4d} {med_h:6.0f} {ap:5.2f} {tp:4d} {fp:4d} {fn:4d} '
              f'{prec:5.2f} {rec:6.2f}  {missed}')
        rows.append({'class': names[cls], 'gt': s.n_gt, 'median_gt_height_px': round(med_h),
                     'ap50': round(ap, 3), 'tp': tp, 'fp': fp, 'fn': fn,
                     'precision': round(prec, 3), 'recall': round(rec, 3),
                     'missed_seen_as': missed,
                     'fp_kinds': ', '.join(f'{k} {v}' for k, v in fp_kind[cls].most_common(3))})
    print(f'\nmAP50 over {len(aps)} classes with ground truth: {np.nanmean(aps):.3f}')
    print('\nrecall by GT box height:')
    for (lo, hi), (hit, total) in sorted(by_height.items()):
        label = f'{lo}-{hi} px' if hi < 10000 else f'>= {lo} px'
        print(f'  {label:12s} {hit:4d}/{total:<4d} = {hit / total if total else 0:.2f}')
    print('\nfalse positives at operating threshold (predicted -> what was really there):')
    for cls, kinds in sorted(fp_kind.items(), key=lambda kv: names[kv[0]]):
        if not kinds:
            continue
        print(f'  {names[cls]:14s} ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common(4)))

    if args.out:
        os.makedirs(os.path.join(args.out, 'examples'), exist_ok=True)
        with open(os.path.join(args.out, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        examples.sort(key=lambda e: -e[0])
        for _, frame, image, dets, gts in examples[:args.examples]:
            vis = draw(image, dets)
            for cls, (x1, y1, x2, y2) in gts:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(vis, 'GT ' + names[cls], (int(x1), int(y2) + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.out, 'examples', frame), vis)
        print(f'\nwrote {args.out}/metrics.csv and {min(len(examples), args.examples)} examples')


if __name__ == '__main__':
    main()
