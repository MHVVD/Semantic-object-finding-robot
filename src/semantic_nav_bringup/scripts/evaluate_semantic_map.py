#!/usr/bin/env python3
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
Score saved semantic maps against config/ground_truth.yaml: precision, recall, error per class.

For every map file (saved by semantic_map_node's ~/save_map service) it
matches map objects one-to-one to ground-truth objects (same label, map point
within --gate m of the object's footprint, Hungarian on centre distance; see
eval_utils.match_map_to_ground_truth) and prints, per class and overall:
    truth / mapped counts, TP, FP, FN, precision, recall,
    mean and median position error of the matches (to the object's centre)
With several maps (repeated runs) it adds a summary: per class the mean and
sample variance over runs of precision, recall and mean error, plus the
pooled counts. --csv writes one row per (run, class) for tools/make_results.py.

No ROS graph needed:
    ros2 run semantic_nav_bringup evaluate_semantic_map.py run1.yaml run2.yaml run3.yaml
"""

import argparse
import csv
import os

from ament_index_python.packages import get_package_share_directory
from semantic_nav_bringup.benchmark_stats import mean_var, per_class_metrics
from semantic_nav_bringup.eval_utils import match_map_to_ground_truth
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['run', 'label', 'truth', 'mapped', 'tp', 'fp', 'fn', 'precision', 'recall',
          'mean_error', 'median_error']


def load_truth(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'projection_error', os.path.join(HERE, 'projection_error.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_objects(path)[0]


def evaluate(map_file, truth, gate):
    with open(map_file) as f:
        objects = yaml.safe_load(f)['objects']
    mapped = [{'label': o['label'], 'xy': tuple(o['position'][:2])} for o in objects]
    matches, _, _ = match_map_to_ground_truth(mapped, truth, gate)
    return per_class_metrics(mapped, truth, matches)


def fmt(value, spec='.2f'):
    return '-' if value is None else format(value, spec)


def print_run(name, rows):
    print(f'\n{name}')
    print(f'  {"class":13s} {"truth":>5s} {"map":>4s} {"TP":>3s} {"FP":>3s} {"FN":>3s} '
          f'{"prec":>5s} {"recall":>6s} {"mean err":>8s} {"median":>6s}')
    for label, r in rows.items():
        print(f'  {label:13s} {r["truth"]:5d} {r["mapped"]:4d} {r["tp"]:3d} {r["fp"]:3d} '
              f'{r["fn"]:3d} {fmt(r["precision"]):>5s} {fmt(r["recall"]):>6s} '
              f'{fmt(r["mean_error"], ".3f"):>8s} {fmt(r["median_error"], ".3f"):>6s}')


def mv(values, spec='.2f'):
    mean, var, n = mean_var(values)
    if mean is None:
        return '-'
    return f'{mean:{spec}}' + ('' if var is None else f' (var {var:{spec}})')


def print_summary(runs):
    labels = list(next(iter(runs.values())))
    for rows in runs.values():
        labels += [lbl for lbl in rows if lbl not in labels]
    print(f'\nSummary over {len(runs)} runs: mean (sample variance); counts pooled')
    print(f'  {"class":13s} {"TP":>3s} {"FP":>3s} {"FN":>3s} {"precision":>18s} '
          f'{"recall":>18s} {"mean err m":>20s}')
    for label in labels:
        rs = [rows[label] for rows in runs.values() if label in rows]
        print(f'  {label:13s} {sum(r["tp"] for r in rs):3d} {sum(r["fp"] for r in rs):3d} '
              f'{sum(r["fn"] for r in rs):3d} {mv([r["precision"] for r in rs]):>18s} '
              f'{mv([r["recall"] for r in rs]):>18s} '
              f'{mv([r["mean_error"] for r in rs], ".3f"):>20s}')


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('maps', nargs='+', help='semantic map YAML files (one per run)')
    parser.add_argument('--names', nargs='*', help='run names (default: file names)')
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    parser.add_argument('--gate', type=float, default=0.5)
    parser.add_argument('--csv', help='write per-(run, class) rows here')
    args = parser.parse_args()
    names = args.names or [os.path.splitext(os.path.basename(m))[0] for m in args.maps]
    if len(names) != len(args.maps):
        parser.error('--names needs one name per map')
    truth = load_truth(args.ground_truth)

    runs = {name: evaluate(m, truth, args.gate) for name, m in zip(names, args.maps)}
    for name, rows in runs.items():
        print_run(name, rows)
    if len(runs) > 1:
        print_summary(runs)
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for name, rows in runs.items():
                for label, r in rows.items():
                    writer.writerow({'run': name, 'label': label,
                                     **{k: r[k] for k in FIELDS[2:]}})


if __name__ == '__main__':
    main()
