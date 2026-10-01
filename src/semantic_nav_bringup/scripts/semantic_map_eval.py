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
Compare a semantic map with config/ground_truth.yaml (no ROS graph needed).

Input, one of:
    --map FILE     a map saved by semantic_map_node's ~/save_map service
    --replay CSV   raw observations recorded by projection_error.py; they are
                   fed frame by frame through the SAME association code the node
                   uses (semantic_nav_mapping.association) with the given
                   parameters -- for tuning without re-driving the robot
    --sweep        with --replay: table over association radius x min_observations

Scoring: map objects are matched one-to-one to same-label ground-truth objects
whose footprint is within --gate m (Hungarian on centre distance). Reports
matched objects, missed objects, false positives (split into duplicates of an
already-matched object and objects where nothing of that class exists) and the
position error of the matches (to the ground-truth centre).
"""

import argparse
import csv
import itertools
import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
from semantic_nav_bringup.eval_utils import distance_to_box, match_map_to_ground_truth
from semantic_nav_mapping.association import Observation, SemanticMap
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_truth(path):
    # Reuse projection_error's loader: same frames and footprint conventions.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'projection_error', os.path.join(HERE, 'projection_error.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_objects(path)[0]


def replay(rows, radius, min_obs, timeout, method, class_radius):
    """Run the node's association on recorded observations, one frame per stamp."""
    smap = SemanticMap(radius, min_obs, timeout, method, class_radius)
    frames = {}
    for r in rows:
        frames.setdefault(float(r['stamp']), []).append(
            Observation(r['label'], (float(r['x']), float(r['y']), float(r['z'])),
                        float(r['confidence'])))
    for stamp in sorted(frames):
        smap.update(frames[stamp], stamp)
    return smap.to_dict()['objects']


def score(objects, truth, gate):
    mapped = [{'label': o['label'], 'xy': tuple(o['position'][:2])} for o in objects]
    matches, fp, fn = match_map_to_ground_truth(mapped, truth, gate)
    matched_truth = {t for _, t, _ in matches}
    duplicates = [i for i in fp if any(
        truth[t]['label'] == mapped[i]['label'] and
        distance_to_box(mapped[i]['xy'], truth[t]['xy'], truth[t]['half'], truth[t]['yaw'])
        <= gate for t in matched_truth)]
    return matches, fp, fn, duplicates


def report(objects, truth, gate):
    matches, fp, fn, dup = score(objects, truth, gate)
    errors = [e for _, _, e in matches]
    print(f'\n{len(objects)} map objects vs {len(truth)} ground-truth objects (gate {gate} m)')
    print(f'  matched         {len(matches):3d}   mean position error {np.mean(errors):.3f} m, '
          f'median {np.median(errors):.3f}, max {np.max(errors):.3f}')
    print(f'  missed          {len(fn):3d}   ' +
          ', '.join(truth[j]['name'] for j in fn))
    print(f'  false positives {len(fp):3d}   ({len(dup)} duplicates of matched objects)')
    print(f'\n{"ground truth":16s} {"map id":>6s} {"obs":>5s} {"conf":>5s} {"error m":>8s}')
    for m, t, e in sorted(matches, key=lambda x: truth[x[1]]['name']):
        o = objects[m]
        print(f'{truth[t]["name"]:16s} {o["id"]:6d} {o["observation_count"]:5d} '
              f'{o["confidence"]:5.2f} {e:8.3f}')
    for i in fp:
        o = objects[i]
        xy = tuple(o['position'][:2])
        near = min(truth, key=lambda t: distance_to_box(xy, t['xy'], t['half'], t['yaw']))
        kind = 'duplicate' if i in dup else 'false'
        print(f'  FP {kind:9s} id {o["id"]:3d} {o["label"]:13s} obs {o["observation_count"]:4d} '
              f'at ({o["position"][0]:.2f}, {o["position"][1]:.2f}) near {near["name"]}')


def main():
    pkg = get_package_share_directory('semantic_nav_bringup')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--ground-truth',
                        default=os.path.join(pkg, 'config', 'ground_truth.yaml'))
    parser.add_argument('--map', help='semantic map YAML saved by the node')
    parser.add_argument('--replay', help='raw observations CSV from projection_error.py')
    parser.add_argument('--gate', type=float, default=0.5)
    parser.add_argument('--radius', type=float, default=0.75)
    parser.add_argument('--min-observations', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=10.0)
    parser.add_argument('--method', default='greedy', choices=['greedy', 'hungarian'])
    parser.add_argument('--class-radius', nargs='*', default=[],
                        help='per-class radius overrides, e.g. bed=1.5 couch=1.2')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    truth = load_truth(args.ground_truth)
    class_radius = {k: float(v) for k, v in (s.rsplit('=', 1) for s in args.class_radius)}

    if args.map:
        with open(args.map) as f:
            report(yaml.safe_load(f)['objects'], truth, args.gate)
        return
    if not args.replay:
        parser.error('give --map FILE or --replay CSV')
    with open(args.replay) as f:
        rows = list(csv.DictReader(f))
    if args.sweep:
        print(f'{"radius":>6s} {"N":>3s} {"objects":>7s} {"matched":>7s} {"missed":>6s} '
              f'{"FP":>3s} {"dup":>4s} {"mean err":>8s}')
        for radius, n in itertools.product((0.4, 0.6, 0.75, 1.0, 1.25),
                                           (1, 2, 3, 5, 10, 20)):
            objs = replay(rows, radius, n, args.timeout, args.method, class_radius)
            matches, fp, fn, dup = score(objs, truth, args.gate)
            err = np.mean([e for _, _, e in matches]) if matches else float('nan')
            print(f'{radius:6.2f} {n:3d} {len(objs):7d} {len(matches):7d} {len(fn):6d} '
                  f'{len(fp):3d} {len(dup):4d} {err:8.3f}')
        return
    report(replay(rows, args.radius, args.min_observations, args.timeout, args.method,
                  class_radius), truth, args.gate)


if __name__ == '__main__':
    main()
