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
Measure detector latency per model / input size / thread count on this CPU (no ROS graph).

Cycles through real frames (a capture_frames dataset) so post-processing sees a
realistic number of boxes. After `--warmup` untimed calls it times `--n` calls
and reports median and 90th-percentile latency of each stage, plus the frame
rate that latency allows (1000 / median total). Run it once with the simulator
stopped (the CPU to itself) and once with it running (what the robot gets).

Usage:
    ros2 run semantic_nav_perception benchmark_detector \
        --images ~/semantic_nav_data/frames/images \
        --models models/yolov8n.onnx models/yolo11n.onnx --sizes 640x640 640x480 416x320
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
from semantic_nav_perception.paths import resolve_path
from semantic_nav_perception.yolo import YoloDetector
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--images', required=True, help='directory of .png/.jpg frames')
    parser.add_argument('--models', nargs='+', default=['models/yolo11n.onnx'])
    parser.add_argument('--sizes', nargs='+', default=['640x480'], help='WxH, multiples of 32')
    parser.add_argument('--threads', type=int, nargs='+', default=[0])
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--csv', help='append results to this CSV file')
    parser.add_argument('--tag', default='', help='free-text column, e.g. "sim running"')
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.images), '*.png')) +
                   glob.glob(os.path.join(os.path.expanduser(args.images), '*.jpg')))
    if not paths:
        raise SystemExit(f'no images in {args.images}')
    images = [cv2.imread(p) for p in paths[:50]]

    rows = []
    print(f'{"model":14s} {"size":>8s} {"thr":>3s} {"pre":>5s} {"infer":>6s} {"p90":>6s} '
          f'{"post":>5s} {"total":>6s} {"FPS":>5s}')
    for model in args.models:
        path = resolve_path(model)
        with open(os.path.splitext(path)[0] + '.yaml') as f:
            names = yaml.safe_load(f)['class_names']
        for size in args.sizes:
            w, h = (int(v) for v in size.lower().split('x'))
            for threads in args.threads:
                det = YoloDetector(path, names, (w, h), args.device, threads)
                for i in range(args.warmup):
                    det(images[i % len(images)], args.conf, 0.5)
                timings = []
                for i in range(args.n):
                    det(images[i % len(images)], args.conf, 0.5)
                    timings.append(det.timing)
                med = {k: float(np.median([t[k] for t in timings])) for k in timings[0]}
                p90 = float(np.percentile([t['infer'] for t in timings], 90))
                row = {'model': os.path.basename(path), 'size': f'{w}x{h}', 'threads': threads,
                       'pre_ms': round(med['pre'], 1), 'infer_ms': round(med['infer'], 1),
                       'infer_p90_ms': round(p90, 1), 'post_ms': round(med['post'], 1),
                       'total_ms': round(med['total'], 1),
                       'fps': round(1000.0 / med['total'], 1), 'tag': args.tag}
                rows.append(row)
                print(f'{row["model"]:14s} {row["size"]:>8s} {threads:3d} {row["pre_ms"]:5.1f} '
                      f'{row["infer_ms"]:6.1f} {p90:6.1f} {row["post_ms"]:5.1f} '
                      f'{row["total_ms"]:6.1f} {row["fps"]:5.1f}', flush=True)
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
