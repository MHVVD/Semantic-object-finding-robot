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
Export Ultralytics YOLO nano checkpoints to ONNX for the detector node.

Runs OUTSIDE ROS, in a throwaway venv (ultralytics pulls in PyTorch):
    python3 -m venv .venv-export
    .venv-export/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    .venv-export/bin/pip install ultralytics onnx onnxslim
    .venv-export/bin/python tools/export_yolo.py            # yolov8n + yolo11n

The ONNX graphs are exported with a dynamic input shape so the node's
`input_size` parameter can pick the resolution at load time (OpenVINO reshapes
the graph once, then compiles it static). Next to every <name>.onnx a
<name>.yaml is written with the class names in index order; the node reads it
so label strings never live in code.

Outputs land in src/semantic_nav_perception/models/ (gitignored, installed by
setup.py). Weights (.pt) are downloaded by Ultralytics on first use.
"""

import argparse
import os
import shutil

from ultralytics import YOLO
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, '..', 'src', 'semantic_nav_perception', 'models')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('weights', nargs='*', default=['yolov8n.pt', 'yolo11n.pt'],
                        help='Ultralytics checkpoints (name or path)')
    parser.add_argument('--out', default=DEFAULT_OUT, help='output directory')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for weights in args.weights:
        model = YOLO(weights)
        onnx_path = model.export(format='onnx', dynamic=True, simplify=True,
                                 opset=args.opset, imgsz=640)
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        dst = os.path.join(args.out, stem + '.onnx')
        shutil.move(onnx_path, dst)
        names = [model.names[i] for i in sorted(model.names)]
        with open(os.path.join(args.out, stem + '.yaml'), 'w') as f:
            yaml.safe_dump({'source_weights': weights, 'class_names': names}, f,
                           sort_keys=False)
        print(f'{weights} -> {dst} ({len(names)} classes)')


if __name__ == '__main__':
    main()
