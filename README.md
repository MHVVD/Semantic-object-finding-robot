# Semantic Object-Finding Robot

A TurtleBot 4 in Gazebo Harmonic maps a house with slam_toolbox, detects objects with a
YOLO nano model on its RGB-D camera, projects each detection into the map frame to build a
**semantic map**, and then navigates to objects on command ("go to the fridge") with Nav2.

ROS 2 Jazzy · Gazebo Harmonic · Nav2 · slam_toolbox · YOLO (ONNX + OpenVINO, CPU)

## Architecture

```
 Gazebo (gz sim -s) ──RGB──> detector_node ──2D boxes──> projector_node ──map-frame obs──> semantic_map_node
        │           ──depth + CameraInfo───────────────────────^                                   │
        │           ──scan──> slam_toolbox ──/map, map->odom TF──> Nav2                             │ ListObjects
        │                                                         ^                                 v
        └───────────────────────────────────── NavigateToPose ────┴──────────── commander_node <── GoTo("fridge")
```

## Packages

| Package | Type | Contents |
|---|---|---|
| `semantic_nav_interfaces` | ament_cmake | `SemanticObject`, `SemanticObjectArray` msgs; `GoTo`, `ListObjects` srvs |
| `semantic_nav_perception` | ament_python | `detector_node`, `projector_node` |
| `semantic_nav_mapping` | ament_python | `semantic_map_node` |
| `semantic_nav_commander` | ament_python | `commander_node` |
| `semantic_nav_bringup` | ament_cmake | launch files, `config/params.yaml`, worlds, RViz configs |

## Build and test

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
colcon test && colcon test-result --verbose
```

Run the node skeletons:

```bash
source install/setup.bash
ros2 launch semantic_nav_bringup semantic_nav.launch.py
```

### Simulation (Milestone 1)

```bash
# headless Gazebo server + RViz (recommended on a laptop)
ros2 launch semantic_nav_bringup sim.launch.py
# with the Gazebo GUI instead
ros2 launch semantic_nav_bringup sim.launch.py headless:=false rviz:=false
# drive (TurtleBot4 on Jazzy expects TwistStamped)
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
# real-time factor and sensor rates
ros2 run semantic_nav_bringup measure_sim_performance.py --duration 30
```

The first run downloads the furniture models from Gazebo Fuel (needs internet, ~70 MB,
cached in `~/.gz/fuel`). Ground truth for evaluation is in
`src/semantic_nav_bringup/config/ground_truth.yaml`.

### Mapping and navigation (Milestone 2)

```bash
# SLAM (slam_toolbox online async) + sim; drive with teleop or the scripted route
ros2 launch semantic_nav_bringup slam.launch.py
ros2 run semantic_nav_bringup mapping_drive.py
ros2 run nav2_map_server map_saver_cli -f src/semantic_nav_bringup/maps/house --ros-args -p use_sim_time:=true
# Nav2 + AMCL on the saved map, then the 5-goal benchmark
ros2 launch semantic_nav_bringup navigation.launch.py
ros2 run semantic_nav_bringup nav_goal_test.py --sim-truth
```

Latest benchmark: 5/5 goals across rooms, 0 recoveries, 214 s sim time,
final error <= 0.15 m, AMCL error vs ground truth 0.07 m mean.

### Detection (Milestone 3)

YOLO11n (Ultralytics) exported to ONNX and run with OpenVINO on the CPU.
OpenVINO has no rosdep key, so install it once for the system Python, pinning
NumPy so cv_bridge keeps working:

```bash
echo "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" > /tmp/np.txt
pip install --user --break-system-packages -c /tmp/np.txt openvino
# export the models (throwaway venv with PyTorch; writes src/semantic_nav_perception/models/)
python3 -m venv .venv-export
.venv-export/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/pip install ultralytics onnx onnxslim
.venv-export/bin/python tools/export_yolo.py
colcon build --symlink-install
```

```bash
# detector on a running sim (debug image in the RViz "Detections" panel)
ros2 launch semantic_nav_bringup navigation.launch.py
ros2 launch semantic_nav_bringup perception.launch.py
# dataset: sim with ground-truth boxes + recorder + a Nav2 camera tour
ros2 launch semantic_nav_bringup navigation.launch.py gt_boxes:=true
ros2 launch semantic_nav_bringup perception.launch.py capture:=true
ros2 run semantic_nav_bringup detection_tour.py
# offline: per-class quality and latency
ros2 run semantic_nav_perception evaluate_detector --data ~/semantic_nav_data/frames
ros2 run semantic_nav_perception benchmark_detector --images ~/semantic_nav_data/frames/images \
    --models models/yolo11n.onnx models/yolov8n.onnx --sizes 640x480 416x320
```

On the i5-8265U: 27-37 ms inference at 640x480 (CPU idle), ~50-70 ms with the
simulation running; mAP50 0.53 on 212 simulator frames. Works well for tv,
toilet and potted plant, partially for bed, couch and chair, and poorly for
dining table, sink and refrigerator (camera 0.24 m above the floor). Fine-tuning
notebook for Colab: [tools/finetune_yolo_colab.ipynb](tools/finetune_yolo_colab.ipynb).

### Projection to 3D (Milestone 4)

`projector_node` pairs each detection message with the depth image and camera
info of the same timestamp (`message_filters.ApproximateTimeSynchronizer`),
takes the median valid depth inside the central half of each box, deprojects
the box centre with the pinhole model and transforms it into `map` with tf2 at
the image timestamp. Output: `/semantic_nav/observations`
(`SemanticObjectArray`, one entry per detection, not yet merged) and
`/semantic_nav/observation_markers` ("Observations" in `rviz/nav.rviz`).
`perception.launch.py` starts it together with the detector.

```bash
ros2 launch semantic_nav_bringup navigation.launch.py
ros2 launch semantic_nav_bringup perception.launch.py
# error against config/ground_truth.yaml while touring the house
ros2 run semantic_nav_bringup projection_error.py --sim-truth --csv obs.csv &
ros2 run semantic_nav_bringup detection_tour.py
```

### Semantic map (Milestone 5)

`semantic_map_node` fuses the per-frame observations into persistent objects:
same-label nearest neighbour within 0.75 m (2.0 m for beds, 1.5 m for couches),
one-to-one per frame, running-mean position, confirmed after 10 sightings,
unconfirmed objects forgotten after 10 s unseen. It publishes
`/semantic_nav/semantic_map` and `/semantic_nav/markers` ("Semantic map" in
`rviz/nav.rviz`) and serves `/semantic_nav/list_objects`,
`/semantic_nav/save_map` and `/semantic_nav/load_map`.

```bash
ros2 launch semantic_nav_bringup navigation.launch.py
ros2 launch semantic_nav_bringup perception.launch.py      # detector + projector + map
ros2 run semantic_nav_bringup detection_tour.py
ros2 service call /semantic_nav/list_objects semantic_nav_interfaces/srv/ListObjects "{label_filter: chair}"
ros2 service call /semantic_nav/save_map semantic_nav_interfaces/srv/MapFile "{path: ''}"
ros2 run semantic_nav_bringup semantic_map_eval.py --map ~/semantic_nav_data/semantic_map.yaml
```

Or in Docker:

```bash
docker build -t semantic_nav .
docker run --rm semantic_nav
```

## Status

- [x] M0 — workspace, interfaces, CI
- [x] M1 — simulation: house world, TurtleBot4, bridge, RViz
- [x] M2 — SLAM map + Nav2/AMCL navigation
- [x] M3 — detection: YOLO11n → ONNX → OpenVINO, evaluated per class
- [x] M4 — projection: depth + intrinsics + TF → map-frame observations
- [x] M5 — semantic map: association, confirmation, services, save/load
- [ ] Commander + Nav2

Problems encountered and their fixes are logged in [docs/PROBLEMS_LOG.md](docs/PROBLEMS_LOG.md).
