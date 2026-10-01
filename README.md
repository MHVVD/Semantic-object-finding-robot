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

### Commander and voice (Milestone 6)

`commander_node` serves `/semantic_nav/go_to` (`GoTo`): it looks the object up
in the semantic map, generates standoff goals around each instance (rejecting
lethal / inscribed / unknown costmap cells and points outside the object's
local free space), picks the instance with the shortest *planned* path, sends
`NavigateToPose` and answers when the robot has arrived. `/semantic_nav/cancel`
stops it. Aliases: fridge, sofa, television, plant, table, loo.

Voice: `voice_command` listens on the default microphone (`arecord`) and runs
Vosk offline speech recognition restricted to commands such as "go to the
fridge", "take me to the sofa", "find the toilet", "stop".

```bash
# one-time: offline speech recognition (numpy pinned as for OpenVINO)
pip install --user --break-system-packages -c /tmp/np.txt vosk
mkdir -p ~/semantic_nav_data/models && cd ~/semantic_nav_data/models
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && unzip vosk-model-small-en-us-0.15.zip

ros2 launch semantic_nav_bringup navigation.launch.py
ros2 launch semantic_nav_bringup perception.launch.py
ros2 launch semantic_nav_bringup commander.launch.py voice:=true   # then just say it
ros2 run semantic_nav_commander go_to fridge                       # or type it
ros2 run semantic_nav_commander go_to --list
ros2 run semantic_nav_commander go_to --cancel
```

### Autonomous exploration (Milestone 7)

One launch, no teleop: the robot starts in an unknown house, builds the map
with slam_toolbox while Nav2 drives it, and builds the semantic map on the way.
`frontier_explorer.py` (ours, default) explores in two phases: map frontiers,
then camera coverage (drive to viewpoints facing surfaces the camera has not
seen yet), and returns to the start. `explorer:=explore_lite` uses
m-explore-ros2 instead (build it from source in a separate overlay; see the
M7 report).

```bash
ros2 launch semantic_nav_bringup exploration.launch.py              # explores on its own
ros2 run semantic_nav_bringup exploration_monitor.py --csv run.csv --save-map run.yaml
```

Measured: complete after ~290 s sim (13.5 min wall at RTF ~0.36), 91.5 % of
the reference map known, 38.6 m driven, 13 / 18 objects in the semantic map.

### Benchmarks (Milestone 8)

Three unattended trials, each: explore the unknown house and save the semantic
map, then restart with the saved occupancy map + AMCL, load that semantic map and
run 20 GoTo commands over random labels. Ground truth comes from the world file
(`config/ground_truth.yaml`) and Gazebo's true robot pose, never from the robot's
own estimates.

```bash
tools/run_benchmarks.sh 1 3            # ~1 h 45 min wall; raw data -> results/raw/
python3 tools/make_results.py          # tables (results/RESULTS.md) + plots
ros2 run semantic_nav_bringup evaluate_semantic_map.py results/raw/semantic_map_run*.yaml
```

Or in Docker:

```bash
docker build -t semantic_nav .
docker run --rm semantic_nav
```

## Results

Three full runs (explore → semantic map → 20 GoTo commands), mean ± sd over runs
(sample variance in [results/RESULTS.md](results/RESULTS.md), which also lists
every missed object and every false positive).

**Semantic map vs ground truth** (18 objects; a map object counts if it has the
right class and lies within 0.5 m of the real object's footprint, one-to-one):

| | precision | recall | mean error | median error | false positives |
|---|---|---|---|---|---|
| run 1 | 0.62 | 0.56 | 0.262 m | 0.215 m | 6 |
| run 2 | 0.59 | 0.72 | 0.275 m | 0.188 m | 9 |
| run 3 | 0.68 | 0.72 | 0.272 m | 0.160 m | 6 |
| **mean ± sd** | **0.63 ± 0.05** | **0.67 ± 0.10** | **0.270 ± 0.007 m** | **0.187 ± 0.028 m** | **7.0 ± 1.7** |

| class | real | precision | recall | mean error |
|---|---|---|---|---|
| potted plant | 3 | 1.00 ± 0.00 | 0.89 ± 0.19 | 0.105 m |
| toilet | 1 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.229 m |
| sink | 2 | 1.00 ± 0.00 | 0.50 ± 0.00 | 0.210 m |
| chair | 6 | 0.87 ± 0.23 | 0.50 ± 0.00 | 0.121 m |
| tv | 2 | 0.58 ± 0.15 | 1.00 ± 0.00 | 0.223 m |
| couch | 1 | 0.67 ± 0.47 | 0.67 ± 0.58 | 0.812 m * |
| refrigerator | 1 | 0.33 ± 0.29 | 0.67 ± 0.58 | 0.334 m |
| bed | 1 | 0.39 ± 0.10 | 1.00 ± 0.00 | 0.949 m * |
| dining table | 1 | 0.00 ± 0.00 | 0.00 ± 0.00 | – |

\* to the object's centre: the camera sees the near face of a 2 m bed / sofa.

![Semantic map vs ground truth, per run](results/semantic_map_vs_truth.png)
![Precision and recall per class](results/semantic_map_precision_recall.png)

**Navigation: 20 GoTo commands per run.** Success = the commander reported
arrival *and* the robot's true final position is within 1 m of a real object of
that class:

| | success | commander said "arrived" | time-to-goal | final distance to object |
|---|---|---|---|---|
| run 1 | 80 % | 95 % | 24.7 s | 0.56 m |
| run 2 | 55 % | 100 % | 45.0 s | 0.69 m |
| run 3 | 80 % | 85 % | 25.2 s | 0.65 m |
| **mean ± sd** | **72 ± 14 %** | **93 ± 8 %** | **31.6 ± 11.6 s** (sim) | **0.63 ± 0.07 m** |

Pooled: 43 / 60 commands succeeded (95 % Wilson interval 59–82 %). Every
failure was a perception failure: 13 drove to a false positive ("wrong place"),
4 asked for a class missing from the map; Nav2 itself never failed or timed out.
With perfect choice among the instances already in the map, 52 / 60 (87 %) would
have succeeded.

![Navigation outcomes and success per label](results/navigation_outcomes.png)
![Time vs distance and final distance per label](results/navigation_time_distance.png)

**Exploration:** 354 ± 12 s sim (917 ± 46 s wall), 89.4 ± 2.9 % map coverage,
41 ± 5 m driven.

![Exploration progress](results/exploration_progress.png)

## Status

- [x] M0 — workspace, interfaces, CI
- [x] M1 — simulation: house world, TurtleBot4, bridge, RViz
- [x] M2 — SLAM map + Nav2/AMCL navigation
- [x] M3 — detection: YOLO11n → ONNX → OpenVINO, evaluated per class
- [x] M4 — projection: depth + intrinsics + TF → map-frame observations
- [x] M5 — semantic map: association, confirmation, services, save/load
- [x] M6 — commander: GoTo service, goal generation, Nav2, CLI and voice
- [x] M7 — autonomous exploration: frontiers + camera coverage, semantic map without teleop
- [x] M8 — evaluation: semantic map precision/recall, GoTo benchmark, 3 runs, failure analysis

Problems encountered and their fixes are logged in [docs/PROBLEMS_LOG.md](docs/PROBLEMS_LOG.md).
