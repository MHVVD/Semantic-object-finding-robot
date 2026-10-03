# Usage by milestone

Detailed commands for every part of the system, in the order it was built. The
[README](../README.md) has the quickstart; the milestone reports explain the why.
All commands assume a sourced workspace (`source install/setup.bash`).

## Simulation (Milestone 1)

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

## Mapping and navigation (Milestone 2)

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

## Detection (Milestone 3)

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

## Projection to 3D (Milestone 4)

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
ros2 run semantic_nav_bringup projection_error.py --sim-truth --raw obs.csv &
ros2 run semantic_nav_bringup detection_tour.py
```

## Semantic map (Milestone 5)

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

## Commander and voice (Milestone 6)

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

## Autonomous exploration (Milestone 7)

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

## Benchmarks (Milestone 8)

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
