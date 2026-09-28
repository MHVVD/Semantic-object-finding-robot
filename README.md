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

Or in Docker:

```bash
docker build -t semantic_nav .
docker run --rm semantic_nav
```

## Status

- [x] M0 — workspace, interfaces, CI
- [ ] Simulation + SLAM
- [ ] Detection (YOLO → ONNX → OpenVINO)
- [ ] Deprojection into the map frame
- [ ] Semantic map (data association)
- [ ] Commander + Nav2

Problems encountered and their fixes are logged in [docs/PROBLEMS_LOG.md](docs/PROBLEMS_LOG.md).
