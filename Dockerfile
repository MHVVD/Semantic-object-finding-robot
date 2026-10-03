# Semantic Object-Finding Robot -- three build targets:
#
#   test   build + unit/lint tests on ros:jazzy-ros-base (what CI runs)
#            docker build --target test -t semantic_nav:test . && docker run --rm semantic_nav:test
#   model  exports YOLO11n to ONNX with Ultralytics (CPU PyTorch, build stage only)
#   demo   the full simulation: Gazebo + SLAM + Nav2 + perception + commander
#            docker compose up demo          (see docker-compose.yml and README)
#
# The YOLO weights are not in the repository (they are Ultralytics', AGPL-3.0);
# the model stage downloads and converts them at build time.

# ---------------------------------------------------------------- test (CI)
FROM ros:jazzy-ros-base AS test

SHELL ["/bin/bash", "-c"]
WORKDIR /ws
COPY src/ src/

# Build and test dependencies only: the simulation stack (Gazebo, TurtleBot4, Nav2)
# is an exec dependency of semantic_nav_bringup and is not needed to build/test.
RUN apt-get update \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths src --ignore-src --rosdistro jazzy -y \
        -t buildtool -t buildtool_export -t build -t build_export -t test \
    && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --event-handlers console_cohesion+

# colcon test-result exits non-zero if any test failed.
CMD ["bash", "-c", "source install/setup.bash && colcon test --event-handlers console_cohesion+ && colcon test-result --verbose"]

# ---------------------------------------------------------------- model export
FROM python:3.12-slim AS model

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics onnx onnxslim
COPY tools/export_yolo.py /export/
RUN cd /export && python export_yolo.py yolo11n.pt --out /models

# ---------------------------------------------------------------- demo (full sim)
FROM osrf/ros:jazzy-desktop-full AS demo

SHELL ["/bin/bash", "-c"]
WORKDIR /ws
COPY src/ src/
COPY --from=model /models/ src/semantic_nav_perception/models/

# Every dependency (Nav2, slam_toolbox, the TurtleBot4 / Create3 simulation, ...);
# OpenVINO from pip, with numpy kept < 2 for the distro's cv_bridge (PROBLEMS_LOG #35).
RUN apt-get update \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths src --ignore-src --rosdistro jazzy -y \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages "openvino==2026.4.0" "numpy<2"

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --event-handlers console_cohesion+

COPY tools/demo.sh /ws/demo.sh
CMD ["bash", "-c", "source install/setup.bash && ./demo.sh"]
