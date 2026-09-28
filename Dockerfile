# Reproducible build/test environment for the Semantic Object-Finding Robot.
#   docker build -t semantic_nav .
#   docker run --rm semantic_nav            # runs colcon test
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]
WORKDIR /ws

COPY src/ src/

# Resolve every dependency declared in the package.xml files (build, exec, test).
RUN apt-get update \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths src --ignore-src --rosdistro jazzy -y \
    && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --event-handlers console_cohesion+

# colcon test-result exits non-zero if any test failed.
CMD ["bash", "-c", "source install/setup.bash && colcon test --event-handlers console_cohesion+ && colcon test-result --verbose"]
