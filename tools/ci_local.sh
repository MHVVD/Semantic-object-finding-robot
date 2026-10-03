#!/bin/bash
# Run the CI "build-and-test" job (.github/workflows/ci.yml) locally, in the same
# ros:jazzy-ros-base container and with the same commands, on a clean copy of the
# committed tree (git archive HEAD: no build/, no untracked files, no models).
#   tools/ci_local.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
docker run --rm -i -v "$ROOT/.git:/repo.git:ro" ros:jazzy-ros-base bash -s <<'EOF'
set -eo pipefail      # no -u: ROS setup scripts read unset variables
command -v git > /dev/null || (apt-get update -qq && apt-get install -y -qq git > /dev/null)
git config --global --add safe.directory /repo.git
mkdir /ci && cd /ci && git --git-dir=/repo.git archive HEAD | tar -x
apt-get update
rosdep update --rosdistro jazzy
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y \
  -t buildtool -t buildtool_export -t build -t build_export -t test
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --event-handlers console_cohesion+
source install/setup.bash
colcon test --event-handlers console_cohesion+
colcon test-result --verbose
EOF
