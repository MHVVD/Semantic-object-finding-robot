#!/bin/bash
# End-to-end demo (the Docker "demo" target runs this; it also works on a host install):
#   1. explore the unknown house autonomously (SLAM + Nav2 + frontier explorer),
#      building the semantic map; stop when the explorer reports "complete"
#   2. list what the robot found and score it against the world's ground truth
#   3. send the robot to a few objects by name, like `go_to fridge`
#
# Environment:
#   RVIZ=true          show RViz (needs an X display)
#   DEMO_OBJECTS=...   comma-separated object names for step 3 (default: toilet,potted plant,tv)
#   DEMO_OUT=dir       where the CSV / semantic map / scores are written (default: ./demo_output)

set -u
OUT=${DEMO_OUT:-demo_output}
mkdir -p "$OUT"
IFS=',' read -r -a OBJECTS <<< "${DEMO_OBJECTS:-toilet,potted plant,tv}"

cleanup() {
    kill -INT %1 %2 2>/dev/null
    sleep 5
    pkill -9 -f "[g]z sim" 2>/dev/null
}
trap cleanup EXIT

echo "=== 1/3 exploring the unknown house (takes ~15-25 min of wall time)"
ros2 launch semantic_nav_bringup exploration.launch.py rviz:="${RVIZ:-false}" \
    > "$OUT/exploration.log" 2>&1 &
ros2 run semantic_nav_bringup exploration_monitor.py --csv "$OUT/exploration.csv" \
    --save-map "$OUT/semantic_map.yaml" 2>&1 | grep --line-buffered -E "sim .* s \||finished"

echo "=== 2/3 the semantic map vs ground truth"
ros2 run semantic_nav_bringup evaluate_semantic_map.py "$OUT/semantic_map.yaml" \
    | tee "$OUT/scores.txt"

echo "=== 3/3 go to objects by name"
ros2 launch semantic_nav_bringup commander.launch.py > "$OUT/commander.log" 2>&1 &
sleep 10
ros2 run semantic_nav_commander go_to --list
for name in "${OBJECTS[@]}"; do
    echo "--- go_to $name"
    ros2 run semantic_nav_commander go_to "$name"
done
echo "=== done; results in $OUT/"
