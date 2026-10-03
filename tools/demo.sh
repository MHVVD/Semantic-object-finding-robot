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
#   DEMO_OUT=dir       where the logs / CSV / semantic map / scores go (default: ./demo_output)
#   HOST_UID/HOST_GID  owner to give DEMO_OUT at the end (set by docker-compose.yml)

set -u
OUT=${DEMO_OUT:-demo_output}
mkdir -p "$OUT"
IFS=',' read -r -a OBJECTS <<< "${DEMO_OBJECTS:-toilet,potted plant,tv}"
LAUNCH='' COMMANDER=''

cleanup() {
    for pid in $COMMANDER $LAUNCH; do kill -INT "$pid" 2>/dev/null; done
    sleep 5
    pkill -9 -f "[g]z sim" 2>/dev/null
    # In Docker this runs as root: hand the results back to the host user.
    [ -n "${HOST_UID:-}" ] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUT"
}
trap cleanup EXIT

echo "=== 1/3 exploring the unknown house (takes ~15-30 min of wall time)"
ros2 launch semantic_nav_bringup exploration.launch.py rviz:="${RVIZ:-false}" \
    > "$OUT/exploration.log" 2>&1 &
LAUNCH=$!
( ros2 run semantic_nav_bringup exploration_monitor.py --csv "$OUT/exploration.csv" \
      --save-map "$OUT/semantic_map.yaml" 2>&1 \
      | grep --line-buffered -E "sim .* s \||finished" ) &
MONITOR=$!
# Whichever ends first: the monitor (explorer reported "complete") or the stack (a crash).
wait -n "$LAUNCH" "$MONITOR"
if kill -0 "$MONITOR" 2>/dev/null; then
    echo "!!! the simulation stack exited before exploration finished; first errors:"
    grep -m 10 "ERROR\|\[Err\]" "$OUT/exploration.log"
    echo "!!! full log: $OUT/exploration.log"
    kill "$MONITOR" 2>/dev/null
    exit 1
fi

echo "=== 2/3 the semantic map vs ground truth"
ros2 run semantic_nav_bringup evaluate_semantic_map.py "$OUT/semantic_map.yaml" \
    | tee "$OUT/scores.txt"

echo "=== 3/3 go to objects by name"
ros2 launch semantic_nav_bringup commander.launch.py > "$OUT/commander.log" 2>&1 &
COMMANDER=$!
sleep 10
ros2 run semantic_nav_commander go_to --list
for name in "${OBJECTS[@]}"; do
    echo "--- go_to $name"
    ros2 run semantic_nav_commander go_to "$name"
done
echo "=== done; results in $OUT/"
