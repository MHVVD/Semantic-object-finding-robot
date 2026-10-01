#!/bin/bash
# Run the M8 benchmark trials unattended: for each trial,
#   1. explore the unknown house (exploration.launch.py) and save the semantic map
#   2. restart in "deployment" mode (saved occupancy map + AMCL, the semantic map
#      from step 1 loaded, detector off) and run 20 GoTo commands
# Raw results go to results/raw/; then run tools/make_results.py.
#
# Usage:  tools/run_benchmarks.sh [first_trial] [last_trial]     (default 1 3)
# Needs: the workspace built and sourced; nothing else running.

set -u
FIRST=${1:-1}
LAST=${2:-3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RAW=$ROOT/results/raw
LOGS=$RAW/logs
mkdir -p "$LOGS"

stop_all() {
    # Patterns match the launched processes but never this script's own command line.
    pkill -INT -f "[r]os2 launch semantic_nav_bringup"
    sleep 5
    pkill -9 -f "[g]z sim"
    pkill -f "[/]opt/ros/jazzy/lib/|[s]emantic_ws/install/"
    sleep 3
    pkill -9 -f "[/]opt/ros/jazzy/lib/|[s]emantic_ws/install/"
    sleep 2
}

for i in $(seq "$FIRST" "$LAST"); do
    echo "=== trial $i: exploration ($(date +%T))"
    ros2 launch semantic_nav_bringup exploration.launch.py rviz:=false \
        > "$LOGS/explore_run$i.log" 2>&1 &
    ros2 run semantic_nav_bringup exploration_monitor.py --duration 3000 \
        --csv "$RAW/explore_run$i.csv" --save-map "$RAW/semantic_map_run$i.yaml" \
        > "$LOGS/monitor_run$i.log" 2>&1
    tail -n 2 "$LOGS/monitor_run$i.log"
    stop_all

    echo "=== trial $i: navigation benchmark ($(date +%T))"
    ros2 launch semantic_nav_bringup navigation.launch.py rviz:=false \
        > "$LOGS/nav_run$i.log" 2>&1 &
    ros2 launch semantic_nav_bringup perception.launch.py detector:=false projector:=false \
        > "$LOGS/semantic_map_run$i.log" 2>&1 &
    ros2 launch semantic_nav_bringup commander.launch.py \
        > "$LOGS/commander_run$i.log" 2>&1 &
    sleep 20
    ros2 run semantic_nav_bringup nav_benchmark.py --run "run$i" --seed "$i" --n 20 \
        --load-map "$RAW/semantic_map_run$i.yaml" --csv "$RAW/nav_run$i.csv" \
        > "$LOGS/benchmark_run$i.log" 2>&1
    grep -c "reached" "$RAW/nav_run$i.csv"
    stop_all
done
echo "=== done ($(date +%T))"
