#!/bin/bash
# Cut a GIF from the demo video with a two-pass palette (small and sharp).
#   tools/make_gif.sh demo.mp4 [start_s=14] [duration_s=25] [out=docs/media/demo.gif]
# Tune size with FPS / WIDTH: FPS=10 WIDTH=640 tools/make_gif.sh demo.mp4
set -euo pipefail
IN=$1
START=${2:-14}
DUR=${3:-25}
OUT=${4:-docs/media/demo.gif}
FPS=${FPS:-12}
WIDTH=${WIDTH:-800}
PALETTE=$(mktemp --suffix=.png)
trap 'rm -f "$PALETTE"' EXIT

ffmpeg -loglevel error -ss "$START" -t "$DUR" -i "$IN" \
    -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen=stats_mode=diff" -y "$PALETTE"
ffmpeg -loglevel error -ss "$START" -t "$DUR" -i "$IN" -i "$PALETTE" \
    -lavfi "fps=$FPS,scale=$WIDTH:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    -y "$OUT"
ls -lh "$OUT"
