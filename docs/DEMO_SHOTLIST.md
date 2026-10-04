# Demo video (90 s) and GIF: script, shot list and ffmpeg commands

> The committed video (`docs/media/demo.mp4`, GIF in the README) was recorded from a real
> autonomous run on a virtual display, with `rviz/demo.rviz` as the RViz layout. The plan
> below is for recording your own version, for example with the Gazebo GUI.

**Story in one line:** *the robot starts knowing nothing, explores and learns where
things are, then goes to the toilet when asked, and here is how well it works.*

The exploration takes ~15 min of wall time, so record raw clips in real time and
speed them up afterwards; the shot list gives the final on-screen durations.

## 0. Setup (once, before recording)

- **Screen.** Close everything else and set one large RViz window. 1366×768 is
  fine; record the full screen. Hide the dock (GNOME: Settings → Ubuntu Desktop → Dock → auto-hide).
- **RViz** (`rviz/nav.rviz`). In the Displays panel turn **on**:
  - Map
  - LaserScan
  - RobotModel
  - Global Planner → Path
  - Observations
  - Semantic map
  - Exploration goal
  - the **Detections** image panel (dock it bottom-right, about 30 % of the width)

  Turn **off**: TF, Amcl Particle Swarm, Local Costmap, VoxelGrid, Bumper Hit.
  Use View → *TopDownOrtho*, zoomed so the whole house fills the view.
- **Gazebo GUI** for the beauty shots only (heavy on the laptop). Start a second
  client against the running headless server: `gz sim -g`. Orbit with the mouse,
  and follow the robot: right-click the turtlebot4 → *Follow*.
- **Terminal:** a large font (Ctrl + several times), dark theme, 2 tabs prepared.
- **Fresh state:** delete `~/.ros/log` noise; `tools/demo.sh` writes its outputs into
  `demo_output/`.

## 1. Raw clips to record

Use `ffmpeg` (below), OBS or SimpleScreenRecorder.

| clip | what | how |
|---|---|---|
| A | terminal: type the one command | `ros2 launch semantic_nav_bringup exploration.launch.py headless:=false` (or `docker compose up demo`) |
| B | Gazebo GUI: slow orbit of the house, then the robot | 20 s real time, before the robot moves much |
| C | RViz: the **whole exploration**, top-down | 15–20 min real time, one continuous recording |
| D | RViz close-up: the Detections panel + markers popping up as the robot passes the living room | 30 s real time, during clip C (zoom in RViz for a second recording, or crop C) |
| E | terminal: `ros2 run semantic_nav_commander go_to --list` | after "complete" |
| F | the command: `ros2 run semantic_nav_commander go_to toilet` (or say "go to the toilet" with `voice:=true`), and RViz while the robot drives there | ~40 s real time, from command to "arrived" |
| G | Gazebo GUI: the robot arriving, facing the toilet | 10 s real time |
| H | title and results cards | images: `results/semantic_map_vs_truth.png` and a title card (below) |

## 2. Shot list (final cut, 90 s)

| t (s) | shot | on screen | caption (drawtext / editor) |
|---|---|---|---|
| 0–4 | H (title card) | Project name + one-line pitch | "Semantic Object-Finding Robot · ROS 2 Jazzy" |
| 4–8 | A | the single launch command | "One command. Unknown house. No teleop." |
| 8–14 | B | Gazebo orbit: the house, then the robot | "TurtleBot 4 · lidar + RGB-D camera" |
| 14–40 | C at 40× | the map grows from nothing; the orange goal arrow jumps between frontiers; semantic-map labels appear | "SLAM + frontier exploration" → (at ~30 s) "camera coverage: look at every surface" |
| 40–50 | D at 1–2× | YOLO boxes in the Detections panel; matching sphere + label appears in the map | "YOLO11n on CPU (OpenVINO) → depth → map frame" |
| 50–56 | C end + E | the "complete" status; `go_to --list` prints the objects found | "Semantic map: 12 / 18 objects, ~0.2 m error" |
| 56–74 | F at 2–3× | the command or the voice; the planned path appears; the robot drives through the doorway | "\"Go to the toilet\" → nearest reachable instance → Nav2" |
| 74–80 | G | robot stops, facing the toilet | "Arrived: 0.6 m from it, facing it" |
| 80–90 | H (results card) | `semantic_map_vs_truth.png` or the README table + repo URL | "3 runs · precision 0.63 · recall 0.67 · GoTo 72 % · every failure analysed" |

**The GIF** (README top) is the 25-second highlight: shots C (fast), D, F
and G, with no title cards and no audio.

## 3. ffmpeg commands

```bash
# --- record the screen (X11) -- one command per raw clip; stop with q
ffmpeg -video_size 1366x768 -framerate 30 -f x11grab -i :0.0+0,0 \
       -c:v libx264 -preset ultrafast -crf 18 raw_C_explore.mkv

# --- speed up (no audio): 40x for the exploration, 2.5x for the GoTo drive
ffmpeg -i raw_C_explore.mkv -filter:v "setpts=PTS/40,fps=30" -an C_fast.mp4
ffmpeg -i raw_F_goto.mkv    -filter:v "setpts=PTS/2.5,fps=30" -an F_fast.mp4

# --- trim (start at 0:12, keep 26 s)
ffmpeg -ss 00:00:12 -t 26 -i C_fast.mp4 -c:v libx264 -crf 18 C_cut.mp4

# --- caption on a segment (bottom-centre, from t = 0 to 5 s of that clip)
ffmpeg -i C_cut.mp4 -vf "drawtext=text='SLAM + frontier exploration':fontcolor=white:\
fontsize=36:box=1:boxcolor=black@0.55:boxborderw=12:x=(w-text_w)/2:y=h-90:\
enable='between(t,0,5)'" -c:v libx264 -crf 18 C_cap.mp4

# --- title / results card from a PNG (4 s)
ffmpeg -loop 1 -t 4 -i title.png -vf "scale=1366:768:force_original_aspect_ratio=decrease,\
pad=1366:768:(ow-iw)/2:(oh-ih)/2:color=0x303030,fps=30,format=yuv420p" -c:v libx264 H_title.mp4

# --- concatenate (all clips must share size / fps / codec: re-encode if not)
printf "file '%s'\n" H_title.mp4 A.mp4 B.mp4 C_cap.mp4 D.mp4 E.mp4 F_cap.mp4 G.mp4 H_results.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c:v libx264 -crf 20 -pix_fmt yuv420p demo.mp4

# --- GIF: 25 s highlight, 12 fps, 800 px wide, two-pass palette (small and sharp)
ffmpeg -ss 14 -t 25 -i demo.mp4 \
       -vf "fps=12,scale=800:-1:flags=lanczos,palettegen=stats_mode=diff" -y palette.png
ffmpeg -ss 14 -t 25 -i demo.mp4 -i palette.png \
       -lavfi "fps=12,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
       -y docs/media/demo.gif
ls -lh docs/media/demo.gif     # aim for < 10 MB; if bigger: fps=10, scale=640, or a shorter -t
```

`tools/make_gif.sh demo.mp4 14 25` runs the two GIF commands.

## 4. Title card text

> **Semantic Object-Finding Robot**
> Explores an unknown house, learns where objects are, goes there when asked.
> ROS 2 Jazzy · Nav2 · slam_toolbox · YOLO11n on CPU (OpenVINO) · Gazebo Harmonic

## 5. Tips

- **Keep it fast.** Never show a static screen for longer than 3 s.
- **Speed up loading.** Record after the stack is up, or cut out the first 20 s
  (Gazebo / Nav2 start-up).
- **A clean voice take.** Use the TTS WAV:
  `ros2 launch semantic_nav_bringup commander.launch.py voice:=true input_wav:=<wav>`.
  Record your own voice separately and lay it over the cut.
- **Show something honest.** In the results card, a false positive in the
  `semantic_map_vs_truth.png` panel is a talking point, not a flaw to hide.
