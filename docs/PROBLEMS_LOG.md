# Problems log

Running record of every problem hit, its root cause, and the fix. Newest last.
Milestone reports pull their "Problems and fixes" section from here.

## M0 — Workspace setup

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | `colcon build --symlink-install` failed in semantic_nav_interfaces: `failed to create symbolic link ... existing path cannot be removed: Is a directory` | The package had first been built *without* `--symlink-install`, so `build/` held real copied directories where the symlink build wants links. | `rm -rf build install log` and rebuild. Rule: never switch install mode on an existing build dir; always use `--symlink-install`. |
| 2 | Warning `Unknown distribution option: 'tests_require'` for every Python package | A newer setuptools (79) in `~/.local` shadows the system one; `tests_require` was removed from setuptools. | Dropped `tests_require`, declared `extras_require={'test': ['pytest']}` (the current ROS 2 template). |
| 3 | `colcon test` ran `unittest` and reported `NO TESTS RAN` for the Python packages | After removing `tests_require`, colcon no longer knew the packages use pytest; it looks for pytest in `tests_require` or `extras_require['test']`. | Same fix as #2 — `extras_require['test']` makes colcon pick its pytest step. |
| 4 | `colcon test-result` warning: `Skipping build/semantic_nav_commander/package.xml: mismatched tag` | Description text `"go to <object>"` — `<object>` was parsed as an XML tag. | Reworded to `"go to OBJECT"`. XML text must not contain raw `<`. |
| 5 | flake8 E501 in `semantic_nav_perception/setup.py` | Description string > 99 chars (ament_flake8 limit). | Shortened the description. |
| 6 | ament_pep257 D213 on every node module docstring | ROS 2's pep257 config wants the multi-line summary to start on the line *after* the opening `"""`. | Moved the summary line down. |
| 7 | ament_pep257 D407/D413 on the launch file | A line reading exactly `Arguments:` is treated as a numpydoc section header needing a `---` underline. | Renamed to `Launch arguments:`. |
| 8 | Stopping `ros2 launch` printed `KeyboardInterrupt` tracebacks from `destroy_node()` | Ctrl-C in a terminal signals the whole process group, so each node gets SIGINT from the terminal *and* again from launch. The second one landed during cleanup. | Adopted the Jazzy demos pattern: `try: rclpy.init(); rclpy.spin(node) except (KeyboardInterrupt, ExternalShutdownException): pass`. A single SIGINT to launch now gives `process has finished cleanly` for all four nodes. (A group-wide double SIGINT may still show `exit code -2` — harmless, no traceback.) |
| 9 | pandoc / weasyprint not installed and `sudo` needs a password | No passwordless sudo for the agent. | Gitignored venv `.venv-docs` with `markdown` + `weasyprint` from pip, and `docs/tools/md2pdf.py` to render reports. |

## M1 — Simulation environment

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 10 | Needed a 640x480 @ 10 Hz camera, but TurtleBot4's OAK-D is fixed at 320x240 @ 30 Hz | `turtlebot4_description/urdf/sensors/oakd.urdf.xacro` hard-codes the values; the macro has no parameters. | Vendored `turtlebot4.urdf.xacro` + `oakd.urdf.xacro` into `semantic_nav_bringup/urdf/` (Apache-2.0, attribution kept) with `camera_width/height/rate` xacro args. |
| 11 | Camera sensor would always use ogre1 | `create3.urdf.xacro` embeds the gz Sensors system with `<render_engine>ogre</render_engine>`. | `sim.launch.py` runs xacro itself and substitutes the `render_engine` launch argument (default `ogre2`); it fails loudly if the tag count is not exactly 1. |
| 12 | With ogre1, dining table, dining chairs and fridge render solid black | Those Fuel models use PBR materials (albedo/normal/roughness maps) that the ogre1 backend does not support. | ogre2 is the default. ogre1 is only a fallback for GPU problems (it also runs ~10% faster). |
| 13 | Fuel's only TV model (`tv_65in_emissive`) would render untextured | It uses an Ogre material script, which Gazebo Harmonic does not support ("Gazebo does not support Ogre material scripts"). | Built a local `models/tv` (bezel + screen box with a PBR `albedo_map`/`emissive_map` of a generated picture). |
| 14 | No realistic potted plant on Fuel (only empty scanned pots or trees) | — | `scripts/generate_local_models.py` generates a tapered pot + 30 arching leaf blades (OBJ, double-sided). Whether stock YOLO detects it is to be checked in the perception milestone. |
| 15 | Real-time factor 0.55 in the furnished house vs 0.97 in the empty house | Chair, Sofa, Toilet and KitchenSink Fuel models are *dynamic* bodies with triangle-mesh collisions, so physics simulated them resting on the floor every step. | `<static>true</static>` inside each `<include>` (SDF lets an include override it). RTF 0.55 → 0.67. |
| 16 | RTF still 0.67 even with the camera effectively off | The RPLIDAR is a `gpu_lidar`: it renders the (now much more detailed) scene on every update, and upstream runs it at 62 Hz. A real RPLIDAR A1 spins at 5.5–10 Hz. | Vendored `rplidar.urdf.xacro` with a `lidar_rate` arg, default 10 Hz. RTF 0.67 → ~0.8. |
| 17 | RTF collapsed to 0.3 after teleporting the robot | The robot had been placed intersecting the couch; contact between the robot and a triangle-mesh collision is expensive every step. | Move it clear. Lesson: during teleop/navigation, pushing into mesh-collision furniture tanks RTF; Nav2's inflation will keep the robot off it later. |
| 18 | Keyboard teleop had no effect | TurtleBot4 on Jazzy expects `geometry_msgs/TwistStamped` on `/cmd_vel`; `teleop_twist_keyboard` publishes `Twist` by default. | Run it with `-p stamped:=true`. |
| 19 | `ros2 topic hz` under `timeout` printed nothing | SIGTERM killed the Python process before its block-buffered stdout (a pipe) was flushed. | Wrote `scripts/measure_sim_performance.py`, which measures RTF from `/clock` and topic rates in both wall and sim time. |
| 20 | After restarting, the new sim hung at "Requesting list of world names" | A previous `gz sim` server survived SIGTERM; two servers both serving world `house` on gz-transport confuse clients. | Kill leftovers with SIGKILL before relaunching. |
| 21 | Ported the TB4 camera static TF with the wrong axis order | The legacy `static_transform_publisher x y z yaw pitch roll` positional form is yaw-first, not roll-first. | Removed it: it only exists for the point cloud, which we don't bridge; images already carry `oakd_rgb_camera_optical_frame`. |
| 22 | `[gz_ros_control] Desired controller update period (0.001 s) is faster than the gazebo simulation period (0.003 s)` | Create3's `control.yaml` asks for 1000 Hz; the world steps at 3 ms (same as upstream TurtleBot4 worlds). | Harmless: the controller runs every physics step (333 Hz). Left as upstream. |
| 23 | flake8 I100 import-order failures in the new script and tests | `flake8-import-order` treats our package as third-party, so it must be sorted alphabetically with `yaml`. | Reordered; the script's post-`sys.path` import carries `# noqa: E402,I100`. |
