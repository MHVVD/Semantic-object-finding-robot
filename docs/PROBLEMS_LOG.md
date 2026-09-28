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
