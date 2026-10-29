# Real Ridgeback camera evaluation

Use the existing Ridgeback ROS 2 Jazzy camera publisher. These tools subscribe
to camera images and return or display predicted waypoints; they do not publish
robot movement. Keep the physical E-stop pressed during stationary collection.
Results are written under ignored `outputs/` so private images stay local.

## Record one bounded case

Start the local model service in one terminal:

```bash
bash real_robot/start_omtrack_ridgeback.sh
```

After visually confirming the intended camera scene, use a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
/usr/bin/python3 -m real_robot.record_live_model_test \
  --output outputs/cases/black-shirt-left-001 \
  --prompt 'Follow the person wearing a black T-shirt.' --seconds 25
```

Choose a new output path for each independent take. `clip.json` records the
prompt, input timestamps and JPEG hashes; `predictions.jsonl` contains every
paired model reply. The recorder requests fresh inferences and resets model
history on the first frame. It never silently overwrites a prior clip. Stop
the model service with Ctrl-C when recording is done.

For camera-only clips, use `real_robot.record_prompt_clip --output NEW_CLIP_DIR
--scene 'what is actually in view'` instead. `--scene` is required: describe
clothing, positions and movement as observed, not as model labels. To compare
two instructions on exactly the same saved frames, run
`real_robot.evaluate_prompt_swap --clip CLIP_DIR --prompt-a TEXT_A --prompt-b
TEXT_B --output NEW_OUTPUT_DIR`. It replays A/B/B/A with a fresh run boundary.

Record one matching person, an empty room, a person who does not match the
instruction, and two-person/object cases as independent sessions. Label
which person matches, whether the target is visible or ambiguous, and the
camera-side position before interpreting model outputs. No person in view is
not the same as a failed target tracker. Model-only mode has no target ID or
absence confidence: a finite waypoint is only a numeric output, not a match.

Stationary RGB clips support prompt and target-presence diagnostics. Physical
waypoint error and waypoint fine-tuning need separately verified timestamped
reference robot poses, frame transforms and prediction offsets; see
`WAYPOINT_DATA_CONTRACT.md`. Do not treat model predictions as ground truth.

## Optional live RViz map

`real_robot/rviz/live_mapping.yaml` uses the front Ridgeback lidar, odometry,
and a dedicated `omtrack_display_map` frame. Both front and rear live scans
appear in `real_robot/rviz/ridgeback_live_map.rviz`. The map updates from the
area the sensors currently see; a stationary robot cannot map unseen rooms.
This launches SLAM display data, not Nav2 or exploration movement.

```bash
source /opt/ros/jazzy/setup.bash
ros2 run slam_toolbox async_slam_toolbox_node --ros-args \
  --params-file real_robot/rviz/live_mapping.yaml \
  -r __ns:=/omtrackvla_display \
  -r /tf:=/r100_0160/tf -r /tf_static:=/r100_0160/tf_static
```

In another terminal, activate the mapper and open RViz:

```bash
source /opt/ros/jazzy/setup.bash
ros2 lifecycle set /omtrackvla_display/slam_toolbox configure
ros2 lifecycle set /omtrackvla_display/slam_toolbox activate
rviz2 -d real_robot/rviz/ridgeback_live_map.rviz --ros-args \
  -r /tf:=/r100_0160/tf -r /tf_static:=/r100_0160/tf_static
```

The mapper publishes `/map`. Avoid starting a second mapper when an existing
system is already publishing `/map`. Stop the mapper and RViz with Ctrl-C in
their terminals.
