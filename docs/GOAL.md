# Goal

## Task

Develop a leader-following system for the lab Ridgeback that works in crowded
environments. The robot follows one person (the leader) and must:

- **keep the right person:** the lock survives passers-by crossing between
  robot and leader, short full occlusions and people who look similar;
- **never switch silently:** when the leader cannot be confirmed, the robot
  stops and searches instead of following someone else;
- **move safely among people:** it keeps a set following distance from the
  leader and a minimum clearance from bystanders;
- **be measurable:** each claim above is backed by a repeatable scenario and a
  metric ([EVALUATION.md](EVALUATION.md)).

## Decisions so far

| Decision | Choice | Consequence |
|---|---|---|
| How the leader is chosen | Lock onto the person in front of the robot at start; a language prompt is optional | No vision-language model is needed in the loop |
| First system | Modular baseline (lidar tracking, camera identity, classic planner) | A reference every learned method is compared against |
| OmTrackVLA | One candidate method, kept under `methods/omtrack_vla/` | Evaluated against the baseline, not the starting point |
| Research questions | Defined after the baseline has numbers | No method comparison is claimed before then |

## Robot

| Sensor / actuator | What it gives | ROS topic |
|---|---|---|
| 2 × 2D lidar, 270° each, front and rear | 360° coverage with overlap | `/r100_0160/sensors/lidar2d_0/scan`, `/r100_0160/sensors/lidar2d_1/scan` |
| RGB camera (front) | Compressed colour images | `/r100_0160/sensors/camera_0/color/image/compressed` |
| Omnidirectional base | Independent `vx`, `vy`, `wz`: can face the leader while side-stepping | `cmd_vel` (not yet published) |
| Onboard GPU | RTX 5060, 8 GB | — |

## Out of scope for now

- Following through doors, lifts or across floors.
- Multiple leaders or handing the lock to a new leader while moving.
- Training a model from scratch; learned parts start from public checkpoints.

## Open questions

- **Target crowd:** a lab or corridor with a few people, or dense public spaces?
  This sets the scenario densities and the speed limits.
- **Camera coverage:** a single front camera means identity can be confirmed
  only in front of the robot; lidar carries the track elsewhere.
- **Lidar mount height:** shin or torso height decides between a leg detector
  and a learned 2D-lidar person detector (DR-SPAAM).
