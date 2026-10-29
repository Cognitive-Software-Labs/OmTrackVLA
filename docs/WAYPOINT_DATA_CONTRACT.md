# Waypoint data contract and legacy compatibility audit

Verified by source inspection on 2026-10-27. No robot movement or new training
was performed. This document distinguishes code behaviour from the unverified
training provenance of the released checkpoint.

## Verified legacy behaviour

| Layer | Contract actually implemented |
| --- | --- |
| Network | Eight triples, scaled by `alpha_task`; the local checkpoint config uses XY scale 2 and no output tanh |
| Text/vision | Text embeddings and pooled visual tokens condition an action token, followed by a waypoint MLP; no selected person ID is returned |
| Converter | For horizon 8, integrate nine commands, append after each integration; stored `trajectory` has nine post-action points |
| Dataset class (`tracking_dataset.py`) | Read `waypoints` if present; otherwise derive labels from `actions`. The converter's `trajectory` field is ignored |
| Action-derived targets | Start at origin, integrate command indices 0 through 7 into nine positions, then use rounded linspace indices `[0,1,2,3,5,6,7,8]` for eight targets |
| Default action-derived timing | `DataConfig.default_dt=0.1`; resulting offsets `[0,.1,.2,.3,.5,.6,.7,.8]` seconds. `integrate_actions_to_waypoints` itself defaults to `dt=0.2`, so a caller that omits `dt` gets different timing |
| Rollout/local command conversion | Use predicted waypoint index 1 and divide components by .1 by default; these are proposed commands, not measured executed velocities |
| Live history | Append current coarse features before forming a 31-slot history; pad with earliest available coarse features; current fine features are separate |
| Dataset history | `images` excludes current in converter examples. On a short history the dataset loop fills earlier slots with current features before seeing earlier images. This differs from live history construction |

The converter docstring says it includes the origin, but its actual loop does
not. Preserved files should not be silently rewritten to match that docstring.
The stored nine-point target is not directly comparable to the eight-point model
output. Short-history construction is another compatibility question to test
before assuming train/inference parity.

## Simulation actions are not measured pose references

The rollout loop observes an image and computes an action before `env.step`,
then saves that requested action as `base_velocity`. It saves distance/facing
metrics after stepping. It does not save a timestamped realised robot pose trace.

The currently checked-in STT config sets `ctrl_freq: 40` and robot action scales
15 longitudinal, 10 lateral and 6.28 angular. The non-cylinder action clips input
components to [-1,1], applies its scales and integrates against simulator control
timing. Navigation-mesh collision filtering/sliding may change actual movement.
These values describe current source/configuration, not proven settings used to
produce the old sample videos. An MP4's 10 fps does not prove a .1-second physical
action duration. The old sample metadata does not resolve this discrepancy.

Therefore:

- We can reproduce the public trainer's mathematical command-derived labels.
- We cannot claim they are verified physical motion in metres/radians.
- We cannot establish the released checkpoint's complete historical target timing
  solely from its configuration and public sample files.
- Physical error scoring and supervised adaptation need a declared reference
  convention and usable motion provenance, not guessed conversion factors.

## Versioned pose-reference format for new data

The new numerical utilities in `real_robot/waypoint_data.py` require:

- Poses `timestamp_s`, `x_m`, `y_m`, `yaw_rad` in one verified planar world frame.
- Metres and radians; body x forward, y left, positive yaw counterclockwise.
- Shared clock/alignment for image and pose timestamps. Prefer time relative to
  session start to avoid unnecessary floating-point loss on large epoch values.
- A declared list of exactly eight nonnegative, strictly increasing prediction
  offsets. No implicit dt, linspace resampling or dropped middle target.
- A caller-selected maximum pose-sample gap based on sensor timing/quality.
- Actual reference coverage through the requested horizon: no extrapolation or
  repeated last action as a substitute for missing labels.

The compatibility offsets `[0,.1,.2,.3,.5,.6,.7,.8]` may be used to reproduce the
legacy target schedule only when explicitly declared. They are not prescribed as
a corrected universal schedule. Changing the schedule is a training/runtime
contract decision; this work does not silently change the pretrained model.

For each future pose at `t+offset`, compute translation in the current robot frame:

```text
dx = world_x(t+offset) - world_x(t)
dy = world_y(t+offset) - world_y(t)
forward = cos(yaw(t))*dx + sin(yaw(t))*dy
left    = -sin(yaw(t))*dx + cos(yaw(t))*dy
heading = wrap(yaw(t+offset) - yaw(t))
```

Interpolation uses linear XY and shortest-arc yaw. Sampling must be sufficiently
dense that yaw changes between records are unambiguous. Gap checks include the
entire horizon, even when every requested offset falls on an exact pose sample.
These numerical checks do not verify odometry drift, calibration or physical
reference quality; those require independent collection/evaluation evidence.

## Explicit trainer example envelope

The current trainer already reads these top-level fields:

- `images`: at most 31 ordered history image paths.
- `current`: current image path.
- `instruction`: exact text.
- `waypoints`: eight explicit triples, to avoid implicit command-derived targets.
- `valid_mask`: eight booleans, with valid labels justified by reference coverage.

The new envelope validator also requires `supervision` metadata:

- `schema`: `omtrack-waypoints-v1`.
- `prediction_offsets_s`: eight explicit offsets.
- `frame`: `current_robot_forward_left_ccw`; `units`: `["m","m","rad"]`.
- `reference_type`: `measured_poses` or `verified_simulator_poses`.
- `session_id`, `current_timestamp_s`, `reference_manifest_sha256` and
  `clock_alignment_record`.

The trainer does not currently enforce or consume that metadata. The numerical
validator is a preparation utility, not a complete training ingestion pipeline.
The full adapter must additionally verify image paths/checksums/timestamps,
reference provenance/quality, history policy, unique-target annotations and
session-level split isolation. A provenance string alone proves none of these.

The pose utility currently rejects incomplete coverage rather than manufacturing
masked future labels. Independently prepared masked examples may be accepted by
the envelope validator, but their masks and reference validity must be audited.

## Verification and remaining work

Analytical unit checks cover stationary/forward/sideways motion, rotated current
frame, a turning arc, yaw wraparound/interpolation, missing future poses, excessive
gaps, duplicate/nonfinite timestamps, invalid offsets and incomplete provenance.
These are mathematical fixtures, not model predictions or real robot trials.

The session adapter writes JSONL examples plus a separate audit report through
`scripts/prepare_pose_session.py`. It hashes and decodes every RGB file, requires
an explicit whole-session split and provenance files, and excludes examples with
insufficient history, reference coverage or pose sampling. These checks validate
the supplied manifest's numerical/schema consistency only: the booleans claiming
frame-transform and clock-alignment review must be supported by the referenced
evidence and an independent reviewer. The trainer itself still does not enforce
the supervision envelope.

A read-only check at `scripts/validate_checkpoint_compat.py` verifies strict
state loading and output parity between the released model and current trainer
architecture using deterministic synthetic feature tensors. It does not verify
visual accuracy, physical label semantics or training memory. The current trainer
can freeze Qwen with `--freeze_llm`, but does not yet initialize from the released
safetensors checkpoint. Actual supervised updates remain gated on reference
quality, independent splits, a trainer initialization path and the measured
failure being targeted.

Controlled basket/position/clothing clips can evaluate target-selection
behaviour, but moving reference demonstrations are separately needed for
waypoint fine-tuning. See `REAL_CAMERA_EVALUATION.md` for the recording protocol.
