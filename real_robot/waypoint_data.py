"""Explicit pose-to-waypoint conversion; no model, ROS or movement side effects.

Inputs must already share a verified world frame and timebase. Positions are
metres, yaw radians counterclockwise, body x forward and body y left. Validation
here checks numerical consistency, not reference quality or physical safety.
"""
import bisect
import json
import math
from pathlib import Path

from real_robot.files import sha256_file


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


class MissingReferenceCoverage(ValueError):
    """Requested time lies outside the reference poses; never extrapolated."""


def wrap_yaw(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def validate_offsets(offsets):
    if not isinstance(offsets, list) or len(offsets) != 8:
        raise ValueError('Explicit prediction_offsets_s must contain eight values')
    values = [number(v, 'prediction offset') for v in offsets]
    if values[0] < 0 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('Prediction offsets must be nonnegative and strictly increasing')
    return values


def validate_poses(poses):
    if not isinstance(poses, list) or len(poses) < 2:
        raise ValueError('At least two timestamped poses are required')
    rows = []
    for pose in poses:
        if not isinstance(pose, dict):
            raise ValueError('Pose must be an object')
        try:
            rows.append([number(pose[k], k) for k in ('timestamp_s', 'x_m', 'y_m', 'yaw_rad')])
        except KeyError as exc:
            raise ValueError('Missing pose field: ' + str(exc)) from exc
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('Pose timestamps must be strictly increasing')
    return rows


def relative_waypoints(poses, current_timestamp_s, prediction_offsets_s, *, max_pose_gap_s):
    """Interpolate measured/reference poses and return eight current-body poses.

    No extrapolation, clipping, command integration or padding of future labels.
    Interpolated yaw follows the shortest arc; collection must sample densely
    enough that rotation between samples is unambiguous. max_pose_gap_s is a
    caller-chosen quality limit, not an assumed property of the dataset.
    """
    gap = number(max_pose_gap_s, 'max pose gap')
    if gap <= 0:
        raise ValueError('max_pose_gap_s must be positive')
    return _relative_waypoints(validate_poses(poses), number(current_timestamp_s, 'current timestamp'),
                               validate_offsets(prediction_offsets_s), gap)


def _relative_waypoints(rows, current, offsets, gap):
    """relative_waypoints on already validated rows, offsets and gap limit."""
    times = [r[0] for r in rows]

    def at(t):
        if t < times[0] or t > times[-1]:
            raise MissingReferenceCoverage('Reference poses do not cover requested time; no extrapolation')
        i = bisect.bisect_left(times, t)
        if i < len(times) and times[i] == t:
            return rows[i][1:]
        left, right = rows[i-1], rows[i]
        interval = right[0]-left[0]
        if interval > gap:
            raise ValueError('Interpolation crosses an excessive pose gap')
        weight = (t-left[0])/interval
        return [left[1]+weight*(right[1]-left[1]),
                left[2]+weight*(right[2]-left[2]),
                wrap_yaw(left[3]+weight*wrap_yaw(right[3]-left[3]))]

    # Even exact query points must not hide a large unobserved interval between
    # current and final future time.
    end = current+offsets[-1]
    for left, right in zip(times, times[1:]):
        if right > current and left < end and right-left > gap:
            raise ValueError('Prediction horizon crosses an excessive pose gap')
    x0, y0, yaw0 = at(current)
    c, s = math.cos(yaw0), math.sin(yaw0)
    output = []
    for offset in offsets:
        x, y, yaw = at(current+offset)
        dx, dy = x-x0, y-y0
        output.append([c*dx+s*dy, -s*dx+c*dy, wrap_yaw(yaw-yaw0)])
    return output


def validate_training_example(example):
    """Validate the explicit v1 label envelope, not image or physical accuracy.

    Existing trainer reads images/current/instruction/waypoints/valid_mask.
    supervision is provenance for a future validated adapter; the trainer does
    not enforce it. Call this validator before passing such data to training.
    """
    if not isinstance(example, dict):
        raise ValueError('Example must be an object')
    if not isinstance(example.get('current'), str) or not example['current'].strip():
        raise ValueError('Current image path is required')
    images = example.get('images')
    if not isinstance(images, list) or len(images) > 31 or any(not isinstance(p, str) or not p.strip() for p in images):
        raise ValueError('images must contain at most 31 image paths')
    if not isinstance(example.get('instruction'), str) or not example['instruction'].strip():
        raise ValueError('Nonempty instruction is required')
    sup = example.get('supervision', {})
    if not isinstance(sup, dict) or sup.get('schema') != 'omtrack-waypoints-v1':
        raise ValueError('Explicit versioned supervision is required')
    validate_offsets(sup.get('prediction_offsets_s'))
    if sup.get('frame') != 'current_robot_forward_left_ccw' or sup.get('units') != ['m', 'm', 'rad']:
        raise ValueError('Unsupported waypoint frame or units')
    if sup.get('reference_type') not in ('measured_poses', 'verified_simulator_poses'):
        raise ValueError('Explicit pose reference required; commands are not measured poses')
    for key in ('session_id', 'reference_manifest_sha256', 'clock_alignment_record'):
        if not isinstance(sup.get(key), str) or not sup[key].strip():
            raise ValueError('Missing supervision provenance: ' + key)
    sha = sup['reference_manifest_sha256']
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('reference_manifest_sha256 must be a lowercase SHA256')
    number(sup.get('current_timestamp_s'), 'current timestamp')
    points = example.get('waypoints')
    if not isinstance(points, list) or len(points) != 8:
        raise ValueError('Eight explicit waypoint targets required')
    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError('Each waypoint must have three values')
        for value in point:
            number(value, 'waypoint')
    mask = example.get('valid_mask')
    if not isinstance(mask, list) or len(mask) != 8 or any(type(v) is not bool for v in mask) or not any(mask):
        raise ValueError('Eight boolean validity values with at least one valid point required')
    return True


def prepare_pose_session(session_manifest, *, prediction_offsets_s, max_pose_gap_s,
                         max_history_gap_s):
    """Validate a timestamped recording session and derive strict-history examples.

    Manifest v1 contains `frames` (RGB filename, timestamp_s, sha256), a separate
    `reference_file` with aligned world-frame poses, and explicit alignment
    provenance. `root` is the session directory. This converts annotations into
    trainer-compatible JSON objects; it does not certify the supplied poses.
    Incomplete future horizons are excluded and counted rather than extrapolated.
    """
    if not isinstance(session_manifest, dict) or session_manifest.get('schema') != 'omtrack-session-v1':
        raise ValueError('Expected schema omtrack-session-v1')
    root = Path(session_manifest.get('root', '.')).resolve()
    session_id = session_manifest.get('session_id')
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError('session_id is required')
    split = session_manifest.get('split')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Assign the whole session to train, validation or test before windowing')
    instruction = session_manifest.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty instruction is required')
    target_id = session_manifest.get('target_id')
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError('Annotated target_id is required')
    if session_manifest.get('reference_type') not in ('measured_poses', 'verified_simulator_poses'):
        raise ValueError('Reference must be timestamped poses, not commands or model outputs')
    if session_manifest.get('frame_transform_verified') is not True or session_manifest.get('clock_alignment_verified') is not True:
        raise ValueError('Frame transforms and camera/reference clock alignment need explicit review')
    alignment = session_manifest.get('clock_alignment_record')
    if not isinstance(alignment, str) or not alignment.strip():
        raise ValueError('Clock alignment evidence record is required')
    alignment_path = (root/alignment).resolve()
    if root not in alignment_path.parents or not alignment_path.is_file():
        raise ValueError('Alignment evidence must be a file within the session directory')
    reference = session_manifest.get('reference_file')
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError('Separate reference_file is required')
    reference_path = (root/reference).resolve()
    if root not in reference_path.parents or not reference_path.is_file():
        raise ValueError('Reference poses must be a file within the session directory')
    reference_sha = sha256_file(reference_path)
    reference_data = json.loads(reference_path.read_text())
    if reference_data.get('frame') != 'world_xy_ccw' or reference_data.get('units') != ['m', 'm', 'rad']:
        raise ValueError('Reference pose frame/units must be world_xy_ccw metres/radians')
    poses = validate_poses(reference_data.get('poses'))
    offsets = validate_offsets(prediction_offsets_s)
    max_pose_gap_s = number(max_pose_gap_s, 'max pose gap')
    max_history_gap_s = number(max_history_gap_s, 'max history gap')
    if max_pose_gap_s <= 0 or max_history_gap_s <= 0:
        raise ValueError('Pose and history gap limits must be positive')
    clock_offset = number(session_manifest.get('camera_to_pose_time_offset_s'), 'camera/pose clock offset')
    frames = session_manifest.get('frames')
    if not isinstance(frames, list) or len(frames) < 32:
        raise ValueError('At least 31 history frames and one current frame are required')
    checked = []
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError('Frame record must be an object')
        name = row.get('image')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('RGB image path is required')
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('RGB image must exist within its session: ' + name)
        sha = row.get('sha256')
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Frame SHA256 must be lowercase hex')
        if sha256_file(path) != sha:
            raise ValueError('RGB checksum mismatch: ' + name)
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError('Cannot decode RGB image: ' + name) from exc
        checked.append(dict(image=str(path), timestamp_s=number(row.get('timestamp_s'), 'frame timestamp'), sha256=sha))
    if any(b['timestamp_s'] <= a['timestamp_s'] for a, b in zip(checked, checked[1:])):
        raise ValueError('RGB timestamps must be strictly increasing')

    examples, rejected = [], {}
    def reject(reason):
        rejected[reason] = rejected.get(reason, 0)+1
    for index in range(31, len(checked)):
        window = checked[index-31:index+1]
        if any(b['timestamp_s']-a['timestamp_s'] > max_history_gap_s for a,b in zip(window,window[1:])):
            reject('history_gap'); continue
        current = checked[index]
        current_pose_time = current['timestamp_s']+clock_offset
        try:
            waypoints = _relative_waypoints(poses, current_pose_time, offsets, max_pose_gap_s)
        except MissingReferenceCoverage:
            reject('missing_future'); continue
        except ValueError:
            reject('pose_gap_or_invalid'); continue
        example = dict(images=[r['image'] for r in window[:-1]], current=current['image'],
            instruction=instruction, waypoints=waypoints, valid_mask=[True]*8,
            session_id=session_id, split=split,
            supervision=dict(schema='omtrack-waypoints-v1', prediction_offsets_s=offsets,
                frame='current_robot_forward_left_ccw', units=['m','m','rad'],
                reference_type=session_manifest['reference_type'], session_id=session_id,
                target_id=target_id, current_timestamp_s=current_pose_time,
                image_sha256=[r['sha256'] for r in window],
                reference_manifest_sha256=reference_sha,
                clock_alignment_record=str(alignment_path),
                clock_alignment_record_sha256=sha256_file(alignment_path),
                frame_transform_verified=True, clock_alignment_verified=True,
                max_pose_gap_s=max_pose_gap_s, max_history_gap_s=max_history_gap_s))
        validate_training_example(example)
        examples.append(example)
    return dict(schema='omtrack-session-v1', session_id=session_id, split=split,
                examples=examples, accepted_windows=len(examples), rejected_windows=rejected,
                reference_sha256=reference_sha,
                qualification='Numerical/schema validation only; supplied frame and clock reviews are assertions, not independently certified measurements.')
