"""Analytical conversion checks; no measured model or robot results."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from real_robot.waypoint_data import (MissingReferenceCoverage, prepare_pose_session, relative_waypoints,
                                      validate_training_example)


OFFSETS = [0, .1, .2, .3, .5, .6, .7, .8]


def poses(fn):
    return [dict(timestamp_s=i/10, x_m=fn(i/10)[0], y_m=fn(i/10)[1], yaw_rad=fn(i/10)[2])
            for i in range(11)]


class WaypointDataTests(unittest.TestCase):
    def convert(self, rows, current=0, offsets=None):
        return relative_waypoints(rows, current, OFFSETS if offsets is None else offsets,
                                  max_pose_gap_s=.15)

    def test_stationary(self):
        self.assertTrue(all(all(abs(v) < 1e-12 for v in p) for p in self.convert(poses(lambda t:(4,2,.3)))))

    def test_nonuniform_forward_offsets(self):
        points = self.convert(poses(lambda t:(2*t,0,0)))
        for point, offset in zip(points, OFFSETS):
            self.assertAlmostEqual(point[0], 2*offset)
            self.assertAlmostEqual(point[1], 0)

    def test_current_frame_rotation(self):
        points = self.convert(poses(lambda t:(0,t,math.pi/2)), current=.1)
        self.assertAlmostEqual(points[-1][0], .8)
        self.assertAlmostEqual(points[-1][1], 0)

    def test_leftward_motion(self):
        self.assertAlmostEqual(self.convert(poses(lambda t:(0,t,0)))[-1][1], .8)

    def test_turning_arc(self):
        points = self.convert(poses(lambda t:(math.sin(t),1-math.cos(t),t)))
        self.assertAlmostEqual(points[-1][0], math.sin(.8))
        self.assertAlmostEqual(points[-1][1], 1-math.cos(.8))
        self.assertAlmostEqual(points[-1][2], .8)

    def test_wraparound_interpolation(self):
        rows = poses(lambda t:(0,0,math.atan2(math.sin(math.radians(179)+t*.1),math.cos(math.radians(179)+t*.1))))
        points = self.convert(rows, current=.05)
        self.assertAlmostEqual(points[-1][2], .08)

    def test_missing_future_rejected(self):
        with self.assertRaises(MissingReferenceCoverage):
            self.convert(poses(lambda t:(t,0,0)), current=.4)

    def test_pose_gap_rejected_even_at_exact_query_times(self):
        rows = poses(lambda t:(t,0,0))
        with self.assertRaises(ValueError):
            self.convert(rows[:2]+rows[5:])

    def test_duplicate_and_nonfinite_times_rejected(self):
        rows = poses(lambda t:(t,0,0))
        for bad in (rows[:1]+rows, [dict(rows[0], timestamp_s=float('nan'))]+rows[1:]):
            with self.assertRaises(ValueError):
                self.convert(bad)

    def test_invalid_offsets_rejected(self):
        rows = poses(lambda t:(t,0,0))
        for offsets in ([0]*8, [0,.1], [-.1,0,.1,.2,.3,.4,.5,.6]):
            with self.assertRaises(ValueError):
                self.convert(rows, offsets=offsets)

    def test_supervision_requires_explicit_pose_provenance(self):
        example = dict(current='current.jpg', images=[], instruction='Follow the target.',
            waypoints=[[0.,0.,0.] for _ in range(8)], valid_mask=[True]*8,
            supervision=dict(schema='omtrack-waypoints-v1', prediction_offsets_s=OFFSETS,
                frame='current_robot_forward_left_ccw', units=['m','m','rad'],
                reference_type='measured_poses', session_id='analytical-test',
                reference_manifest_sha256='0'*64, clock_alignment_record='test fixture only',
                current_timestamp_s=0))
        self.assertTrue(validate_training_example(example))
        for mutate in (
            lambda e:e['supervision'].update(reference_type='velocity_commands'),
            lambda e:e.pop('waypoints'),
            lambda e:e.update(valid_mask=[1]*8),
            lambda e:e['supervision'].pop('prediction_offsets_s'),
        ):
            bad = copy.deepcopy(example);mutate(bad)
            with self.assertRaises(ValueError):
                validate_training_example(bad)

    def test_session_windows_use_strict_history_and_versioned_poses(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);Image.new('RGB',(2,2),(1,2,3)).save(root/'frame.png')
            image_sha=hashlib.sha256((root/'frame.png').read_bytes()).hexdigest()
            (root/'alignment.txt').write_text('reviewed clock alignment test evidence\n')
            pose_data=dict(frame='world_xy_ccw',units=['m','m','rad'],
                poses=[dict(timestamp_s=i/10,x_m=i/10,y_m=0.,yaw_rad=0.) for i in range(351)])
            (root/'poses.json').write_text(json.dumps(pose_data))
            session=dict(schema='omtrack-session-v1',root=str(root),session_id='pilot-1',split='train',
                instruction='Follow target.',target_id='person-a',reference_type='measured_poses',
                reference_file='poses.json',frame_transform_verified=True,clock_alignment_verified=True,
                clock_alignment_record='alignment.txt',camera_to_pose_time_offset_s=0,
                frames=[dict(image='frame.png',timestamp_s=i/10,sha256=image_sha) for i in range(34)])
            result=prepare_pose_session(session_manifest=session,prediction_offsets_s=OFFSETS,
                                        max_pose_gap_s=.11,max_history_gap_s=.11)
            self.assertEqual(result['accepted_windows'],3)
            self.assertEqual(len(result['examples'][0]['images']),31)
            self.assertEqual(len(result['examples'][0]['waypoints']),8)
            self.assertEqual(result['examples'][0]['supervision']['session_id'],'pilot-1')
            self.assertEqual(result['rejected_windows'],{})
            pose_data['poses']=pose_data['poses'][:36]
            (root/'poses.json').write_text(json.dumps(pose_data))
            result=prepare_pose_session(session_manifest=session,prediction_offsets_s=OFFSETS,
                                        max_pose_gap_s=.11,max_history_gap_s=.11)
            self.assertEqual(result['accepted_windows'],0)
            self.assertEqual(result['rejected_windows'],{'missing_future':3})

    def test_session_rejects_modified_rgb_and_missing_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'frame.jpg').write_bytes(b'not-an-image')
            session=dict(schema='omtrack-session-v1',root=str(root),session_id='bad',split='train',
                instruction='Follow target.',target_id='a',reference_type='measured_poses',
                reference_file='poses.json',frame_transform_verified=True,clock_alignment_verified=True,
                clock_alignment_record='alignment.txt',camera_to_pose_time_offset_s=0,
                frames=[dict(image='frame.jpg',timestamp_s=i,sha256='0'*64) for i in range(32)])
            with self.assertRaises(ValueError):
                prepare_pose_session(session_manifest=session,prediction_offsets_s=OFFSETS,
                                     max_pose_gap_s=.15,max_history_gap_s=.5)
            session['split']=''
            with self.assertRaises(ValueError):
                prepare_pose_session(session_manifest=session,prediction_offsets_s=OFFSETS,
                                     max_pose_gap_s=.15,max_history_gap_s=.5)


if __name__ == '__main__':
    unittest.main()
