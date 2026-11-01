import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leader_follow.identity.lock_on import LeaderLock
from leader_follow.interfaces import CameraTrack
from leader_follow.perception.camera_people import parse_tracks

LEADER = np.array([1.0, 0.0, 0.0], np.float32)
OTHER = np.array([0.0, 1.0, 0.0], np.float32)


def person(track_id, feature, score=0.9):
    return CameraTrack(track_id, np.array([0, 0, 10, 30], np.float32), score, feature)


def propose(track_id):
    return lambda tracks: (next((t for t in tracks if t.track_id == track_id), None), 'proposed')


def nobody(_tracks):
    return None, 'nobody_proposed'


class LeaderLockTests(unittest.TestCase):
    def lock_on(self, lock, frames=3):
        for _ in range(frames):
            update = lock.update([person(1, LEADER), person(2, OTHER)], propose(1))
        return update

    def test_locks_after_consecutive_confirmations(self):
        lock = LeaderLock(acquire_frames=3)
        lock.update([person(1, LEADER)], propose(1))
        self.assertEqual((lock.state, lock.reason), ('SEARCHING', 'confirming_stable_identity'))
        update = self.lock_on(lock, frames=2)
        self.assertEqual(lock.state, 'LOCKED')
        self.assertTrue(update.valid(lock.state))
        self.assertEqual(update.selected.track_id, 1)

    def test_missing_leader_is_never_replaced_by_a_different_person(self):
        lock = LeaderLock(acquire_frames=3, lost_frames=4)
        self.lock_on(lock)
        for _ in range(10):
            # The other person is proposed, but does not match the leader's gallery.
            update = lock.update([person(2, OTHER)], propose(2))
            self.assertFalse(update.valid(lock.state))
        self.assertEqual((lock.state, lock.reason), ('LOST', 'candidate_appearance_differs'))

    def test_leader_under_a_new_track_id_is_recovered_after_confirmation(self):
        lock = LeaderLock(acquire_frames=3)
        self.lock_on(lock)
        lock.update([], nobody)
        self.assertEqual((lock.state, lock.reason), ('UNCERTAIN', 'target_missing'))
        for _ in range(3):
            update = lock.update([person(7, LEADER), person(2, OTHER)], propose(7))
        self.assertEqual((lock.state, lock.reason, update.selected.track_id), ('LOCKED', 'identity_recovered', 7))

    def test_appearance_change_drops_the_lock(self):
        lock = LeaderLock()
        self.lock_on(lock)
        lock.update([person(1, OTHER)], nobody)
        self.assertEqual((lock.state, lock.reason), ('UNCERTAIN', 'appearance_mismatch'))

    def test_gallery_is_bounded(self):
        lock = LeaderLock(gallery_size=4)
        self.lock_on(lock)
        for _ in range(20):
            lock.update([person(1, LEADER)], nobody)
        self.assertEqual(len(lock.gallery), 4)


class ParseTracksTests(unittest.TestCase):
    def test_rows_map_to_tracks_and_bad_indices_are_dropped(self):
        features = np.eye(2, dtype=np.float32)
        rows = np.array([[1, 2, 3, 4, 9, 0.8, 0, 1], [0, 0, 1, 1, 5, 0.5, 0, 7]], np.float32)
        tracks = parse_tracks(rows, features)
        self.assertEqual(len(tracks), 1)
        self.assertEqual((tracks[0].track_id, round(tracks[0].score, 3)), (9, 0.8))
        self.assertEqual(parse_tracks(np.zeros((1, 7), np.float32), features), [])
        np.testing.assert_array_equal(tracks[0].feature, features[1])


if __name__ == '__main__':
    unittest.main()
