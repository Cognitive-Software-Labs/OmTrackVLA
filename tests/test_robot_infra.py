import sys
import tempfile
import types
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.files import require_new_path, sha256_bytes, sha256_file
from robot.latency import percentiles
from robot.owned_processes import OwnedProcesses
from robot.ros_camera import is_jpeg, stamp_ns


class FileTests(unittest.TestCase):
    def test_file_and_bytes_digests_agree(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'frame.bin'
            path.write_bytes(b'x' * (5 * 1024 * 1024))
            self.assertEqual(sha256_file(path), sha256_bytes(path.read_bytes()))

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):
                require_new_path(td)
            require_new_path(Path(td) / 'new')


class LatencyTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(percentiles(list(range(101)))['p95'], 95.0)
        self.assertIsNone(percentiles([]))


class RosCameraTests(unittest.TestCase):
    def test_stamp_and_format(self):
        header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=2, nanosec=5))
        self.assertEqual(stamp_ns(header), 2_000_000_005)
        self.assertTrue(is_jpeg('rgb8; jpeg compressed bgr8'))
        self.assertFalse(is_jpeg('png'))


class OwnedProcessTests(unittest.TestCase):
    def test_stop_ends_only_spawned_children(self):
        with tempfile.TemporaryDirectory() as td:
            record = []
            owned = OwnedProcesses(Path(td), record)
            child = owned.spawn('sleeper', [sys.executable, '-c', 'import time; time.sleep(60)'])
            owned.stop(timeout=5)
            self.assertIsNotNone(child.poll())
            self.assertEqual(record[0]['name'], 'sleeper')
            self.assertTrue((Path(td) / 'sleeper.log').exists())


if __name__ == '__main__':
    unittest.main()
