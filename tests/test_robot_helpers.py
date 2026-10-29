import socket
import sys
import tempfile
import threading
import types
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_robot.files import require_new_path, sha256_bytes, sha256_file
from real_robot.latency import percentiles
from real_robot.model_client import ModelClient, has_finite_waypoints, require_fresh_waypoints
from real_robot.owned_processes import OwnedProcesses
from real_robot.protocol import receive_message, send_message
from real_robot.ros_camera import is_jpeg, stamp_ns

POINTS = [[0.1 * i, 0.0, 0.0] for i in range(8)]


def serve_once(reply_for):
    """Fake inference_server: answer one request on a free loopback port."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)

    def run():
        conn, _ = listener.accept()
        with conn:
            send_message(conn, reply_for(receive_message(conn)))
        listener.close()

    threading.Thread(target=run, daemon=True).start()
    return listener.getsockname()[1]


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


class WaypointTests(unittest.TestCase):
    def test_shape_and_finiteness(self):
        self.assertTrue(has_finite_waypoints(POINTS))
        self.assertFalse(has_finite_waypoints(POINTS[:7]))
        self.assertFalse(has_finite_waypoints(POINTS[:7] + [[0, float('nan'), 0]]))
        self.assertFalse(has_finite_waypoints(None))
        self.assertFalse(has_finite_waypoints([['0', '0', '0']] * 8))
        self.assertFalse(has_finite_waypoints([[True, 0, 0]] * 8))

    def test_cached_reply_is_not_fresh(self):
        require_fresh_waypoints({'planner_updated': True, 'trajectory': POINTS})
        with self.assertRaises(RuntimeError):
            require_fresh_waypoints({'planner_updated': False, 'trajectory': POINTS})


class ModelClientTests(unittest.TestCase):
    def test_round_trip(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'], 'frame_id': req['frame_id'],
                                       'mode': 'model_only', 'prompt_seen': req['prompt'],
                                       'reset_seen': req['reset_target'], 'trajectory': POINTS})
        with ModelClient(port) as client:
            reply = client.infer(b'\xff\xd8jpeg', 'Follow me.', 7, True, frame_id='f7', mode='model_only')
        self.assertEqual((reply['prompt_seen'], reply['reset_seen']), ('Follow me.', True))

    def test_pre_encoded_image_is_sent_unchanged(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'], 'seen': req['image_b64']})
        with ModelClient(port) as client:
            self.assertEqual(client.infer('anBlZw==', 'Follow me.', 1, False)['seen'], 'anBlZw==')

    def test_mismatched_request_id_is_rejected(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'] + 1})
        with ModelClient(port) as client, self.assertRaises(ValueError):
            client.infer(b'jpeg', 'Follow me.', 1, False)

    def test_wrong_server_mode_is_rejected(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'], 'mode': 'full'})
        with ModelClient(port) as client, self.assertRaises(ValueError):
            client.infer(b'jpeg', 'Follow me.', 1, False, mode='model_only')


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
